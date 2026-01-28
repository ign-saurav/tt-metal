# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
PI0 Performant Runner with Trace + 2CQ Support

This module provides a performant runner for PI0 model inference using:
- Trace: Reduces dispatch overhead by pre-recording operations
- 2 Command Queues: Overlaps input transfers with computation
  - CQ0: Model operations (trace execution)
  - CQ1: Input/output transfers

Usage:
    runner = PI0PerformantRunner(model, device)
    runner.capture_trace_2cq(inputs)
    output = runner.run(inputs)
    runner.release()
"""

from typing import List, Tuple
import torch
import ttnn

from models.experimental.pi0.tt.ttnn_pi0_model import PI0ModelTTNN


class PI0PerformantRunner:
    """
    Performant runner for PI0 model with trace+2cq optimization.

    This runner implements:
    1. Trace capture: Records model execution for replay
    2. Dual command queues: Overlaps I/O with computation
    3. Event synchronization: Coordinates between queues
    """

    CQ_OPS = 0  # Command queue for operations
    CQ_IO = 1  # Command queue for I/O transfers

    def __init__(
        self,
        model: PI0ModelTTNN,
        device: ttnn.Device,
    ):
        """
        Initialize performant runner.

        Args:
            model: PI0ModelTTNN instance
            device: TTNN device
        """
        self.model = model
        self.device = device

        # Trace state
        self.trace_id = None
        self.trace_captured = False

        # Event synchronization
        self.op_event = None
        self.write_event = None

        # Input tensors (will be allocated during capture)
        self.lang_tokens_ttnn = None
        self.lang_masks_ttnn = None
        self.state_ttnn = None
        self.lang_tokens_spec = None
        self.lang_masks_spec = None
        self.state_spec = None

        # DRAM tensors for persistent storage
        self.lang_tokens_dram = None
        self.lang_masks_dram = None
        self.state_dram = None

        # L1 tensors for operations
        self.lang_tokens_l1 = None
        self.lang_masks_l1 = None
        self.state_l1 = None

        # Trace execution state
        self.prefix_kv_cache = None
        self.timesteps = None
        self.num_steps = None
        self.timesteps_ttnn = None
        self.x_t_ttnn = None
        self.t_tensor_persistent = None
        self.action_output_tensor = None

    def _prepare_inputs_for_trace(
        self,
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
        state: torch.Tensor,
    ) -> Tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
        """
        Prepare inputs for trace capture/execution.

        Converts PyTorch tensors to TTNN format and allocates on device.
        """
        # Convert to TTNN format
        lang_tokens_ttnn = ttnn.from_torch(
            lang_tokens,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
        )
        lang_masks_ttnn = ttnn.from_torch(
            lang_masks.float(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
        )
        state_ttnn = ttnn.from_torch(
            state,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
        )

        return lang_tokens_ttnn, lang_masks_ttnn, state_ttnn

    def capture_trace_2cq(
        self,
        images: List[torch.Tensor],
        img_masks: List[torch.Tensor],
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
        state: torch.Tensor,
    ):
        """
        Capture trace with 2CQ setup.

        This method:
        1. Runs warmup iterations to configure JIT operations
        2. Sets up DRAM tensors for persistent storage
        3. Captures trace of model execution
        4. Sets up event synchronization

        Args:
            images: List of input images
            img_masks: Image masks
            lang_tokens: Language tokens
            lang_masks: Language masks
            state: Robot state
        """
        # Prepare inputs and store specs
        lang_tokens_ttnn, lang_masks_ttnn, state_ttnn = self._prepare_inputs_for_trace(lang_tokens, lang_masks, state)

        # Store specs for later allocation
        self.lang_tokens_spec = lang_tokens_ttnn.spec
        self.lang_masks_spec = lang_masks_ttnn.spec
        self.state_spec = state_ttnn.spec

        # Warmup run 1: Configure JIT operations
        print("  Warmup run 1: Configuring JIT operations...")
        _ = self.model.sample_actions(
            images=images,
            img_masks=img_masks,
            lang_tokens=lang_tokens,
            lang_masks=lang_masks,
            state=state,
        )
        ttnn.synchronize_device(self.device)

        # Warmup run 2: Optimized execution
        print("  Warmup run 2: Optimized execution...")
        _ = self.model.sample_actions(
            images=images,
            img_masks=img_masks,
            lang_tokens=lang_tokens,
            lang_masks=lang_masks,
            state=state,
        )
        ttnn.synchronize_device(self.device)

        # Setup DRAM tensors for persistent storage
        print("  Setting up DRAM tensors...")
        dram_mem_config = ttnn.DRAM_MEMORY_CONFIG
        self.lang_tokens_dram = ttnn.to_memory_config(lang_tokens_ttnn, dram_mem_config)
        self.lang_masks_dram = ttnn.to_memory_config(lang_masks_ttnn, dram_mem_config)
        self.state_dram = ttnn.to_memory_config(state_ttnn, dram_mem_config)

        # Convert DRAM to L1 for operations
        self.lang_tokens_l1 = ttnn.to_memory_config(self.lang_tokens_dram, ttnn.L1_MEMORY_CONFIG)
        self.lang_masks_l1 = ttnn.to_memory_config(self.lang_masks_dram, ttnn.L1_MEMORY_CONFIG)
        self.state_l1 = ttnn.to_memory_config(self.state_dram, ttnn.L1_MEMORY_CONFIG)

        # Capture trace
        print("  Capturing trace...")
        # Initialize events (before trace capture)
        self.op_event = ttnn.record_event(self.device, self.CQ_OPS)

        # Convert PyTorch tensors to TTNN host tensors (BEFORE trace capture)
        lang_tokens_host = ttnn.from_torch(lang_tokens, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
        lang_masks_host = ttnn.from_torch(lang_masks.float(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        state_host = ttnn.from_torch(state, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

        # Copy inputs to DRAM on CQ_IO (BEFORE trace capture)
        ttnn.wait_for_event(self.CQ_IO, self.op_event)
        ttnn.copy_host_to_device_tensor(lang_tokens_host, self.lang_tokens_dram, cq_id=self.CQ_IO)
        ttnn.copy_host_to_device_tensor(lang_masks_host, self.lang_masks_dram, cq_id=self.CQ_IO)
        ttnn.copy_host_to_device_tensor(state_host, self.state_dram, cq_id=self.CQ_IO)
        self.write_event = ttnn.record_event(self.device, self.CQ_IO)
        ttnn.wait_for_event(self.CQ_OPS, self.write_event)

        # Convert DRAM to L1 (BEFORE trace capture)
        self.lang_tokens_l1 = ttnn.to_memory_config(self.lang_tokens_dram, ttnn.L1_MEMORY_CONFIG)
        self.lang_masks_l1 = ttnn.to_memory_config(self.lang_masks_dram, ttnn.L1_MEMORY_CONFIG)
        self.state_l1 = ttnn.to_memory_config(self.state_dram, ttnn.L1_MEMORY_CONFIG)

        # Record event before trace capture
        self.op_event = ttnn.record_event(self.device, self.CQ_OPS)

        # Compute prefix embeddings BEFORE trace capture (involves from_torch for images)
        print("  Computing prefix embeddings (before trace capture)...")
        prefix_embs, _, _ = self.model.embed_prefix(images, img_masks, self.lang_tokens_l1, self.lang_masks_l1)

        # Forward prefix through VLM and cache KV (BEFORE trace capture)
        print("  Forwarding prefix through VLM (before trace capture)...")
        _, prefix_kv_cache = self.model.backbone.forward_vlm(prefix_embs, use_cache=True)
        self.prefix_kv_cache = prefix_kv_cache

        # Pre-compute timesteps and initial noise (BEFORE trace capture)
        batch_size = lang_tokens.shape[0]
        num_steps = self.model.denoise_config.num_steps
        self.timesteps = [1.0 - i / num_steps for i in range(num_steps + 1)]
        self.num_steps = num_steps

        # Pre-compute timesteps tensor
        pad_steps = ((num_steps + 31) // 32) * 32
        timestep_indices = ttnn.arange(0, pad_steps, 1, device=self.device, dtype=ttnn.bfloat16)
        timestep_indices = ttnn.to_layout(timestep_indices, ttnn.TILE_LAYOUT)
        timestep_values = ttnn.multiply(timestep_indices, -1.0 / num_steps)
        self.timesteps_ttnn = ttnn.add(timestep_values, 1.0)
        self.timesteps_ttnn = ttnn.reshape(self.timesteps_ttnn, (1, pad_steps))
        ttnn.deallocate(timestep_indices)
        ttnn.deallocate(timestep_values)

        # Sample initial noise
        x_t_torch = torch.randn(batch_size, self.model.config.action_horizon, self.model.config.action_dim)
        self.x_t_ttnn = ttnn.from_torch(
            x_t_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        # Warmup one denoising step to compile kernels BEFORE trace capture
        print("  Warming up denoising step to compile kernels...")
        i = 0
        t_tensor = ttnn.slice(self.timesteps_ttnn, [0, i], [batch_size, i + 1])
        t_tensor = ttnn.reshape(t_tensor, (batch_size,))
        suffix_embs, _, _, _ = self.model.embed_suffix(self.state_l1, self.x_t_ttnn, t_tensor)
        expert_output, _ = self.model.backbone.forward_expert(suffix_embs, past_key_values=self.prefix_kv_cache)
        if not self.model.config.pi05:
            action_output = ttnn.slice(
                expert_output, [0, 1, 0], [expert_output.shape[0], expert_output.shape[1], expert_output.shape[2]]
            )
        else:
            action_output = expert_output
        velocity = self.model.suffix_embedding.project_output(action_output)
        t = self.timesteps[i]
        t_next = self.timesteps[i + 1]
        dt = t_next - t
        velocity_scaled = ttnn.mul(velocity, dt)
        self.x_t_ttnn = ttnn.add(self.x_t_ttnn, velocity_scaled, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.synchronize_device(self.device)
        ttnn.deallocate(t_tensor)

        # Reset x_t_ttnn for trace capture
        x_t_torch = torch.randn(batch_size, self.model.config.action_horizon, self.model.config.action_dim)
        self.x_t_ttnn = ttnn.from_torch(
            x_t_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        # Create persistent timestep tensor for trace execution
        t_tensor = ttnn.slice(self.timesteps_ttnn, [0, i], [batch_size, i + 1])
        t_tensor = ttnn.reshape(t_tensor, (batch_size,))
        self.t_tensor_persistent = t_tensor

        # Begin trace capture of one denoising step (NO from_torch or tensor creation calls here!)
        print("  Beginning trace capture of denoising step...")
        self.trace_id = ttnn.begin_trace_capture(self.device, cq_id=self.CQ_OPS)

        # Run one denoising step to capture trace (all kernels compiled, tensors cached)
        suffix_embs, _, _, _ = self.model.embed_suffix(self.state_l1, self.x_t_ttnn, self.t_tensor_persistent)
        expert_output, _ = self.model.backbone.forward_expert(suffix_embs, past_key_values=self.prefix_kv_cache)
        if not self.model.config.pi05:
            self.action_output_tensor = ttnn.slice(
                expert_output, [0, 1, 0], [expert_output.shape[0], expert_output.shape[1], expert_output.shape[2]]
            )
        else:
            self.action_output_tensor = expert_output

        # End trace capture (before project_output, dt multiplication, and Euler step)
        # These are excluded because project_output creates new tensor and dt changes per step
        ttnn.end_trace_capture(self.device, self.trace_id, cq_id=self.CQ_OPS)

        self.trace_captured = True
        print("  ✅ Trace captured successfully")

    def run(
        self,
        images: List[torch.Tensor],
        img_masks: List[torch.Tensor],
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
        state: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run inference using trace+2cq.

        This method:
        1. Copies inputs to DRAM on CQ_IO (overlapped with previous execution)
        2. Waits for previous operations to complete
        3. Converts DRAM to L1
        4. Executes trace on CQ_OPS (non-blocking)
        5. Returns output

        Args:
            images: List of input images
            img_masks: Image masks
            lang_tokens: Language tokens
            lang_masks: Language masks
            state: Robot state

        Returns:
            Sampled actions (PyTorch tensor)
        """
        # Wait for previous operations to complete (if any)
        if self.op_event is not None:
            ttnn.wait_for_event(self.CQ_IO, self.op_event)

        # Convert PyTorch tensors to TTNN host tensors
        lang_tokens_host = ttnn.from_torch(lang_tokens, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
        lang_masks_host = ttnn.from_torch(lang_masks.float(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        state_host = ttnn.from_torch(state, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

        # Copy inputs to DRAM on CQ_IO (overlapped with previous execution)
        ttnn.copy_host_to_device_tensor(lang_tokens_host, self.lang_tokens_dram, cq_id=self.CQ_IO)
        ttnn.copy_host_to_device_tensor(lang_masks_host, self.lang_masks_dram, cq_id=self.CQ_IO)
        ttnn.copy_host_to_device_tensor(state_host, self.state_dram, cq_id=self.CQ_IO)
        self.write_event = ttnn.record_event(self.device, self.CQ_IO)

        # Wait for writes to complete before operations
        ttnn.wait_for_event(self.CQ_OPS, self.write_event)

        if self.trace_captured and self.trace_id is not None:
            # Convert DRAM to L1 (reuse tensors)
            self.lang_tokens_l1 = ttnn.to_memory_config(
                self.lang_tokens_dram, ttnn.L1_MEMORY_CONFIG, output_tensor=self.lang_tokens_l1
            )
            self.lang_masks_l1 = ttnn.to_memory_config(
                self.lang_masks_dram, ttnn.L1_MEMORY_CONFIG, output_tensor=self.lang_masks_l1
            )
            self.state_l1 = ttnn.to_memory_config(self.state_dram, ttnn.L1_MEMORY_CONFIG, output_tensor=self.state_l1)

            # Recompute prefix embeddings (images may change between runs)
            # This is done BEFORE trace execution to avoid from_torch during trace
            prefix_embs, _, _ = self.model.embed_prefix(images, img_masks, self.lang_tokens_l1, self.lang_masks_l1)
            _, self.prefix_kv_cache = self.model.backbone.forward_vlm(prefix_embs, use_cache=True)

            # Sample initial noise
            batch_size = lang_tokens.shape[0]
            x_t_torch = torch.randn(batch_size, self.model.config.action_horizon, self.model.config.action_dim)
            self.x_t_ttnn = ttnn.from_torch(
                x_t_torch,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )

            # Record event before trace execution
            self.op_event = ttnn.record_event(self.device, self.CQ_OPS)

            # Execute denoising loop using trace for each step
            for i in range(self.num_steps):
                t = self.timesteps[i]
                t_next = self.timesteps[i + 1]
                dt = t_next - t

                # Update persistent timestep tensor with current step's value
                t_tensor_slice = ttnn.slice(self.timesteps_ttnn, [0, i], [batch_size, i + 1])
                t_tensor_slice = ttnn.reshape(t_tensor_slice, (batch_size,))
                ttnn.copy(t_tensor_slice, self.t_tensor_persistent)
                ttnn.deallocate(t_tensor_slice)

                # Execute trace (replays operations up to action_output)
                ttnn.execute_trace(self.device, self.trace_id, cq_id=self.CQ_OPS, blocking=False)
                ttnn.synchronize_device(self.device)

                # Run operations not in trace: project_output, dt multiplication, and Euler step
                velocity = self.model.suffix_embedding.project_output(self.action_output_tensor)
                velocity_scaled = ttnn.mul(velocity, dt)
                self.x_t_ttnn = ttnn.add(self.x_t_ttnn, velocity_scaled, memory_config=ttnn.L1_MEMORY_CONFIG)

            # Convert final output to PyTorch
            output = ttnn.to_torch(self.x_t_ttnn)
        else:
            # No trace capture - use regular execution with 2CQ I/O overlap
            # The 2CQ benefit is that we overlapped the I/O transfers with previous execution
            # Convert DRAM to L1 for model execution
            lang_tokens_l1 = ttnn.to_memory_config(self.lang_tokens_dram, ttnn.L1_MEMORY_CONFIG)
            lang_masks_l1 = ttnn.to_memory_config(self.lang_masks_dram, ttnn.L1_MEMORY_CONFIG)
            state_l1 = ttnn.to_memory_config(self.state_dram, ttnn.L1_MEMORY_CONFIG)

            # Record event before model execution
            self.op_event = ttnn.record_event(self.device, self.CQ_OPS)

            # Run model (model will do from_torch internally, but I/O is already done)
            output = self.model.sample_actions(
                images=images,
                img_masks=img_masks,
                lang_tokens=lang_tokens,  # Pass PyTorch tensors as model expects
                lang_masks=lang_masks,
                state=state,
            )

            # Synchronize to ensure completion
            ttnn.synchronize_device(self.device)

            # Clean up L1 tensors
            ttnn.deallocate(lang_tokens_l1)
            ttnn.deallocate(lang_masks_l1)
            ttnn.deallocate(state_l1)

        return output

    def release(self):
        """Release trace resources."""
        if self.trace_id is not None:
            ttnn.release_trace(self.device, self.trace_id)
            self.trace_id = None
            self.trace_captured = False
