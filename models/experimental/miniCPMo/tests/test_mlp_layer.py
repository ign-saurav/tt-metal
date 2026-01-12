# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Test MLP Layer in MiniCPMo Qwen Model

This test validates the MLP layer implementation by:
1. Loading MiniCPM weights and extracting MLP weights for a single layer
2. Running TT MLP forward pass with test inputs
3. Comparing output with PyTorch reference MLP output

Test cases cover both demo scenarios:
- demo_image: Vision understanding with image embeddings
- demo_audio_understanding: Audio understanding with audio embeddings

Usage:
    pytest test_mlp_layer.py -v -s

    # Run specific test:
    pytest test_mlp_layer.py::test_mlp_layer_image_demo -v -s
    pytest test_mlp_layer.py::test_mlp_layer_audio_demo -v -s
"""

import pytest
import torch
import torch.nn as nn
import ttnn
import os
from pathlib import Path
from loguru import logger
from typing import Optional, Tuple


# --- Configuration ---
MODEL_PATH = "openbmb/MiniCPM-o-2_6"

# Qwen2 MLP hidden dim ratio (intermediate_size / hidden_size)
# For MiniCPM-o: hidden_size=3584, intermediate_size=18944
HIDDEN_SIZE = 3584
INTERMEDIATE_SIZE = 18944


class PyTorchQwen2MLP(nn.Module):
    """
    Reference PyTorch Qwen2 MLP implementation.

    Architecture:
        - gate_proj (w1): hidden_size -> intermediate_size
        - up_proj (w3): hidden_size -> intermediate_size
        - down_proj (w2): intermediate_size -> hidden_size

    Forward: down_proj(silu(gate_proj(x)) * up_proj(x))
    """

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)  # w1
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)  # w3
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)  # w2
        self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

    def load_weights(self, state_dict: dict, layer_num: int = 0, prefix: str = "layers"):
        """
        Load weights from state dict.

        Handles both tt_transformers format (w1, w2, w3) and HuggingFace format.
        """
        # Try tt_transformers format first
        w1_key = f"{prefix}.{layer_num}.feed_forward.w1.weight"
        w2_key = f"{prefix}.{layer_num}.feed_forward.w2.weight"
        w3_key = f"{prefix}.{layer_num}.feed_forward.w3.weight"

        # Alternative HuggingFace format
        hf_w1_key = f"model.layers.{layer_num}.mlp.gate_proj.weight"
        hf_w2_key = f"model.layers.{layer_num}.mlp.down_proj.weight"
        hf_w3_key = f"model.layers.{layer_num}.mlp.up_proj.weight"

        if w1_key in state_dict:
            logger.info(f"Loading weights from tt_transformers format")
            self.gate_proj.weight.data = state_dict[w1_key].clone()
            self.down_proj.weight.data = state_dict[w2_key].clone()
            self.up_proj.weight.data = state_dict[w3_key].clone()
        elif hf_w1_key in state_dict:
            logger.info(f"Loading weights from HuggingFace format")
            self.gate_proj.weight.data = state_dict[hf_w1_key].clone()
            self.down_proj.weight.data = state_dict[hf_w2_key].clone()
            self.up_proj.weight.data = state_dict[hf_w3_key].clone()
        else:
            raise ValueError(f"Could not find MLP weights for layer {layer_num}. " f"Tried keys: {w1_key}, {hf_w1_key}")


def calculate_pcc(expected: torch.Tensor, actual: torch.Tensor) -> float:
    """Calculate Pearson Correlation Coefficient between two tensors."""
    expected_flat = expected.float().flatten()
    actual_flat = actual.float().flatten()

    expected_mean = expected_flat.mean()
    actual_mean = actual_flat.mean()

    expected_centered = expected_flat - expected_mean
    actual_centered = actual_flat - actual_mean

    numerator = (expected_centered * actual_centered).sum()
    denominator = torch.sqrt((expected_centered**2).sum() * (actual_centered**2).sum())

    if denominator < 1e-10:
        return 1.0 if numerator < 1e-10 else 0.0

    return (numerator / denominator).item()


def find_input_file(filename: str) -> Optional[str]:
    """Find input file in various locations."""
    locations = [
        filename,
        Path.cwd() / filename,
        Path.home() / "ign_tt" / "forked" / "tt-metal" / filename,
        Path(__file__).parent / filename,
        Path(__file__).parent.parent / filename,
    ]
    for loc in locations:
        if Path(loc).exists():
            return str(loc)
    return None


def create_random_input(
    batch_size: int, seq_len: int, hidden_size: int, dtype: torch.dtype = torch.bfloat16
) -> torch.Tensor:
    """Create random input tensor for MLP testing."""
    torch.manual_seed(42)
    return torch.randn(batch_size, seq_len, hidden_size, dtype=dtype)


def load_image_demo_input() -> Tuple[torch.Tensor, str]:
    """
    Load or create input for image demo scenario.

    Image demo processes visual tokens from SigLIP encoder + resampler.
    Returns tensor of shape [batch, seq_len, hidden_size].
    """
    # Try to load saved image embeddings
    img_embd_path = find_input_file("img_qwen_input_embds.pt")
    if img_embd_path:
        logger.info(f"Loading image embeddings from: {img_embd_path}")
        embeddings = torch.load(img_embd_path, map_location="cpu")
        return embeddings, "saved"

    # Create synthetic image-like input
    # Vision tokens typically have 32-64 tokens after resampling
    logger.info("Creating synthetic image demo input")
    batch_size = 1
    seq_len = 64  # Typical resampled vision token count
    embeddings = create_random_input(batch_size, seq_len, HIDDEN_SIZE)
    return embeddings, "synthetic"


def load_audio_demo_input() -> Tuple[torch.Tensor, str]:
    """
    Load or create input for audio understanding demo scenario.

    Audio demo processes Whisper-encoded audio through audio projector.
    Returns tensor of shape [batch, seq_len, hidden_size].
    """
    # Try to load saved audio embeddings
    aud_embd_path = find_input_file("aud_qwen_input_embds.pt")
    if aud_embd_path:
        logger.info(f"Loading audio embeddings from: {aud_embd_path}")
        embeddings = torch.load(aud_embd_path, map_location="cpu")
        return embeddings, "saved"

    # Create synthetic audio-like input
    # Audio tokens can be longer due to temporal nature
    logger.info("Creating synthetic audio demo input")
    batch_size = 1
    seq_len = 256  # Typical audio token sequence
    embeddings = create_random_input(batch_size, seq_len, HIDDEN_SIZE)
    return embeddings, "synthetic"


@pytest.fixture
def mesh_device(request):
    """Fixture to get the mesh device for TT operations."""
    num_devices = getattr(request, "param", 1)
    device = ttnn.open_device(device_id=0)
    yield device
    ttnn.close_device(device)


@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_mlp_layer_image_demo(mesh_device):
    """
    Test MLP layer with image demo scenario inputs.

    This validates the MLP layer used during vision understanding:
    - Input: Visual embeddings from SigLIP + Resampler
    - Processing: Single MLP layer forward pass
    - Comparison: TT output vs PyTorch reference
    """
    logger.info("=" * 60)
    logger.info("Testing MLP Layer - Image Demo Scenario")
    logger.info("=" * 60)

    # Set HF_MODEL environment variable
    if not os.environ.get("HF_MODEL"):
        os.environ["HF_MODEL"] = MODEL_PATH

    # 1. Load inputs
    logger.info("\n1. Loading Image Demo Inputs...")
    inputs, input_source = load_image_demo_input()

    if inputs.dim() == 2:
        inputs = inputs.unsqueeze(0)

    batch_size, seq_len, hidden_size = inputs.shape
    logger.info(f"   Input shape: {inputs.shape} (source: {input_source})")
    logger.info(f"   Input dtype: {inputs.dtype}")

    # Limit sequence length for memory
    MAX_SEQ_LEN = 128
    if seq_len > MAX_SEQ_LEN:
        logger.info(f"   Truncating sequence from {seq_len} to {MAX_SEQ_LEN}")
        inputs = inputs[:, :MAX_SEQ_LEN, :]
        seq_len = MAX_SEQ_LEN

    # 2. Load weights
    logger.info("\n2. Loading MLP Weights...")
    from models.experimental.miniCPMo.tt.minicpm_weight_bridge import MiniCPMWeightBridge

    bridge = MiniCPMWeightBridge(MODEL_PATH)
    qwen_weights = bridge.get_qwen_weights()
    logger.info(f"   Loaded {len(qwen_weights)} weight tensors")

    # 3. Create PyTorch reference MLP
    logger.info("\n3. Creating PyTorch Reference MLP...")
    layer_num = 0  # Test first layer
    pt_mlp = PyTorchQwen2MLP(hidden_size=HIDDEN_SIZE, intermediate_size=INTERMEDIATE_SIZE)
    pt_mlp.load_weights(qwen_weights, layer_num=layer_num)
    pt_mlp = pt_mlp.to(torch.bfloat16).eval()
    logger.info(f"   PyTorch MLP: gate_proj={pt_mlp.gate_proj.weight.shape}")

    # 4. Run PyTorch forward
    logger.info("\n4. Running PyTorch Forward...")
    with torch.no_grad():
        pt_output = pt_mlp(inputs.to(torch.bfloat16))
    logger.info(f"   PT output shape: {pt_output.shape}")
    logger.info(f"   PT output range: [{pt_output.min():.4f}, {pt_output.max():.4f}]")

    # 5. Create TT MLP
    logger.info("\n5. Creating TT MLP...")
    from models.experimental.miniCPMo.tt_transformers.common import create_tt_model

    tt_model_args, tt_model, tt_kv_cache, _ = create_tt_model(
        mesh_device=mesh_device,
        instruct=False,
        max_batch_size=1,
        optimizations=None,
        max_seq_len=1024,
        paged_attention_config=None,
        dtype=ttnn.bfloat8_b,
        state_dict=qwen_weights,
        dummy_weights=False,
        num_layers=1,  # Only load first layer for MLP test
    )

    # Get the MLP from the first decoder layer
    tt_mlp = tt_model.layers[0].feed_forward
    logger.info(f"   TT MLP layer created")

    # 6. Run TT forward
    logger.info("\n6. Running TT Forward...")

    # Convert inputs to TT tensor
    tt_input = ttnn.from_torch(
        inputs.unsqueeze(1).to(torch.bfloat16),  # [batch, 1, seq, hidden]
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    # Run MLP forward with "prefill" mode
    tt_output = tt_mlp.forward(tt_input, mode="prefill")

    # Convert back to torch
    tt_output_torch = ttnn.to_torch(tt_output).float()
    ttnn.deallocate(tt_input)
    ttnn.deallocate(tt_output)

    # Reshape output
    while tt_output_torch.dim() > 3:
        tt_output_torch = tt_output_torch.squeeze(1) if tt_output_torch.shape[1] == 1 else tt_output_torch.squeeze(0)

    logger.info(f"   TT output shape: {tt_output_torch.shape}")
    logger.info(f"   TT output range: [{tt_output_torch.min():.4f}, {tt_output_torch.max():.4f}]")

    # 7. Compare outputs
    logger.info("\n7. Comparing Outputs...")

    # Ensure shapes match
    pt_output_np = pt_output.float()
    if tt_output_torch.shape != pt_output_np.shape:
        logger.warning(f"   Shape mismatch: TT={tt_output_torch.shape}, PT={pt_output_np.shape}")
        # Try to match shapes
        if tt_output_torch.dim() == 4 and pt_output_np.dim() == 3:
            tt_output_torch = tt_output_torch.squeeze(0)

    # Calculate PCC
    pcc = calculate_pcc(pt_output_np, tt_output_torch)
    logger.info(f"   PCC: {pcc:.6f}")

    # Calculate max absolute difference
    max_diff = (pt_output_np - tt_output_torch).abs().max().item()
    logger.info(f"   Max absolute difference: {max_diff:.6f}")

    # Calculate mean absolute difference
    mean_diff = (pt_output_np - tt_output_torch).abs().mean().item()
    logger.info(f"   Mean absolute difference: {mean_diff:.6f}")

    logger.info("=" * 60)

    # Assertions
    assert pcc > 0.95, f"PCC {pcc:.6f} is below threshold 0.95"
    logger.info("✅ MLP Layer Image Demo Test PASSED")


@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_mlp_layer_audio_demo(mesh_device):
    """
    Test MLP layer with audio understanding demo scenario inputs.

    This validates the MLP layer used during audio understanding:
    - Input: Audio embeddings from Whisper + Audio Projector
    - Processing: Single MLP layer forward pass
    - Comparison: TT output vs PyTorch reference
    """
    logger.info("=" * 60)
    logger.info("Testing MLP Layer - Audio Understanding Demo Scenario")
    logger.info("=" * 60)

    # Set HF_MODEL environment variable
    if not os.environ.get("HF_MODEL"):
        os.environ["HF_MODEL"] = MODEL_PATH

    # 1. Load inputs
    logger.info("\n1. Loading Audio Demo Inputs...")
    inputs, input_source = load_audio_demo_input()

    if inputs.dim() == 2:
        inputs = inputs.unsqueeze(0)

    batch_size, seq_len, hidden_size = inputs.shape
    logger.info(f"   Input shape: {inputs.shape} (source: {input_source})")
    logger.info(f"   Input dtype: {inputs.dtype}")

    # Limit sequence length for memory
    MAX_SEQ_LEN = 256
    if seq_len > MAX_SEQ_LEN:
        # Take last tokens to preserve suffix instructions
        start_idx = seq_len - MAX_SEQ_LEN
        logger.info(f"   Truncating: taking tokens [{start_idx}:{seq_len}] (last {MAX_SEQ_LEN} of {seq_len})")
        inputs = inputs[:, start_idx:, :]
        seq_len = MAX_SEQ_LEN

    # 2. Load weights
    logger.info("\n2. Loading MLP Weights...")
    from models.experimental.miniCPMo.tt.minicpm_weight_bridge import MiniCPMWeightBridge

    bridge = MiniCPMWeightBridge(MODEL_PATH)
    qwen_weights = bridge.get_qwen_weights()
    logger.info(f"   Loaded {len(qwen_weights)} weight tensors")

    # 3. Create PyTorch reference MLP
    logger.info("\n3. Creating PyTorch Reference MLP...")
    layer_num = 0  # Test first layer
    pt_mlp = PyTorchQwen2MLP(hidden_size=HIDDEN_SIZE, intermediate_size=INTERMEDIATE_SIZE)
    pt_mlp.load_weights(qwen_weights, layer_num=layer_num)
    pt_mlp = pt_mlp.to(torch.bfloat16).eval()
    logger.info(f"   PyTorch MLP: gate_proj={pt_mlp.gate_proj.weight.shape}")

    # 4. Run PyTorch forward
    logger.info("\n4. Running PyTorch Forward...")
    with torch.no_grad():
        pt_output = pt_mlp(inputs.to(torch.bfloat16))
    logger.info(f"   PT output shape: {pt_output.shape}")
    logger.info(f"   PT output range: [{pt_output.min():.4f}, {pt_output.max():.4f}]")

    # 5. Create TT MLP
    logger.info("\n5. Creating TT MLP...")
    from models.experimental.miniCPMo.tt_transformers.common import create_tt_model

    tt_model_args, tt_model, tt_kv_cache, _ = create_tt_model(
        mesh_device=mesh_device,
        instruct=False,
        max_batch_size=1,
        optimizations=None,
        max_seq_len=1024,
        paged_attention_config=None,
        dtype=ttnn.bfloat8_b,
        state_dict=qwen_weights,
        dummy_weights=False,
        num_layers=1,  # Only load first layer for MLP test
    )

    # Get the MLP from the first decoder layer
    tt_mlp = tt_model.layers[0].feed_forward
    logger.info(f"   TT MLP layer created")

    # 6. Run TT forward
    logger.info("\n6. Running TT Forward...")

    # Convert inputs to TT tensor
    tt_input = ttnn.from_torch(
        inputs.unsqueeze(1).to(torch.bfloat16),  # [batch, 1, seq, hidden]
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    # Run MLP forward with "prefill" mode for longer sequences
    tt_output = tt_mlp.forward(tt_input, mode="prefill")

    # Convert back to torch
    tt_output_torch = ttnn.to_torch(tt_output).float()
    ttnn.deallocate(tt_input)
    ttnn.deallocate(tt_output)

    # Reshape output
    while tt_output_torch.dim() > 3:
        tt_output_torch = tt_output_torch.squeeze(1) if tt_output_torch.shape[1] == 1 else tt_output_torch.squeeze(0)

    logger.info(f"   TT output shape: {tt_output_torch.shape}")
    logger.info(f"   TT output range: [{tt_output_torch.min():.4f}, {tt_output_torch.max():.4f}]")

    # 7. Compare outputs
    logger.info("\n7. Comparing Outputs...")

    # Ensure shapes match
    pt_output_np = pt_output.float()
    if tt_output_torch.shape != pt_output_np.shape:
        logger.warning(f"   Shape mismatch: TT={tt_output_torch.shape}, PT={pt_output_np.shape}")
        # Try to match shapes
        if tt_output_torch.dim() == 4 and pt_output_np.dim() == 3:
            tt_output_torch = tt_output_torch.squeeze(0)

    # Calculate PCC
    pcc = calculate_pcc(pt_output_np, tt_output_torch)
    logger.info(f"   PCC: {pcc:.6f}")

    # Calculate max absolute difference
    max_diff = (pt_output_np - tt_output_torch).abs().max().item()
    logger.info(f"   Max absolute difference: {max_diff:.6f}")

    # Calculate mean absolute difference
    mean_diff = (pt_output_np - tt_output_torch).abs().mean().item()
    logger.info(f"   Mean absolute difference: {mean_diff:.6f}")

    logger.info("=" * 60)

    # Assertions
    assert pcc > 0.95, f"PCC {pcc:.6f} is below threshold 0.95"
    logger.info("✅ MLP Layer Audio Understanding Demo Test PASSED")


@pytest.mark.parametrize("mesh_device", [1], indirect=True)
@pytest.mark.parametrize("mode", ["prefill", "decode"])
def test_mlp_layer_modes(mesh_device, mode):
    """
    Test MLP layer in both prefill and decode modes.

    This ensures the MLP works correctly in both operation modes:
    - prefill: Processing multiple tokens at once (initial prompt)
    - decode: Processing single token at a time (generation)
    """
    logger.info("=" * 60)
    logger.info(f"Testing MLP Layer - Mode: {mode}")
    logger.info("=" * 60)

    # Set HF_MODEL environment variable
    if not os.environ.get("HF_MODEL"):
        os.environ["HF_MODEL"] = MODEL_PATH

    # 1. Create test input based on mode
    logger.info("\n1. Creating Test Inputs...")
    if mode == "prefill":
        batch_size, seq_len = 1, 64
    else:  # decode
        batch_size, seq_len = 1, 32  # Decode mode uses padded 32-token blocks

    inputs = create_random_input(batch_size, seq_len, HIDDEN_SIZE)
    logger.info(f"   Input shape: {inputs.shape}")

    # 2. Load weights
    logger.info("\n2. Loading MLP Weights...")
    from models.experimental.miniCPMo.tt.minicpm_weight_bridge import MiniCPMWeightBridge

    bridge = MiniCPMWeightBridge(MODEL_PATH)
    qwen_weights = bridge.get_qwen_weights()

    # 3. Create PyTorch reference MLP
    logger.info("\n3. Creating PyTorch Reference MLP...")
    layer_num = 0
    pt_mlp = PyTorchQwen2MLP(hidden_size=HIDDEN_SIZE, intermediate_size=INTERMEDIATE_SIZE)
    pt_mlp.load_weights(qwen_weights, layer_num=layer_num)
    pt_mlp = pt_mlp.to(torch.bfloat16).eval()

    # 4. Run PyTorch forward
    logger.info("\n4. Running PyTorch Forward...")
    with torch.no_grad():
        pt_output = pt_mlp(inputs.to(torch.bfloat16))
    logger.info(f"   PT output shape: {pt_output.shape}")

    # 5. Create TT MLP
    logger.info("\n5. Creating TT MLP...")
    from models.experimental.miniCPMo.tt_transformers.common import create_tt_model

    tt_model_args, tt_model, tt_kv_cache, _ = create_tt_model(
        mesh_device=mesh_device,
        instruct=False,
        max_batch_size=1,
        optimizations=None,
        max_seq_len=1024,
        paged_attention_config=None,
        dtype=ttnn.bfloat8_b,
        state_dict=qwen_weights,
        dummy_weights=False,
        num_layers=1,
    )

    tt_mlp = tt_model.layers[0].feed_forward

    # 6. Run TT forward
    logger.info(f"\n6. Running TT Forward (mode={mode})...")

    tt_input = ttnn.from_torch(
        inputs.unsqueeze(1).to(torch.bfloat16),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    tt_output = tt_mlp.forward(tt_input, mode=mode)

    tt_output_torch = ttnn.to_torch(tt_output).float()
    ttnn.deallocate(tt_input)
    ttnn.deallocate(tt_output)

    # Reshape output
    while tt_output_torch.dim() > 3:
        tt_output_torch = tt_output_torch.squeeze(1) if tt_output_torch.shape[1] == 1 else tt_output_torch.squeeze(0)

    logger.info(f"   TT output shape: {tt_output_torch.shape}")

    # 7. Compare outputs
    logger.info("\n7. Comparing Outputs...")
    pt_output_np = pt_output.float()

    if tt_output_torch.shape != pt_output_np.shape:
        if tt_output_torch.dim() == 4 and pt_output_np.dim() == 3:
            tt_output_torch = tt_output_torch.squeeze(0)

    pcc = calculate_pcc(pt_output_np, tt_output_torch)
    logger.info(f"   PCC: {pcc:.6f}")

    logger.info("=" * 60)

    assert pcc > 0.90, f"PCC {pcc:.6f} is below threshold 0.90 for mode={mode}"
    logger.info(f"✅ MLP Layer Mode Test PASSED (mode={mode})")


@pytest.mark.parametrize("mesh_device", [1], indirect=True)
@pytest.mark.parametrize(
    "test_case",
    [
        # Case 1: Prefill mode with seq_len=128 (from terminal: shape [1, 1, 128, 3584])
        {
            "name": "prefill_seq128",
            "mode": "prefill",
            "seq_len": 128,
            "description": "Prefill mode - normal values, shape [1, 1, 128, 3584]",
        },
        # Case 2: Decode mode with seq_len=32 (from terminal: shape [1, 1, 32, 3584])
        {
            "name": "decode_seq32",
            "mode": "decode",
            "seq_len": 32,
            "description": "Decode mode - shape [1, 1, 32, 3584] with padding",
        },
        # Case 3: Prefill mode with seq_len=384 (from demo_audio: L1 buffer overflow)
        # This tests the circular buffer overflow issue during prefill with larger sequences
        {
            "name": "prefill_seq384",
            "mode": "prefill",
            "seq_len": 384,
            "description": "Prefill mode - larger sequence causing L1 buffer overflow",
        },
    ],
    ids=lambda x: x["name"],
)
def test_mlp_layer_demo_cases(mesh_device, test_case):
    """
    Parameterized test for MLP layer with specific failing cases from demo_image.py.

    Test cases from terminal output:
    1. Prefill: shape [1, 1, 128, 3584] - normal input values
    2. Decode: shape [1, 1, 32, 3584] - first row normal, padding rows zeros

    The decode case had exploded values (1e+17) in padding rows indicating
    upstream corruption, not MLP issues.
    """
    mode = test_case["mode"]
    seq_len = test_case["seq_len"]

    logger.info("=" * 60)
    logger.info(f"Testing MLP Layer - {test_case['description']}")
    logger.info("=" * 60)

    # Set HF_MODEL environment variable
    if not os.environ.get("HF_MODEL"):
        os.environ["HF_MODEL"] = MODEL_PATH

    # 1. Create test input matching the specific case
    logger.info(f"\n1. Creating Test Input (mode={mode}, seq_len={seq_len})...")
    batch_size = 1
    torch.manual_seed(42)

    if mode == "decode":
        # Decode: first row normal, rest zeros (simulating single-token with padding)
        inputs = torch.zeros(batch_size, seq_len, HIDDEN_SIZE, dtype=torch.bfloat16)
        inputs[:, 0, :] = torch.randn(batch_size, HIDDEN_SIZE, dtype=torch.bfloat16) * 0.5
        logger.info(f"   First row range: [{inputs[:, 0, :].min():.4f}, {inputs[:, 0, :].max():.4f}]")
        logger.info(f"   Padding rows: zeros")
    else:
        # Prefill: all rows have normal values
        inputs = torch.randn(batch_size, seq_len, HIDDEN_SIZE, dtype=torch.bfloat16) * 0.5
        logger.info(f"   Input range: [{inputs.min():.4f}, {inputs.max():.4f}]")

    logger.info(f"   Input shape: {inputs.shape} -> will be [1, 1, {seq_len}, {HIDDEN_SIZE}]")

    # 2. Load weights
    logger.info("\n2. Loading MLP Weights...")
    from models.experimental.miniCPMo.tt.minicpm_weight_bridge import MiniCPMWeightBridge

    bridge = MiniCPMWeightBridge(MODEL_PATH)
    qwen_weights = bridge.get_qwen_weights()

    # 3. Create PyTorch reference MLP
    logger.info("\n3. Creating PyTorch Reference MLP...")
    layer_num = 0
    pt_mlp = PyTorchQwen2MLP(hidden_size=HIDDEN_SIZE, intermediate_size=INTERMEDIATE_SIZE)
    pt_mlp.load_weights(qwen_weights, layer_num=layer_num)
    pt_mlp = pt_mlp.to(torch.bfloat16).eval()

    # 4. Run PyTorch forward
    logger.info("\n4. Running PyTorch Forward...")
    with torch.no_grad():
        pt_output = pt_mlp(inputs.to(torch.bfloat16))
    logger.info(f"   PT output shape: {pt_output.shape}")
    logger.info(f"   PT output range: [{pt_output.min():.4f}, {pt_output.max():.4f}]")

    # 5. Create TT MLP
    logger.info("\n5. Creating TT MLP...")
    from models.experimental.miniCPMo.tt_transformers.common import create_tt_model

    tt_model_args, tt_model, tt_kv_cache, _ = create_tt_model(
        mesh_device=mesh_device,
        instruct=False,
        max_batch_size=1,
        optimizations=None,
        max_seq_len=1024,
        paged_attention_config=None,
        dtype=ttnn.bfloat8_b,
        state_dict=qwen_weights,
        dummy_weights=False,
        num_layers=1,
    )

    tt_mlp = tt_model.layers[0].feed_forward

    # 6. Run TT forward
    logger.info(f"\n6. Running TT Forward (mode={mode})...")

    tt_input = ttnn.from_torch(
        inputs.unsqueeze(1).to(torch.bfloat16),  # [1, 1, seq_len, 3584]
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    logger.info(f"   TT input shape: {tt_input.shape}")

    # Run MLP forward
    tt_output = tt_mlp.forward(tt_input, mode=mode)

    tt_output_torch = ttnn.to_torch(tt_output).float()
    ttnn.deallocate(tt_input)
    ttnn.deallocate(tt_output)

    # Reshape output
    while tt_output_torch.dim() > 3:
        tt_output_torch = tt_output_torch.squeeze(1) if tt_output_torch.shape[1] == 1 else tt_output_torch.squeeze(0)

    logger.info(f"   TT output shape: {tt_output_torch.shape}")
    logger.info(f"   TT output range: [{tt_output_torch.min():.4f}, {tt_output_torch.max():.4f}]")

    # Check for NaN/Inf/exploded values
    has_nan = torch.isnan(tt_output_torch).any()
    has_inf = torch.isinf(tt_output_torch).any()
    max_val = tt_output_torch.abs().max().item()
    has_exploded = max_val > 1e10

    logger.info(f"   Has NaN: {has_nan}, Has Inf: {has_inf}, Max value: {max_val:.4e}")

    # 7. Compare outputs
    logger.info("\n7. Comparing Outputs...")
    pt_output_np = pt_output.float()

    if tt_output_torch.shape != pt_output_np.shape:
        if tt_output_torch.dim() == 4 and pt_output_np.dim() == 3:
            tt_output_torch = tt_output_torch.squeeze(0)

    pcc = calculate_pcc(pt_output_np, tt_output_torch)
    max_diff = (pt_output_np - tt_output_torch).abs().max().item()

    logger.info(f"   PCC: {pcc:.6f}")
    logger.info(f"   Max absolute difference: {max_diff:.6f}")

    logger.info("=" * 60)

    # Assertions
    assert not has_nan, f"[{test_case['name']}] TT output contains NaN values"
    assert not has_inf, f"[{test_case['name']}] TT output contains Inf values"
    assert not has_exploded, f"[{test_case['name']}] Output has exploded values (max={max_val:.4e})"

    pcc_threshold = 0.85 if mode == "decode" else 0.90
    assert pcc > pcc_threshold, f"[{test_case['name']}] PCC {pcc:.6f} is below threshold {pcc_threshold}"

    logger.info(f"✅ MLP Layer Test PASSED: {test_case['name']}")


@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_mlp_layer_multiple_layers(mesh_device):
    """
    Test MLP layers across multiple decoder layers.

    This ensures consistency across the model's depth.
    """
    logger.info("=" * 60)
    logger.info("Testing MLP Layers Across Multiple Decoders")
    logger.info("=" * 60)

    # Set HF_MODEL environment variable
    if not os.environ.get("HF_MODEL"):
        os.environ["HF_MODEL"] = MODEL_PATH

    # 1. Create test input
    logger.info("\n1. Creating Test Inputs...")
    inputs = create_random_input(1, 32, HIDDEN_SIZE)
    logger.info(f"   Input shape: {inputs.shape}")

    # 2. Load weights
    logger.info("\n2. Loading MLP Weights...")
    from models.experimental.miniCPMo.tt.minicpm_weight_bridge import MiniCPMWeightBridge

    bridge = MiniCPMWeightBridge(MODEL_PATH)
    qwen_weights = bridge.get_qwen_weights()

    # 3. Test first 3 layers
    num_layers_to_test = 3
    logger.info(f"\n3. Testing {num_layers_to_test} decoder layers...")

    from models.experimental.miniCPMo.tt_transformers.common import create_tt_model

    tt_model_args, tt_model, tt_kv_cache, _ = create_tt_model(
        mesh_device=mesh_device,
        instruct=False,
        max_batch_size=1,
        optimizations=None,
        max_seq_len=1024,
        paged_attention_config=None,
        dtype=ttnn.bfloat8_b,
        state_dict=qwen_weights,
        dummy_weights=False,
        num_layers=num_layers_to_test,
    )

    pcc_results = []

    for layer_num in range(num_layers_to_test):
        logger.info(f"\n   === Layer {layer_num} ===")

        # Create PyTorch reference MLP
        pt_mlp = PyTorchQwen2MLP(hidden_size=HIDDEN_SIZE, intermediate_size=INTERMEDIATE_SIZE)
        pt_mlp.load_weights(qwen_weights, layer_num=layer_num)
        pt_mlp = pt_mlp.to(torch.bfloat16).eval()

        # Run PyTorch forward
        with torch.no_grad():
            pt_output = pt_mlp(inputs.to(torch.bfloat16))

        # Get TT MLP for this layer
        tt_mlp = tt_model.layers[layer_num].feed_forward

        # Run TT forward
        tt_input = ttnn.from_torch(
            inputs.unsqueeze(1).to(torch.bfloat16),
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

        tt_output = tt_mlp.forward(tt_input, mode="prefill")
        tt_output_torch = ttnn.to_torch(tt_output).float()
        ttnn.deallocate(tt_input)
        ttnn.deallocate(tt_output)

        # Reshape
        while tt_output_torch.dim() > 3:
            tt_output_torch = (
                tt_output_torch.squeeze(1) if tt_output_torch.shape[1] == 1 else tt_output_torch.squeeze(0)
            )

        # Calculate PCC
        pt_output_np = pt_output.float()
        if tt_output_torch.shape != pt_output_np.shape and tt_output_torch.dim() == 4:
            tt_output_torch = tt_output_torch.squeeze(0)

        pcc = calculate_pcc(pt_output_np, tt_output_torch)
        pcc_results.append(pcc)
        logger.info(f"   Layer {layer_num} PCC: {pcc:.6f}")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Summary:")
    for i, pcc in enumerate(pcc_results):
        status = "✅" if pcc > 0.90 else "❌"
        logger.info(f"   Layer {i}: PCC = {pcc:.6f} {status}")

    avg_pcc = sum(pcc_results) / len(pcc_results)
    logger.info(f"   Average PCC: {avg_pcc:.6f}")
    logger.info("=" * 60)

    # All layers should have PCC > 0.90
    for i, pcc in enumerate(pcc_results):
        assert pcc > 0.90, f"Layer {i} PCC {pcc:.6f} is below threshold 0.90"

    logger.info("✅ MLP Multiple Layers Test PASSED")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
