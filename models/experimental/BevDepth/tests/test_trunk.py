# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger
import ttnn

from ttnn.model_preprocessing import preprocess_model_parameters, infer_ttnn_module_args
from tests.ttnn.utils_for_testing import check_with_pcc

from models.experimental.BevDepth.reference.bevdepth.exps.nuscenes.mv.bev_depth_lss_r50_256x704_128x128_24e_2key import (
    BEVDepthLightningModel,
)
from models.experimental.BevDepth.tt.bev_depth_head import TtResNet, head_optimisations
from models.experimental.BevDepth.tt.custom_preprocessing import create_custom_mesh_preprocessor


class TrunkTestInfra:
    def __init__(
        self,
        device,
        model_config,
    ):
        super().__init__()
        if not hasattr(self, "_model_initialized"):
            torch.manual_seed(42)
            torch.cuda.manual_seed_all(42)
            torch.backends.cudnn.deterministic = True
            self._model_initialized = True

        # Core config
        self.device = device
        self.num_devices = device.get_num_devices()
        self.model_config = model_config

        # Mesh mappers
        self.inputs_mesh_mapper, self.weights_mesh_mapper, self.output_mesh_composer = self.get_mesh_mappers(device)

        # Torch model - load full model and extract trunk for reference
        lightning_model = BEVDepthLightningModel()
        lightning_model.load_checkpoint("../reference/checkpoints/bev_depth_lss_r50_256x704_128x128_24e_2key.pth")
        torch_head_model = lightning_model.model.head
        torch_head_model.eval()
        torch_trunk_model = torch_head_model.trunk  # Extract trunk for reference forward pass

        # Synthetic input - same as test_head.py
        self.torch_input_tensor = self._create_input_tensor()

        # Torch output - use no_grad to ensure no gradients are computed
        with torch.no_grad():
            # Reference trunk forward pass
            # The trunk collects outputs: [input, after layer1, after layer2, after layer3]
            # This matches the reference BEVDepthHead.forward() behavior
            # We manually call trunk components because maxpool is deleted in BEVDepthHead.__init__
            trunk_outs = [self.torch_input_tensor]
            x = self.torch_input_tensor.float()

            # Conv1 + norm1 + relu (not included in outputs, but needed for processing)
            x = torch_trunk_model.conv1(x)
            x = torch_trunk_model.norm1(x)
            x = torch_trunk_model.relu(x)

            # Process res_layers and collect outputs based on out_indices
            # out_indices=[0, 1, 2] means we collect after layer1, layer2, layer3
            for i, layer_name in enumerate(torch_trunk_model.res_layers):
                res_layer = getattr(torch_trunk_model, layer_name)
                x = res_layer(x)
                if i in torch_trunk_model.out_indices:
                    trunk_outs.append(x)

            # trunk_outs should be: [input, after layer1, after layer2, after layer3]
            self.torch_output = trunk_outs  # List of 4 tensors

        logger.info(f"Reference trunk outputs: {len(self.torch_output)} tensors")
        for i, out in enumerate(self.torch_output):
            logger.info(f"  Output {i}: {out.shape}")

        # Preprocess model parameters - preprocess the full head model to get trunk parameters
        # This matches the approach in test_head.py
        parameters = preprocess_model_parameters(
            initialize_model=lambda: torch_head_model,
            custom_preprocessor=create_custom_mesh_preprocessor(self.weights_mesh_mapper),
            device=None,
        )
        parameters.conv_args = infer_ttnn_module_args(
            model=torch_head_model,
            run_model=lambda m: m(self.torch_input_tensor),
            device=None,
        )

        # Extract trunk parameters from the preprocessed parameters
        trunk_params = parameters.get("trunk", {})

        # Initialize TTNN trunk model
        self.ttnn_model = TtResNet(trunk_params, model_config, layer_optimisations=head_optimisations)

        # Run model in phases and validate
        logger.info(f"Running TTNN Trunk model")

        # Rebuild TTNN input (since buffers may be freed across passes)
        tt_host_tensor = ttnn.from_torch(
            self.torch_input_tensor.permute(0, 2, 3, 1),  # NCHW -> NHWC
            dtype=ttnn.bfloat16,
            device=self.device,
            mesh_mapper=self.inputs_mesh_mapper,
        )
        self.input_tensor = ttnn.to_device(tt_host_tensor, self.device, memory_config=ttnn.L1_MEMORY_CONFIG)

        # Optional: reinstantiate TTNN model
        self.ttnn_model = TtResNet(trunk_params, model_config, layer_optimisations=head_optimisations)

        self.run()
        self.validate()

    def _create_input_tensor(self):
        shape = (2, 160, 128, 128)  # NCHW format
        logger.info(f"Generating synthetic input tensor of shape {shape}")
        # Explicitly set requires_grad=False to ensure no gradients are tracked
        return torch.randn(shape, dtype=torch.float32, requires_grad=False)

    @classmethod
    def get_mesh_mappers(cls, device):
        if device.get_num_devices() != 1:
            return (
                ttnn.ShardTensorToMesh(device, dim=0),
                None,
                ttnn.ConcatMeshToTensor(device, dim=0),
            )
        return None, None, None

    def run(self):
        logger.info("Running TTNN Trunk model...")
        self.ttnn_output = self.ttnn_model(self.input_tensor, self.device)
        return self.ttnn_output

    def _tt_to_torch_nchw(self, tt_tensor, expected_shape):
        """Convert TTNN tensor (NHWC) to PyTorch tensor (NCHW)"""
        torch_tensor = ttnn.to_torch(tt_tensor, device=self.device, mesh_composer=self.output_mesh_composer)
        torch_tensor = torch.reshape(
            torch_tensor,
            (expected_shape[0], expected_shape[2], expected_shape[3], expected_shape[1]),
        )
        return torch.permute(torch_tensor, (0, 3, 1, 2))

    def validate(self):
        logger.info("Validating TTNN trunk output against PyTorch...")

        # Output is a tuple of 4 tensors: (x, x1, x2, x3)
        # x: input (after conversion to NHWC and back)
        # x1: after layer1
        # x2: after layer2
        # x3: after layer3
        assert isinstance(self.torch_output, list), "Torch output should be a list"
        assert isinstance(self.ttnn_output, tuple), "TTNN output should be a tuple"
        assert len(self.torch_output) == len(
            self.ttnn_output
        ), f"Output length mismatch: {len(self.torch_output)} vs {len(self.ttnn_output)}"
        assert len(self.torch_output) == 4, f"Expected 4 outputs, got {len(self.torch_output)}"

        pcc_threshold = 0.95
        all_passed = True
        all_messages = []

        # Iterate through each output
        output_names = ["x (input)", "x1 (after layer1)", "x2 (after layer2)", "x3 (after layer3)"]
        for output_idx, (torch_tensor, tt_tensor, output_name) in enumerate(
            zip(self.torch_output, self.ttnn_output, output_names)
        ):
            logger.info(f"\nValidating {output_name} (output {output_idx})...")
            logger.info(f"  Torch shape: {torch_tensor.shape}")
            logger.info(f"  TTNN shape: {tt_tensor.shape}")

            # Convert TTNN tensor to torch format (NHWC -> NCHW)
            tt_tensor_torch = self._tt_to_torch_nchw(tt_tensor, torch_tensor.shape)

            # Deallocate to save memory
            ttnn.deallocate(tt_tensor)

            # Evaluate PCC
            passed, msg = check_with_pcc(torch_tensor, tt_tensor_torch, pcc=pcc_threshold)

            if not passed:
                all_passed = False
                error_msg = f"Output {output_idx} ({output_name}) PCC check failed: {msg}"
                logger.error(error_msg)
                all_messages.append(error_msg)
            else:
                logger.info(f"Output {output_idx} ({output_name}): PCC check passed : {msg}")

        assert all_passed, f"Some PCC checks failed:\n" + "\n".join(all_messages)

        logger.info(
            f"BEVDepth Trunk Tested: "
            f"act_dtype={self.model_config['ACTIVATIONS_DTYPE']}, "
            f"weight_dtype={self.model_config['WEIGHTS_DTYPE']}, "
            f"math_fidelity={self.model_config['MATH_FIDELITY']}, "
        )

        return True, "All PCC checks passed"


model_config = {
    "MATH_FIDELITY": ttnn.MathFidelity.HiFi4,
    "WEIGHTS_DTYPE": ttnn.bfloat16,
    "ACTIVATIONS_DTYPE": ttnn.bfloat16,
}


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_trunk(device):
    TrunkTestInfra(
        device=device,
        model_config=model_config,
    )
