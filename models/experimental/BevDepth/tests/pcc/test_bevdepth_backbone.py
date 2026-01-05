# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger
import ttnn

from models.experimental.BevDepth.tt.ttnn_bevdepth_backbone import TtBaseLSSFPN
from models.common.utility_functions import comp_pcc

# Import common utilities
from models.experimental.BevDepth.common import (
    load_reference_model,
    create_dummy_inputs,
)

# Import parameter preparation functions from custom_preprocessing
from models.experimental.BevDepth.tt.custom_preprocessing import (
    prepare_backbone_parameters,
    prepare_neck_parameters,
    prepare_depthnet_parameters,
    extract_depthnet_state_dict,
)
from models.experimental.BevDepth.common import download_bevdepth_weights


# ----------------------------------------------------#
# ResNet50 Backbone functions
# ----------------------------------------------------#


class BackboneTestInfra:
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

        # LSS configuration (from base_exp.py)
        self.lss_conf = {
            "x_bound": [-51.2, 51.2, 0.8],
            "y_bound": [-51.2, 51.2, 0.8],
            "z_bound": [-5.0, 3.0, 0.2],
            "d_bound": [2.0, 58.0, 0.5],
            "final_dim": [256, 704],
            "downsample_factor": 16,
            "output_channels": 80,
        }

        # Load reference model
        logger.info("Loading reference model...")
        self.reference_model = load_reference_model()
        if self.reference_model is None:
            raise RuntimeError("Failed to load reference model")
        # Get reference backbone
        self.torch_backbone = self.reference_model.model.backbone
        self.torch_backbone.eval()

        # Create synthetic input
        self.torch_input_imgs, self.mats_dict = self._create_input_tensors()

        # Prepare parameters
        logger.info("Preparing parameters...")
        self.backbone_params = prepare_backbone_parameters()
        self.neck_params = prepare_neck_parameters()
        self.depthnet_params = prepare_depthnet_parameters(extract_depthnet_state_dict(download_bevdepth_weights()))

        # Initialize TTNN model
        logger.info("Initializing TTNN backbone...")
        self.ttnn_model = TtBaseLSSFPN(
            device=device,
            backbone_parameters=self.backbone_params,
            neck_parameters=self.neck_params,
            depthnet_parameters=self.depthnet_params,
            lss_conf=self.lss_conf,
            model_config=self.model_config,
        )

    def _create_input_tensors(self):
        """Create synthetic input tensors."""
        batch_size = self.model_config.get("batch_size", 1)
        shape = (batch_size, 2, 6, 3, 256, 704)  # (B, num_sweeps, num_cameras, 3, H, W)
        logger.info(f"Generating synthetic input images of shape {shape}")
        imgs, mats_dict = create_dummy_inputs(
            batch_size=batch_size,
            num_sweeps=2,
            num_cameras=6,
            img_h=256,
            img_w=704,
        )
        return imgs, mats_dict

    def run(self):
        """Run forward pass on TTNN model."""
        logger.info("Running TTNN model forward pass...")
        self.ttnn_output = self.ttnn_model(
            self.torch_input_imgs,
            self.mats_dict,
            is_return_depth=False,
        )
        logger.info(f"TTNN output shape: {self.ttnn_output.shape}")
        return self.ttnn_output

    def validate(self):
        """Validate TTNN output against reference model."""
        logger.info("Running reference backbone...")
        with torch.no_grad():
            self.torch_output = self.torch_backbone(self.torch_input_imgs, self.mats_dict, is_return_depth=False)

        # Ensure both outputs are on CPU and have the same dtype
        ref_output = self.torch_output.cpu().float()
        ttnn_output = self.ttnn_output.cpu().float() if isinstance(self.ttnn_output, torch.Tensor) else self.ttnn_output

        logger.info(f"Reference output shape: {ref_output.shape}")
        logger.info(f"TTNN output shape: {ttnn_output.shape}")

        # Compare outputs using PCC
        pcc_result = comp_pcc(ref_output, ttnn_output)
        pcc_value = pcc_result[1] if isinstance(pcc_result, tuple) else pcc_result

        logger.info(f"PCC: {pcc_value:.6f}")

        # Assert PCC threshold
        assert pcc_value > 0.99, f"PCC {pcc_value:.6f} is below threshold 0.99"

        return pcc_value


model_config = {
    "MATH_FIDELITY": ttnn.MathFidelity.HiFi4,
    "WEIGHTS_DTYPE": ttnn.bfloat16,
    "ACTIVATIONS_DTYPE": ttnn.bfloat16,
    "batch_size": 1,
    "neck_in_channels": [256, 512, 1024, 2048],
    "neck_out_channels": [128, 128, 128, 128],
    "neck_upsample_strides": [0.25, 0.5, 1, 2],
    "depthnet_in_channels": 512,
    "depthnet_mid_channels": 512,
    "depthnet_context_channels": 80,
    "depthnet_depth_channels": 112,
    "use_torch_fallback": True,
}


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_backbone(device):
    """Test TTNN BEVDepth backbone against reference model."""
    test_infra = BackboneTestInfra(
        device=device,
        model_config=model_config,
    )

    # Run forward pass
    test_infra.run()

    # Validate against reference
    pcc_value = test_infra.validate()

    logger.info(f"✓ Test passed with PCC: {pcc_value:.6f}")
    return pcc_value
