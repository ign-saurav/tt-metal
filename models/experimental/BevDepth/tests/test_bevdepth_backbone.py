# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger
import ttnn

from models.experimental.BevDepth.reference.bevdepth.exps.nuscenes.mv.bev_depth_lss_r50_256x704_128x128_24e_2key import (
    BEVDepthLightningModel,
)
from models.experimental.BevDepth.tt.ttnn_bevdepth_backbone import TtBaseLSSFPN
from models.common.utility_functions import comp_pcc


def download_bevdepth_weights():
    """Download BEVDepth pretrained weights"""
    import urllib.request
    import os

    url = "https://github.com/Megvii-BaseDetection/BEVDepth/releases/download/v0.0.2/bev_depth_lss_r50_256x704_128x128_24e_2key.pth"
    weights_path = "/tmp/bevdepth_weights.pth"

    if not os.path.exists(weights_path):
        logger.info(f"Downloading weights from {url}")
        urllib.request.urlretrieve(url, weights_path)
        logger.info(f"Downloaded weights to {weights_path}")

    return weights_path


def load_reference_model():
    """Load the reference BEVDepth model."""
    logger.info("Loading reference BEVDepth model...")
    lightning_model = BEVDepthLightningModel()
    checkpoint_path = download_bevdepth_weights()

    import os

    if not os.path.exists(checkpoint_path):
        logger.warning(f"Checkpoint not found at {checkpoint_path}")
        return None

    lightning_model.load_checkpoint(checkpoint_path, verbose=False)

    lightning_model.model.eval()
    return lightning_model


def create_dummy_inputs(batch_size=1, num_sweeps=2, num_cameras=6, img_h=256, img_w=640):
    """Create dummy input images and transformation matrices."""
    # Images: (B, num_sweeps, num_cameras, 3, H, W)
    imgs = torch.randn((batch_size, num_sweeps, num_cameras, 3, img_h, img_w), dtype=torch.float32, requires_grad=False)

    # Transformation matrices
    mats_dict = {
        # Sensor to ego transformation (camera to vehicle coordinates)
        "sensor2ego_mats": torch.eye(4)
        .unsqueeze(0)
        .unsqueeze(0)
        .unsqueeze(0)
        .repeat(batch_size, num_sweeps, num_cameras, 1, 1),
        # Intrinsic camera parameters
        "intrin_mats": torch.eye(4)
        .unsqueeze(0)
        .unsqueeze(0)
        .unsqueeze(0)
        .repeat(batch_size, num_sweeps, num_cameras, 1, 1),
        # Image data augmentation matrix
        "ida_mats": torch.eye(4)
        .unsqueeze(0)
        .unsqueeze(0)
        .unsqueeze(0)
        .repeat(batch_size, num_sweeps, num_cameras, 1, 1),
        # Sensor to sensor transformation (for temporal alignment)
        "sensor2sensor_mats": torch.eye(4)
        .unsqueeze(0)
        .unsqueeze(0)
        .unsqueeze(0)
        .repeat(batch_size, num_sweeps, num_cameras, 1, 1),
        # Bird's eye view data augmentation
        "bda_mat": torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1),
    }

    return imgs, mats_dict


# ----------------------------------------------------#
# ResNet50 Backbone functions
# ----------------------------------------------------#


def extract_backbone_state_dict(checkpoint_path):
    """Extract backbone weights from BEVDepth checkpoint"""
    import gc

    logger.info(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        del checkpoint
        gc.collect()
    else:
        state_dict = checkpoint

    # Extract only img_backbone weights
    backbone_state = {}
    for key, value in state_dict.items():
        if key.startswith("model.backbone.img_backbone."):
            new_key = key.replace("model.backbone.img_backbone.", "")
            backbone_state[new_key] = value

    # Free the full state_dict to save memory
    del state_dict
    gc.collect()

    logger.info(f"Extracted {len(backbone_state)} backbone parameters")

    return backbone_state


def fuse_conv_bn_weights(conv_weight, bn_weight, bn_bias, bn_mean, bn_var, eps=1e-5):
    """
    Fuse BatchNorm parameters into conv weights for inference.
    """
    # Calculate scale factor from BN
    std = torch.sqrt(bn_var + eps)
    scale = bn_weight / std

    # Fuse into conv weight: multiply each output channel by its scale
    # conv_weight shape: (out_channels, in_channels, kH, kW)
    fused_weight = conv_weight * scale.view(-1, 1, 1, 1)

    # Fuse into bias
    fused_bias = bn_bias - (bn_weight * bn_mean / std)

    return fused_weight, fused_bias


def fuse_batchnorm_into_conv(state_dict):
    """Fuse all BatchNorm layers into their corresponding conv layers"""
    fused_state = {}

    # Fuse bn1 into conv1
    if "conv1.weight" in state_dict and "bn1.weight" in state_dict:
        fused_weight, fused_bias = fuse_conv_bn_weights(
            state_dict["conv1.weight"],
            state_dict["bn1.weight"],
            state_dict["bn1.bias"],
            state_dict["bn1.running_mean"],
            state_dict["bn1.running_var"],
        )
        fused_state["conv1.weight"] = fused_weight
        fused_state["conv1.bias"] = fused_bias

    # Fuse BN in each bottleneck block
    for layer_idx in range(1, 5):
        layer_name = f"layer{layer_idx}"
        block_idx = 0
        while True:
            block_prefix = f"{layer_name}.{block_idx}."

            # Check if this block exists
            if f"{block_prefix}conv1.weight" not in state_dict:
                break

            # Fuse bn1, bn2, bn3 for each conv in the block
            for conv_idx in range(1, 4):
                conv_key = f"{block_prefix}conv{conv_idx}.weight"
                bn_key = f"{block_prefix}bn{conv_idx}.weight"

                if conv_key in state_dict and bn_key in state_dict:
                    fused_weight, fused_bias = fuse_conv_bn_weights(
                        state_dict[conv_key],
                        state_dict[bn_key],
                        state_dict[f"{block_prefix}bn{conv_idx}.bias"],
                        state_dict[f"{block_prefix}bn{conv_idx}.running_mean"],
                        state_dict[f"{block_prefix}bn{conv_idx}.running_var"],
                    )
                    fused_state[conv_key] = fused_weight
                    fused_state[f"{block_prefix}conv{conv_idx}.bias"] = fused_bias

            # Fuse downsample BN if exists
            if f"{block_prefix}downsample.0.weight" in state_dict:
                if f"{block_prefix}downsample.1.weight" in state_dict:
                    fused_weight, fused_bias = fuse_conv_bn_weights(
                        state_dict[f"{block_prefix}downsample.0.weight"],
                        state_dict[f"{block_prefix}downsample.1.weight"],
                        state_dict[f"{block_prefix}downsample.1.bias"],
                        state_dict[f"{block_prefix}downsample.1.running_mean"],
                        state_dict[f"{block_prefix}downsample.1.running_var"],
                    )
                    fused_state[f"{block_prefix}downsample.0.weight"] = fused_weight
                    fused_state[f"{block_prefix}downsample.0.bias"] = fused_bias

            block_idx += 1

    # Copy all other weights that don't need fusion
    for key, value in state_dict.items():
        if key not in fused_state and not key.startswith("bn") and "downsample.1" not in key:
            fused_state[key] = value

    logger.info(f"Fused BatchNorm into conv weights. Original keys: {len(state_dict)}, Fused keys: {len(fused_state)}")
    return fused_state


def prepare_ttnn_parameters(state_dict):
    """Keep weights as PyTorch tensors - convert during conv2d call"""

    class Parameters:
        pass

    params = Parameters()

    params.conv1 = Parameters()
    params.conv1.weight = state_dict["conv1.weight"].to(torch.bfloat16)
    params.conv1.bias = state_dict.get("conv1.bias", None)
    if params.conv1.bias is not None:
        params.conv1.bias = params.conv1.bias.to(torch.bfloat16)

    # Layers - keep as PyTorch tensors
    for layer_idx in range(1, 5):
        layer_name = f"layer{layer_idx}"
        layer_params = []

        block_idx = 0
        while True:
            block_prefix = f"{layer_name}.{block_idx}."
            if not any(k.startswith(block_prefix) for k in state_dict.keys()):
                break

            block_params = Parameters()

            # All weights as PyTorch tensors - load fused biases from state_dict
            block_params.conv1 = Parameters()
            block_params.conv1.weight = state_dict[f"{block_prefix}conv1.weight"].to(torch.bfloat16)
            # Load fused bias if it exists (from batch norm fusion)
            conv1_bias_key = f"{block_prefix}conv1.bias"
            if conv1_bias_key in state_dict:
                block_params.conv1.bias = state_dict[conv1_bias_key].to(torch.bfloat16)
            else:
                block_params.conv1.bias = None

            block_params.conv2 = Parameters()
            block_params.conv2.weight = state_dict[f"{block_prefix}conv2.weight"].to(torch.bfloat16)
            # Load fused bias if it exists
            conv2_bias_key = f"{block_prefix}conv2.bias"
            if conv2_bias_key in state_dict:
                block_params.conv2.bias = state_dict[conv2_bias_key].to(torch.bfloat16)
            else:
                block_params.conv2.bias = None

            block_params.conv3 = Parameters()
            block_params.conv3.weight = state_dict[f"{block_prefix}conv3.weight"].to(torch.bfloat16)
            # Load fused bias if it exists
            conv3_bias_key = f"{block_prefix}conv3.bias"
            if conv3_bias_key in state_dict:
                block_params.conv3.bias = state_dict[conv3_bias_key].to(torch.bfloat16)
            else:
                block_params.conv3.bias = None

            if f"{block_prefix}downsample.0.weight" in state_dict:
                block_params.downsample = [Parameters()]
                block_params.downsample[0].weight = state_dict[f"{block_prefix}downsample.0.weight"].to(torch.bfloat16)
                # Load fused bias if it exists
                downsample_bias_key = f"{block_prefix}downsample.0.bias"
                if downsample_bias_key in state_dict:
                    block_params.downsample[0].bias = state_dict[downsample_bias_key].to(torch.bfloat16)
                else:
                    block_params.downsample[0].bias = None

            layer_params.append(block_params)
            block_idx += 1

        setattr(params, layer_name, layer_params)

    return params


def extract_neck_state_dict(checkpoint_path):
    """Extract neck (SECONDFPN) weights from BEVDepth checkpoint"""
    import gc

    logger.info(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        del checkpoint
        gc.collect()
    else:
        state_dict = checkpoint

    # Extract only img_neck weights - keep the full prefix for prepare_secondfpn_parameters
    neck_state = {}
    # Try different prefix patterns
    patterns = [
        "model.backbone.img_neck.",
        "img_backbone.img_neck.",
        "backbone.img_neck.",
        "img_neck.",
    ]

    for pattern in patterns:
        for key, value in state_dict.items():
            if key.startswith(pattern):
                neck_state[key] = value

    # Free the full state_dict to save memory
    del state_dict
    gc.collect()

    logger.info(f"Extracted {len(neck_state)} neck parameters")
    return neck_state


def extract_depthnet_state_dict(checkpoint_path):
    """Extract depthnet weights from BEVDepth checkpoint"""
    import gc

    logger.info(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        del checkpoint
        gc.collect()
    else:
        state_dict = checkpoint

    # Extract only depth_net weights - keep the full prefix for prepare_depthnet_parameters
    depthnet_state = {}
    # Try different prefix patterns
    patterns = [
        "model.backbone.depth_net.",
        "img_backbone.depth_net.",
        "backbone.depth_net.",
        "depth_net.",
    ]

    for pattern in patterns:
        for key, value in state_dict.items():
            if key.startswith(pattern):
                depthnet_state[key] = value

    # Free the full state_dict to save memory
    del state_dict
    gc.collect()

    logger.info(f"Extracted {len(depthnet_state)} depthnet parameters")
    return depthnet_state


def prepare_backbone_parameters():
    """Prepare parameters for ResNet50 backbone."""
    logger.info("Preparing backbone parameters...")
    checkpoint_path = download_bevdepth_weights()
    backbone_state = extract_backbone_state_dict(checkpoint_path)
    backbone_state = fuse_batchnorm_into_conv(backbone_state)
    return prepare_ttnn_parameters(backbone_state)


def prepare_neck_parameters():
    """Prepare parameters for SECONDFPN neck (4 levels matching reference base_exp.py)."""
    logger.info("Preparing neck parameters...")
    from models.experimental.BevDepth.tt.ttnn_secondfpn import prepare_secondfpn_parameters

    checkpoint_path = download_bevdepth_weights()
    neck_state = extract_neck_state_dict(checkpoint_path)

    # Match reference base_exp.py: 4 levels from ResNet50 outputs
    in_channels = [256, 512, 1024, 2048]
    out_channels = [128, 128, 128, 128]
    upsample_strides = [0.25, 0.5, 1, 2]
    return prepare_secondfpn_parameters(
        neck_state,
        in_channels=in_channels,
        out_channels=out_channels,
        upsample_strides=upsample_strides,
    )


def prepare_depthnet_parameters():
    """Prepare parameters for DepthNet."""
    logger.info("Preparing depthnet parameters...")
    from models.experimental.BevDepth.tt.ttnn_depthnet import prepare_depthnet_parameters

    checkpoint_path = download_bevdepth_weights()
    depthnet_state = extract_depthnet_state_dict(checkpoint_path)

    # depth_channels = len(torch.arange(2.0, 58.0, 0.5)) = 112
    # This must match d_bound in lss_conf
    return prepare_depthnet_parameters(
        depthnet_state,
        in_channels=512,
        mid_channels=512,
        depth_channels=112,
    )


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
        self.depthnet_params = prepare_depthnet_parameters()

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
        shape = (batch_size, 2, 6, 3, 256, 640)  # (B, num_sweeps, num_cameras, 3, H, W)
        logger.info(f"Generating synthetic input images of shape {shape}")
        imgs, mats_dict = create_dummy_inputs(
            batch_size=batch_size,
            num_sweeps=2,
            num_cameras=6,
            img_h=256,
            img_w=640,
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
        # Fix the pcc as it should be > 0.99
        assert pcc_value > 0.99, f"PCC {pcc_value:.6f} is below threshold 0.99"

        return pcc_value


model_config = {
    "MATH_FIDELITY": ttnn.MathFidelity.HiFi4,
    "WEIGHTS_DTYPE": ttnn.bfloat16,
    "ACTIVATIONS_DTYPE": ttnn.bfloat16,
    "batch_size": 1,
    # Match reference base_exp.py config
    "neck_in_channels": [256, 512, 1024, 2048],
    "neck_out_channels": [128, 128, 128, 128],
    "neck_upsample_strides": [0.25, 0.5, 1, 2],
    "use_torch_conv_transpose": False,  # Use pure TTNN for conv_transpose2d
    "depthnet_in_channels": 512,
    "depthnet_mid_channels": 512,
    "depthnet_context_channels": 80,
    # depth_channels = len(torch.arange(2.0, 58.0, 0.5)) = 112 (from d_bound in lss_conf)
    "depthnet_depth_channels": 112,
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
