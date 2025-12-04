# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger
import ttnn

from models.experimental.BevDepth.reference.bevdepth.exps.nuscenes.mv.bev_depth_lss_r50_256x704_128x128_24e_2key import (
    BEVDepthLightningModel,
)
from models.experimental.BevDepth.tt.bev_depth_backbone import TtBaseLSSFPN
from models.experimental.BevDepth.tt.ttnn_secondfpn import prepare_secondfpn_parameters
from models.experimental.BevDepth.tt.ttnn_depthnet import (
    prepare_depthnet_parameters as prepare_depthnet_parameters_ttnn,
)
from tests.ttnn.utils_for_testing import check_with_pcc


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

    result = lightning_model.load_checkpoint(checkpoint_path, verbose=False)
    if not result["success"]:
        logger.warning(f"Failed to load checkpoint: {result.get('error', 'Unknown error')}")
        return None

    lightning_model.model.eval()
    return lightning_model


def create_dummy_inputs(batch_size=1, num_sweeps=2, num_cameras=6, img_h=256, img_w=704):
    """Create dummy input images and transformation matrices."""
    # Images: (B, num_sweeps, num_cameras, 3, H, W)
    imgs = torch.randn(batch_size, num_sweeps, num_cameras, 3, img_h, img_w)

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


def fuse_conv_bn_weights(conv_weight, bn_weight, bn_bias, bn_mean, bn_var, eps=1e-5):
    """Fuse BatchNorm parameters into conv weights for inference."""
    std = torch.sqrt(bn_var + eps)
    scale = bn_weight / std
    fused_weight = conv_weight * scale.view(-1, 1, 1, 1)
    fused_bias = bn_bias - (bn_weight * bn_mean / std)
    return fused_weight, fused_bias


def _fuse_block_conv_bn(state_dict, block_prefix, conv_idx, fused_state):
    """Helper to fuse conv and BN for a single conv layer in a block."""
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


def extract_backbone_state_dict(checkpoint_path):
    """Extract backbone weights from BEVDepth checkpoint"""
    logger.info(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    state_dict = checkpoint.get("state_dict", checkpoint)
    if "state_dict" in checkpoint:
        import gc

        del checkpoint
        gc.collect()

    # Extract only img_backbone weights
    backbone_state = {
        key.replace("model.backbone.img_backbone.", ""): value
        for key, value in state_dict.items()
        if key.startswith("model.backbone.img_backbone.")
    }

    import gc

    del state_dict
    gc.collect()
    logger.info(f"Extracted {len(backbone_state)} backbone parameters")
    return backbone_state


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
            if f"{block_prefix}conv1.weight" not in state_dict:
                break

            # Fuse bn1, bn2, bn3 for each conv in the block
            for conv_idx in range(1, 4):
                _fuse_block_conv_bn(state_dict, block_prefix, conv_idx, fused_state)

            # Fuse downsample BN if exists
            downsample_conv_key = f"{block_prefix}downsample.0.weight"
            downsample_bn_key = f"{block_prefix}downsample.1.weight"
            if downsample_conv_key in state_dict and downsample_bn_key in state_dict:
                fused_weight, fused_bias = fuse_conv_bn_weights(
                    state_dict[downsample_conv_key],
                    state_dict[downsample_bn_key],
                    state_dict[f"{block_prefix}downsample.1.bias"],
                    state_dict[f"{block_prefix}downsample.1.running_mean"],
                    state_dict[f"{block_prefix}downsample.1.running_var"],
                )
                fused_state[downsample_conv_key] = fused_weight
                fused_state[f"{block_prefix}downsample.0.bias"] = fused_bias

            block_idx += 1

    # Copy all other weights that don't need fusion
    for key, value in state_dict.items():
        if key not in fused_state and not key.startswith("bn") and "downsample.1" not in key:
            fused_state[key] = value

    logger.info(f"Fused BatchNorm into conv weights. Original keys: {len(state_dict)}, Fused keys: {len(fused_state)}")
    return fused_state


def _create_conv_params(state_dict, prefix, conv_name=""):
    """Helper to create conv parameters from state dict."""

    class Parameters:
        pass

    params = Parameters()
    if conv_name:
        weight_key = f"{prefix}{conv_name}.weight"
        bias_key = f"{prefix}{conv_name}.bias"
    else:
        # For downsample where prefix already includes the full path
        weight_key = f"{prefix}.weight"
        bias_key = f"{prefix}.bias"

    params.weight = state_dict[weight_key].to(torch.bfloat16)
    params.bias = state_dict.get(bias_key, None)
    if params.bias is not None:
        params.bias = params.bias.to(torch.bfloat16)

    return params


def prepare_ttnn_parameters(state_dict, device):
    """Prepare TTNN parameters from fused state dict"""

    class Parameters:
        pass

    params = Parameters()
    params.conv1 = _create_conv_params(state_dict, "", "conv1")

    # Process layers
    for layer_idx in range(1, 5):
        layer_name = f"layer{layer_idx}"
        layer_params = []
        block_idx = 0

        while True:
            block_prefix = f"{layer_name}.{block_idx}."
            if not any(k.startswith(block_prefix) for k in state_dict.keys()):
                break

            block_params = Parameters()
            block_params.conv1 = _create_conv_params(state_dict, block_prefix, "conv1")
            block_params.conv2 = _create_conv_params(state_dict, block_prefix, "conv2")
            block_params.conv3 = _create_conv_params(state_dict, block_prefix, "conv3")

            # Handle downsample
            downsample_key = f"{block_prefix}downsample.0.weight"
            if downsample_key in state_dict:
                block_params.downsample = [_create_conv_params(state_dict, f"{block_prefix}downsample.0", "")]

            layer_params.append(block_params)
            block_idx += 1

        setattr(params, layer_name, layer_params)

    return params


def prepare_backbone_parameters(reference_model, device):
    """Prepare parameters for ResNet50 backbone."""
    logger.info("Preparing backbone parameters...")
    checkpoint_path = download_bevdepth_weights()
    backbone_state = extract_backbone_state_dict(checkpoint_path)
    backbone_state = fuse_batchnorm_into_conv(backbone_state)
    return prepare_ttnn_parameters(backbone_state, device)


def prepare_neck_parameters(reference_model):
    """Prepare parameters for SECONDFPN neck."""
    logger.info("Preparing neck parameters...")
    checkpoint_path = download_bevdepth_weights()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Extract state dict
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    # Extract neck state - keep the full prefix so prepare_secondfpn_parameters can detect it
    # The function looks for prefixes like "model.backbone.img_neck.", "backbone.img_neck.", etc.
    # So we need to preserve the full key with prefix intact
    neck_state = {}
    for k, v in state_dict.items():
        # Look for keys containing img_neck (with various possible prefixes)
        if "img_neck" in k or ("backbone" in k and "neck" in k.lower() and "depth" not in k.lower()):
            neck_state[k] = v

    if not neck_state:
        logger.error(f"Could not find neck parameters. Sample keys: {list(state_dict.keys())[:20]}")
        raise KeyError("No neck parameters found in checkpoint")

    logger.info(f"Extracted {len(neck_state)} neck parameters. Sample keys: {list(neck_state.keys())[:5]}")

    # Prepare TTNN parameters - function will detect and handle the prefix
    in_channels = [256, 512, 128]
    out_channels = [128, 128, 128]
    neck_params = prepare_secondfpn_parameters(neck_state, in_channels=in_channels, out_channels=out_channels)

    return neck_params


def prepare_depthnet_parameters(reference_model):
    """Prepare parameters for DepthNet."""
    logger.info("Preparing depthnet parameters...")
    checkpoint_path = download_bevdepth_weights()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Extract state dict
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    # Extract depthnet state - keep the full prefix so prepare_depthnet_parameters_ttnn can detect it
    # The function looks for prefixes like "model.backbone.depth_net.", "backbone.depth_net.", etc.
    # So we need to preserve the full key with prefix intact
    depthnet_state = {}
    for k, v in state_dict.items():
        # Look for keys containing depth_net (with various possible prefixes)
        if "depth_net" in k or ("backbone" in k and "depth" in k.lower() and "net" in k.lower()):
            depthnet_state[k] = v

    if not depthnet_state:
        logger.error(f"Could not find depthnet parameters. Sample keys: {list(state_dict.keys())[:20]}")
        raise KeyError("No depthnet parameters found in checkpoint")

    logger.info(f"Extracted {len(depthnet_state)} depthnet parameters. Sample keys: {list(depthnet_state.keys())[:5]}")

    # Prepare TTNN parameters using the imported function - function will detect and handle the prefix
    # Match reference model config: depth_net_conf = dict(in_channels=512, mid_channels=512)
    # depth_channels calculated from d_bound: (58.0 - 2.0) / 0.5 = 112, but checkpoint uses 118
    in_channels = 512
    mid_channels = 512  # Match reference: mid_channels=512
    depth_channels = 118  # Checkpoint uses 118
    depthnet_params = prepare_depthnet_parameters_ttnn(
        depthnet_state,
        in_channels=in_channels,
        mid_channels=mid_channels,
        depth_channels=depth_channels,
    )

    return depthnet_params


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

        # Load reference model
        logger.info("Loading reference model...")
        self.reference_model = load_reference_model()
        if self.reference_model is None:
            raise RuntimeError("Failed to load reference model")

        # Get LSS configuration from reference model
        backbone_conf = self.reference_model.backbone_conf
        self.lss_conf = {
            "x_bound": backbone_conf.get("x_bound", [-51.2, 51.2, 0.8]),
            "y_bound": backbone_conf.get("y_bound", [-51.2, 51.2, 0.8]),
            "z_bound": backbone_conf.get("z_bound", [-5.0, 3.0, 0.2]),
            "d_bound": backbone_conf.get("d_bound", [2.0, 58.0, 0.5]),
            "final_dim": backbone_conf.get("final_dim", [256, 704]),
            "downsample_factor": backbone_conf.get("downsample_factor", 16),
            "output_channels": backbone_conf.get("output_channels", 80),
        }

        # Get reference backbone
        self.torch_backbone = self.reference_model.model.backbone
        self.torch_backbone.eval()

        # Create synthetic input
        self.torch_input_imgs, self.mats_dict = self._create_input_tensors()

        # Get reference output
        logger.info("Running reference backbone...")
        with torch.no_grad():
            self.torch_output = self.torch_backbone(self.torch_input_imgs, self.mats_dict, is_return_depth=False)

        # Prepare parameters
        logger.info("Preparing parameters...")
        backbone_params = prepare_backbone_parameters(self.reference_model, device)
        neck_params = prepare_neck_parameters(self.reference_model)
        depthnet_params = prepare_depthnet_parameters(self.reference_model)

        # Initialize TTNN model
        logger.info("Initializing TTNN backbone...")
        self.ttnn_model = TtBaseLSSFPN(
            device=device,
            backbone_parameters=backbone_params,
            neck_parameters=neck_params,
            depthnet_parameters=depthnet_params,
            lss_conf=self.lss_conf,
            model_config=self.model_config,
        )

        # Run model and validate
        logger.info("Running TTNN backbone...")
        self.run()
        self.validate()

    def _create_input_tensors(self):
        """Create synthetic input tensors."""
        batch_size = self.model_config.get("batch_size", 1)
        shape = (batch_size, 2, 6, 3, 256, 704)  # (B, num_sweeps, num_cameras, 3, H, W)
        logger.info(f"Generating synthetic input images of shape {shape}")
        imgs = torch.randn(shape, dtype=torch.float32, requires_grad=False)
        mats_dict = create_dummy_inputs(
            batch_size=batch_size,
            num_sweeps=2,
            num_cameras=6,
            img_h=256,
            img_w=704,
        )[
            1
        ]  # Get only mats_dict
        return imgs, mats_dict

    def run(self):
        """Run TTNN model."""
        logger.info("Running TTNN backbone model...")
        self.ttnn_output = self.ttnn_model(self.torch_input_imgs, self.mats_dict, is_return_depth=False)
        return self.ttnn_output

    def validate(self):
        """Validate TTNN output against PyTorch."""
        logger.info("Validating TTNN output against PyTorch...")

        # Both outputs should be tensors
        assert isinstance(
            self.torch_output, torch.Tensor
        ), f"Torch output should be a tensor, got {type(self.torch_output)}"
        assert isinstance(
            self.ttnn_output, torch.Tensor
        ), f"TTNN output should be a tensor, got {type(self.ttnn_output)}"

        # Check shapes match
        torch_shape = self.torch_output.shape
        tt_shape = self.ttnn_output.shape
        assert torch_shape == tt_shape, f"Shape mismatch: torch {torch_shape} vs ttnn {tt_shape}"

        # Evaluate PCC
        pcc_threshold = 0.99
        passed, msg = check_with_pcc(self.torch_output, self.ttnn_output, pcc=pcc_threshold)

        if not passed:
            error_msg = f"Backbone PCC check failed: {msg}"
            logger.error(error_msg)
            assert False, error_msg
        else:
            logger.info(f"Backbone PCC check passed: {msg}")

        logger.info(
            f"BEVDepth Backbone Tested: "
            f"act_dtype={self.model_config['ACTIVATIONS_DTYPE']}, "
            f"weight_dtype={self.model_config['WEIGHTS_DTYPE']}, "
            f"math_fidelity={self.model_config['MATH_FIDELITY']}, "
        )

        logger.info("✓ BEVDepth Backbone test passed!")
        return True, "All PCC checks passed"


model_config = {
    "MATH_FIDELITY": ttnn.MathFidelity.LoFi,
    "WEIGHTS_DTYPE": ttnn.bfloat16,
    "ACTIVATIONS_DTYPE": ttnn.bfloat16,
    "batch_size": 1,
    "neck_in_channels": [256, 512, 128],
    "neck_out_channels": [128, 128, 128],
    "neck_upsample_strides": [4, 2, 1],
    "depthnet_in_channels": 512,
    "depthnet_mid_channels": 512,
    "depthnet_context_channels": 80,
    "depthnet_depth_channels": 118,
}


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_backbone(device):
    BackboneTestInfra(
        device=device,
        model_config=model_config,
    )
