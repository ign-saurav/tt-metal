"""
Fixed BEVDepth ONNX Export for 2-key Temporal Model

Key Fix: The backbone outputs 80 channels, but for 2-key model,
we need to concatenate features from 2 timesteps (80 * 2 = 160)
before feeding to the head.
"""

import os
from pathlib import Path
from typing import Optional, Tuple

import torch
from torch import nn
from torch.onnx import register_custom_op_symbolic

from models.experimental.BevDepth.reference.bevdepth.exps.nuscenes.mv.bev_depth_lss_r50_256x704_128x128_24e_2key import (
    BEVDepthLightningModel,
)


def _voxel_pooling_inference_fallback(
    geom_xyz: torch.Tensor,
    depth_features: torch.Tensor,
    context_features: torch.Tensor,
    voxel_num: torch.Tensor,
) -> torch.Tensor:
    """Pure PyTorch replacement for the CUDA voxel pooling op.

    This function handles flexible input sizes by inferring spatial dimensions
    from the actual tensor shapes rather than assuming they match geom_xyz.
    """
    device = context_features.device
    if isinstance(voxel_num, torch.Tensor):
        voxel_sizes = voxel_num.detach().cpu().tolist()
    else:
        voxel_sizes = voxel_num if isinstance(voxel_num, (list, tuple)) else [voxel_num]
    num_voxel_x, num_voxel_y, num_voxel_z = [int(v) for v in voxel_sizes]

    B, num_cams, num_depth, num_height, num_width, _ = geom_xyz.shape
    channels = context_features.shape[1]
    depth_channels = depth_features.shape[1]

    # depth_features and context_features come in as (B*num_cams, C, H, W)
    # Infer actual spatial dimensions from tensor shapes (flexible for different input sizes)
    if len(depth_features.shape) == 4:
        # depth_features: (B*num_cams, depth_channels, H, W)
        _, _, actual_height, actual_width = depth_features.shape
    else:
        # Fallback: infer from total elements
        depth_total_elements = depth_features.numel()
        spatial_size = depth_total_elements // (B * num_cams * depth_channels)
        # Try to match geom_xyz aspect ratio, then adjust
        if spatial_size == num_height * num_width:
            actual_height, actual_width = num_height, num_width
        else:
            # Infer from spatial size - try to maintain aspect ratio
            aspect_ratio = num_width / num_height if num_height > 0 else 1.0
            actual_width = int((spatial_size * aspect_ratio) ** 0.5)
            actual_height = spatial_size // actual_width
            # Ensure valid dimensions
            while actual_height * actual_width != spatial_size and actual_width > 0:
                actual_width -= 1
                actual_height = spatial_size // actual_width if actual_width > 0 else 1

    # Reshape depth_features and context_features from (B*num_cams, ...) to (B, num_cams, ...)
    # Use actual spatial dimensions from tensor shapes
    depth = depth_features.view(B, num_cams, depth_channels, actual_height, actual_width)
    # Extract first num_depth channels (depth_channels should be >= num_depth)
    if depth_channels >= num_depth:
        depth = depth[:, :, :num_depth, :, :]
    else:
        # If fewer channels, pad or handle gracefully
        raise ValueError(f"depth_channels ({depth_channels}) < num_depth ({num_depth})")

    # context_features: (B*num_cams, channels, H, W)
    if len(context_features.shape) == 4:
        _, _, ctx_height, ctx_width = context_features.shape
        if ctx_height != actual_height or ctx_width != actual_width:
            # Reshape to match depth spatial dimensions if needed
            context_features = context_features.view(B, num_cams, channels, ctx_height, ctx_width)
            # Interpolate if dimensions don't match
            if ctx_height != actual_height or ctx_width != actual_width:
                context_features = torch.nn.functional.interpolate(
                    context_features.view(B * num_cams, channels, ctx_height, ctx_width),
                    size=(actual_height, actual_width),
                    mode="bilinear",
                    align_corners=False,
                ).view(B, num_cams, channels, actual_height, actual_width)
        else:
            context_features = context_features.view(B, num_cams, channels, actual_height, actual_width)
    else:
        context_features = context_features.view(B, num_cams, channels, actual_height, actual_width)

    context = context_features.permute(0, 1, 3, 4, 2).contiguous().unsqueeze(2).expand(-1, -1, num_depth, -1, -1, -1)

    # Reshape geom_xyz to match actual feature map dimensions if they differ
    if num_height != actual_height or num_width != actual_width:
        # Reshape geom_xyz from (B, num_cams, num_depth, num_height, num_width, 3)
        # to (B, num_cams, num_depth, actual_height, actual_width, 3)
        # First flatten spatial dimensions, then reshape
        geom_xyz_flat = geom_xyz.view(B, num_cams, num_depth, num_height * num_width, 3)
        # If total elements match, we can reshape directly
        if num_height * num_width == actual_height * actual_width:
            geom_xyz = geom_xyz_flat.view(B, num_cams, num_depth, actual_height, actual_width, 3)
        else:
            # If dimensions don't match, we need to interpolate the coordinates
            # Interpolate each coordinate channel separately
            # Convert to float for interpolation (interpolate requires float)
            geom_xyz_reshaped = geom_xyz.view(B, num_cams, num_depth, num_height, num_width, 3).float()
            # Permute to (B, num_cams, num_depth, 3, num_height, num_width) for interpolation
            geom_xyz_perm = geom_xyz_reshaped.permute(0, 1, 2, 5, 3, 4).contiguous()
            geom_xyz_interp = torch.nn.functional.interpolate(
                geom_xyz_perm.view(B * num_cams * num_depth, 3, num_height, num_width),
                size=(actual_height, actual_width),
                mode="bilinear",
                align_corners=False,
            )
            # Reshape back to (B, num_cams, num_depth, actual_height, actual_width, 3)
            # Keep as float for now, will convert to long when extracting coordinates
            geom_xyz = (
                geom_xyz_interp.view(B, num_cams, num_depth, 3, actual_height, actual_width)
                .permute(0, 1, 2, 4, 5, 3)
                .contiguous()
            )

    geom = geom_xyz.long()
    x = geom[..., 0]
    y = geom[..., 1]
    z = geom[..., 2]

    valid_mask = (x >= 0) & (x < num_voxel_x) & (y >= 0) & (y < num_voxel_y) & (z >= 0) & (z < num_voxel_z)
    valid = valid_mask.to(depth.dtype)

    depth = depth.unsqueeze(-1)
    contributions = depth * context * valid.unsqueeze(-1)

    batch_indices = torch.arange(B, device=device).view(B, 1, 1, 1, 1)
    batch_indices = batch_indices.expand_as(depth[..., 0])

    x = x.clamp(0, num_voxel_x - 1)
    y = y.clamp(0, num_voxel_y - 1)

    flat_index = batch_indices * (num_voxel_y * num_voxel_x) + y * num_voxel_x + x

    bev = torch.zeros(B * num_voxel_y * num_voxel_x, channels, device=device, dtype=context_features.dtype)
    bev.index_add_(0, flat_index.view(-1).long(), contributions.view(-1, channels))
    bev = bev.view(B, num_voxel_y, num_voxel_x, channels).permute(0, 3, 1, 2).contiguous()
    return bev


def enable_voxel_pooling_fallback():
    """Monkey-patch BEVDepth to use the PyTorch voxel pooling path."""
    from models.experimental.BevDepth.reference.bevdepth.layers.backbones import base_lss_fpn

    try:
        from models.experimental.BevDepth.reference.bevdepth.ops import voxel_pooling_inference
    except ImportError:
        voxel_pooling_inference = None

    def _fallback(*args, **kwargs):
        return _voxel_pooling_inference_fallback(*args, **kwargs)

    base_lss_fpn.voxel_pooling_inference = _fallback
    if voxel_pooling_inference is not None:
        voxel_pooling_inference.voxel_pooling_inference = _fallback
    print("✓ Voxel pooling fallback enabled for ONNX export")


def enable_dcn_fallback():
    """Replace DCN (deformable conv) with a standard Conv2d for ONNX export.

    ONNX doesn't natively support deformable convolution, so we need to
    replace DCN layers with regular Conv2d for export compatibility.
    """
    try:
        # Import the build_conv_layer function
        from models.experimental.BevDepth.reference.bevdepth.layers.heads.conv import build_conv_layer
        from models.experimental.BevDepth.reference.bevdepth.layers.backbones import base_lss_fpn

        # Store original function
        original_build_conv_layer = build_conv_layer

        def build_conv_layer_wrapper(cfg, *args, **kwargs):
            if cfg is not None and cfg.get("type") in {"DCN", "DCNv2", "DeformConv2d", "ModulatedDeformConv2d"}:
                # Replace DCN with Conv2d for ONNX export
                cfg = cfg.copy()
                cfg["type"] = "Conv2d"
                # Remove DCN-specific parameters
                for key in ("deform_groups", "fallback_on_stride", "im2col_step", "with_modulated_dcn"):
                    cfg.pop(key, None)
                # Set bias=False to match typical DCN configuration
                if "bias" not in cfg:
                    cfg["bias"] = False
                print(
                    f"  Converting DCN to Conv2d for ONNX export: {cfg.get('in_channels')} -> {cfg.get('out_channels')}"
                )

            # Use the original build_conv_layer but with modified config
            return original_build_conv_layer(cfg, *args, **kwargs)

        # Monkey-patch the build_conv_layer in base_lss_fpn
        base_lss_fpn.build_conv_layer = build_conv_layer_wrapper

        # Also patch it in the conv module if needed
        import models.experimental.BevDepth.reference.bevdepth.layers.heads.conv as conv_module

        conv_module.build_conv_layer = build_conv_layer_wrapper

        print("✓ DCN fallback enabled for ONNX export (DCN -> Conv2d)")
    except Exception as exc:
        print(f"Warning: failed to enable DCN fallback ({exc})")
        import traceback

        traceback.print_exc()


def enable_linalg_inv_fallback():
    """Provide ONNX symbolic / runtime fallback for torch.linalg.inv."""
    try:
        if getattr(torch.linalg.inv, "__name__", "") != "_bevdepth_inverse":

            def _bevdepth_inverse(x):
                return torch.inverse(x)

            _bevdepth_inverse.__name__ = "_bevdepth_inverse"
            torch.linalg.inv = _bevdepth_inverse

        def _linalg_inv_symbolic(g, self):
            return g.op("com.microsoft::MatrixInverse", self)

        for opset in range(9, 19):
            register_custom_op_symbolic("aten::linalg_inv", _linalg_inv_symbolic, opset)
    except Exception as exc:
        print(f"Warning: failed to enable linalg.inv fallback ({exc})")


class BEVDepthExportWrapper(nn.Module):
    """Wrapper with proper temporal aggregation for 2-key model."""

    def __init__(self, core_model: nn.Module):
        super().__init__()
        self.core_model = core_model

        # Check if this is a 2-key model by inspecting the head's expected input channels
        try:
            expected_channels = core_model.head.trunk.conv1.in_channels
            backbone_channels = core_model.backbone.output_channels
            self.num_sweeps = expected_channels // backbone_channels

            print(f"Detected configuration:")
            print(f"  Backbone output channels: {backbone_channels}")
            print(f"  Head expected channels: {expected_channels}")
            print(f"  Number of sweeps: {self.num_sweeps}")

            self.needs_temporal_concat = self.num_sweeps > 1

        except Exception as e:
            print(f"Warning: Could not detect temporal configuration: {e}")
            self.needs_temporal_concat = False
            self.num_sweeps = 1

    def forward(
        self,
        sweep_imgs,
        sensor2ego_mats,
        intrin_mats,
        ida_mats,
        sensor2sensor_mats,
        bda_mat,
        timestamps,
    ):
        mats_dict = {
            "sensor2ego_mats": sensor2ego_mats,
            "intrin_mats": intrin_mats,
            "ida_mats": ida_mats,
            "sensor2sensor_mats": sensor2sensor_mats,
            "bda_mat": bda_mat,
        }

        if self.needs_temporal_concat and sweep_imgs.shape[1] == self.num_sweeps:
            # For 2-key model: process each sweep separately and concatenate
            print(f"Processing {self.num_sweeps} temporal sweeps separately...")

            bev_features_list = []

            for sweep_idx in range(self.num_sweeps):
                # Extract single sweep
                sweep_img = sweep_imgs[:, sweep_idx : sweep_idx + 1, ...]  # Keep sweep dimension
                sweep_mats = {
                    "sensor2ego_mats": sensor2ego_mats[:, sweep_idx : sweep_idx + 1, ...],
                    "intrin_mats": intrin_mats[:, sweep_idx : sweep_idx + 1, ...],
                    "ida_mats": ida_mats[:, sweep_idx : sweep_idx + 1, ...],
                    "sensor2sensor_mats": sensor2sensor_mats[:, sweep_idx : sweep_idx + 1, ...],
                    "bda_mat": bda_mat,  # BDA mat is per-batch, not per-sweep
                }

                # Get BEV features for this sweep
                bev_feat = self.core_model.backbone(sweep_img, sweep_mats)
                bev_features_list.append(bev_feat)

            # Concatenate features from all sweeps
            bev_features = torch.cat(bev_features_list, dim=1)
            print(f"Concatenated BEV features shape: {bev_features.shape}")

            # Pass to head
            return self.core_model.head(bev_features)
        else:
            # Standard forward (single sweep or already aggregated)
            return self.core_model(sweep_imgs, mats_dict, timestamps)


def build_model(
    ckpt_path: Optional[str] = None,
    device: Optional[torch.device] = None,
):
    """Build BEVDepth model, optionally loading a checkpoint."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create model
    lightning_model = BEVDepthLightningModel()
    core_model = lightning_model.model.to(device)

    # Load checkpoint if provided
    if ckpt_path is not None and os.path.isfile(ckpt_path):
        print(f"Loading checkpoint from: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device)

        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
                # Remove 'model.' prefix if present (from Lightning checkpoints)
                if any(k.startswith("model.") for k in state_dict.keys()):
                    state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
            elif "model" in checkpoint:
                state_dict = checkpoint["model"]
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        # Load weights
        missing_keys, unexpected_keys = core_model.load_state_dict(state_dict, strict=False)
        if missing_keys:
            print(f"⚠ Missing keys: {len(missing_keys)}")
        if unexpected_keys:
            print(f"⚠ Unexpected keys: {len(unexpected_keys)}")
        print("✓ Checkpoint loaded successfully")

    core_model.eval()
    return core_model, device


def _prepare_inputs(device: torch.device) -> Tuple[Tuple[torch.Tensor, ...], Tuple[str, ...]]:
    """
    Prepare input tensors for ONNX export.

    Supports flexible input sizes via environment variables:
    - BEVDEPTH_EXPORT_HEIGHT: Image height (default: 256)
    - BEVDEPTH_EXPORT_WIDTH: Image width (default: 640, supports 704 as well)
    - BEVDEPTH_EXPORT_BATCH: Batch size (default: 1)
    - BEVDEPTH_EXPORT_SWEEPS: Number of temporal sweeps (default: 2)

    Examples:
        # Export with 256x704 input
        BEVDEPTH_EXPORT_WIDTH=704 python export_onnx.py

        # Export with 256x640 input (default)
        python export_onnx.py
    """
    B = int(os.environ.get("BEVDEPTH_EXPORT_BATCH", 1))
    S = int(os.environ.get("BEVDEPTH_EXPORT_SWEEPS", 2))  # Default to 2 for 2-key
    N = 6
    C = 3
    H = int(os.environ.get("BEVDEPTH_EXPORT_HEIGHT", 256))
    W = int(os.environ.get("BEVDEPTH_EXPORT_WIDTH", 640))

    sweep_imgs = torch.rand(B, S, N, C, H, W, device=device)
    sensor2ego_mats = torch.rand(B, S, N, 4, 4, device=device)
    intrin_mats = torch.rand(B, S, N, 4, 4, device=device)
    ida_mats = torch.rand(B, S, N, 4, 4, device=device)
    sensor2sensor_mats = torch.rand(B, S, N, 4, 4, device=device)
    bda_mat = torch.rand(B, 4, 4, device=device)
    timestamps = torch.rand(B, S, N, device=device)

    inputs = (
        sweep_imgs,
        sensor2ego_mats,
        intrin_mats,
        ida_mats,
        sensor2sensor_mats,
        bda_mat,
        timestamps,
    )
    input_names = (
        "sweep_imgs",
        "sensor2ego_mats",
        "intrin_mats",
        "ida_mats",
        "sensor2sensor_mats",
        "bda_mat",
        "timestamps",
    )
    return inputs, input_names


def export_bevdepth_backbone(
    model: nn.Module,
    device: torch.device,
    output_path: Path,
    inputs: Tuple[torch.Tensor, ...],
    input_names: Tuple[str, ...],
):
    """Export backbone only (without temporal concatenation)."""

    class BackboneWrapper(nn.Module):
        def __init__(self, core: nn.Module):
            super().__init__()
            self.core = core

        def forward(self, *args):
            sweep_imgs, sensor2ego, intrin, ida, sensor2sensor, bda, timestamps = args
            mats = {
                "sensor2ego_mats": sensor2ego,
                "intrin_mats": intrin,
                "ida_mats": ida,
                "sensor2sensor_mats": sensor2sensor,
                "bda_mat": bda,
            }
            return self.core.backbone(sweep_imgs, mats)

    wrapper = BackboneWrapper(model).to(device).eval()

    # Define dynamic axes for flexible input sizes (supports both 256x704 and 256x640)
    dynamic_axes = {
        "sweep_imgs": {0: "batch_size", 4: "height", 5: "width"},  # (B, S, N, C, H, W)
        "sensor2ego_mats": {0: "batch_size"},
        "intrin_mats": {0: "batch_size"},
        "ida_mats": {0: "batch_size"},
        "sensor2sensor_mats": {0: "batch_size"},
        "bda_mat": {0: "batch_size"},
        "timestamps": {0: "batch_size"},
        "bev_features": {0: "batch_size"},  # Output can also vary
    }

    torch.onnx.export(
        wrapper,
        inputs,
        str(output_path),
        opset_version=18,
        input_names=list(input_names),  # Convert tuple to list
        output_names=["bev_features"],
        do_constant_folding=False,
        training=torch.onnx.TrainingMode.EVAL,
        dynamic_axes=dynamic_axes,
        verbose=False,
    )
    print(f"Backbone ONNX saved to {output_path}")


def export_bev_full(
    model: nn.Module,
    device: torch.device,
    output_path: Path,
    inputs: Tuple[torch.Tensor, ...],
    input_names: Tuple[str, ...],
):
    """Export full model with proper temporal handling."""
    wrapped = BEVDepthExportWrapper(model).to(device)
    wrapped.eval()

    # Test forward pass first
    print("\nTesting forward pass with wrapper...")
    with torch.no_grad():
        try:
            output = wrapped(*inputs)
            print(f"✓ Forward pass successful")
            if isinstance(output, (list, tuple)):
                print(f"  Output is list/tuple with {len(output)} elements")
            else:
                print(f"  Output shape: {output.shape}")
        except Exception as e:
            print(f"✗ Forward pass failed: {e}")
            import traceback

            traceback.print_exc()
            raise

    # Define dynamic axes for flexible input sizes (supports both 256x704 and 256x640)
    dynamic_axes = {
        "sweep_imgs": {0: "batch_size", 4: "height", 5: "width"},  # (B, S, N, C, H, W)
        "sensor2ego_mats": {0: "batch_size"},
        "intrin_mats": {0: "batch_size"},
        "ida_mats": {0: "batch_size"},
        "sensor2sensor_mats": {0: "batch_size"},
        "bda_mat": {0: "batch_size"},
        "timestamps": {0: "batch_size"},
        "bev_output": {0: "batch_size"},  # Output can also vary
    }

    torch.onnx.export(
        wrapped,
        inputs,
        str(output_path),
        opset_version=18,
        input_names=list(input_names),  # Convert tuple to list
        output_names=["bev_output"],
        do_constant_folding=False,
        training=torch.onnx.TrainingMode.EVAL,
        dynamic_axes=None,
        do_constant_folding=True,
        # dynamic_axes=dynamic_axes,
        verbose=False,
    )
    print(f"Full model ONNX saved to {output_path}")


def export_bevdepth_onnx(
    ckpt_path: Optional[str] = None,
    output_dir: str = "onnx_exports",
    export_full: bool = True,
    export_backbone: bool = True,
):
    """Main export function."""
    enable_voxel_pooling_fallback()
    enable_dcn_fallback()
    enable_linalg_inv_fallback()

    device_override = os.environ.get("BEVDEPTH_EXPORT_DEVICE")
    export_device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device_override is None
        else torch.device(device_override)
    )

    model, device = build_model(ckpt_path, export_device)

    # Print model info
    print("\n" + "=" * 60)
    print("Model Configuration")
    print("=" * 60)
    print(f"Backbone output channels: {model.backbone.output_channels}")
    print(f"Head input channels: {model.head.trunk.conv1.in_channels}")
    print(f"Expected sweeps: {model.head.trunk.conv1.in_channels // model.backbone.output_channels}")
    print("=" * 60 + "\n")

    inputs, input_names = _prepare_inputs(device)
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    if export_backbone:
        export_bevdepth_backbone(
            model,
            device,
            output_dir_path / "bevdepth_backbone_3.onnx",
            inputs,
            input_names,
        )

    if export_full:
        export_bev_full(
            model,
            device,
            output_dir_path / "bevdepth_full_4.onnx",
            inputs,
            input_names,
        )


if __name__ == "__main__":
    import sys

    # Get checkpoint path from command line or use default
    if len(sys.argv) > 1:
        ckpt_path = sys.argv[1]
    else:
        # Try to find checkpoint in the reference/checkpoints directory
        script_dir = Path(__file__).parent
        default_ckpt = script_dir / "reference" / "checkpoints" / "bev_depth_lss_r50_256x704_128x128_24e_2key.pth"
        ckpt_path = str(default_ckpt) if default_ckpt.is_file() else None

    if ckpt_path and not os.path.isfile(ckpt_path):
        print(f"Error: Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    print(f"Using checkpoint: {ckpt_path}")

    # Export with 2 sweeps for 2-key model
    os.environ["BEVDEPTH_EXPORT_SWEEPS"] = "2"

    export_bevdepth_onnx(
        ckpt_path,
        export_full=True,
        export_backbone=True,
    )
