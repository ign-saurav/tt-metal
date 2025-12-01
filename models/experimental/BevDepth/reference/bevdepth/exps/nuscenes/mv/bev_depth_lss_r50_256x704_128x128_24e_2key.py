# Copyright (c) Megvii Inc. All rights reserved.
"""
mAP: 0.3304
mATE: 0.7021
mASE: 0.2795
mAOE: 0.5346
mAVE: 0.5530
mAAE: 0.2274
NDS: 0.4355
Eval time: 171.8s

Per-class results:
Object Class    AP      ATE     ASE     AOE     AVE     AAE
car     0.499   0.540   0.165   0.211   0.650   0.233
truck   0.278   0.719   0.218   0.265   0.547   0.215
bus     0.386   0.661   0.211   0.171   1.132   0.274
trailer 0.168   1.034   0.235   0.548   0.408   0.168
construction_vehicle    0.075   1.124   0.510   1.177   0.111   0.385
pedestrian      0.284   0.757   0.298   0.966   0.578   0.301
motorcycle      0.335   0.624   0.263   0.621   0.734   0.237
bicycle 0.305   0.554   0.264   0.653   0.263   0.006
traffic_cone    0.462   0.516   0.355   nan     nan     nan
barrier 0.512   0.491   0.275   0.200   nan     nan
"""
# from bevdepth.exps.base_cli import run_cli
from models.experimental.BevDepth.reference.bevdepth.exps.nuscenes.base_exp import (
    BEVDepthLightningModel as BaseBEVDepthLightningModel,
)
from models.experimental.BevDepth.reference.bevdepth.models.base_bev_depth import BaseBEVDepth


class BEVDepthLightningModel(BaseBEVDepthLightningModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.key_idxes = [-1]
        self.head_conf["bev_backbone_conf"]["in_channels"] = 80 * (len(self.key_idxes) + 1)
        self.head_conf["bev_neck_conf"]["in_channels"] = [80 * (len(self.key_idxes) + 1), 160, 320, 640]
        self.head_conf["train_cfg"]["code_weights"] = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        self.model = BaseBEVDepth(self.backbone_conf, self.head_conf, is_train_depth=True)


def get_bevdepth_model(**kwargs):
    """
    Instantiates the BEVDepthLightningModel and returns the underlying self.model.

    Args:
        **kwargs: Keyword arguments for the BEVDepthLightningModel.
    Returns:
        BaseBEVDepth: The instantiated model.
    """
    lightning_model = BEVDepthLightningModel(**kwargs)
    return lightning_model.model


if __name__ == "__main__":
    import torch
    import os

    # Get model structure
    model = get_bevdepth_model()
    print("=" * 50)
    print("Model Structure:")
    print("=" * 50)
    print(model)

    # Load real weights from checkpoint
    # Checkpoint is at: .../BevDepth/reference/checkpoints/bev_depth_lss_r50_256x704_128x128_24e_2key.pth
    # File is at: .../BevDepth/reference/bevdepth/exps/nuscenes/mv/
    # Need to go up 4 levels to reach 'reference' directory
    file_dir = os.path.dirname(__file__)
    # Go up: mv -> nuscenes -> exps -> bevdepth -> reference
    for _ in range(4):
        file_dir = os.path.dirname(file_dir)
    checkpoint_path = os.path.join(file_dir, "checkpoints", "bev_depth_lss_r50_256x704_128x128_24e_2key.pth")

    print("\n" + "=" * 50)
    print("Loading weights from checkpoint...")
    print("=" * 50)
    print(f"Checkpoint path: {checkpoint_path}")

    if not os.path.exists(checkpoint_path):
        print(f"✗ Checkpoint file not found at: {checkpoint_path}")
        print("Please ensure the checkpoint file exists.")
    else:
        try:
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location="cpu")

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

            # Check which DCN implementation is available
            from models.experimental.BevDepth.reference.bevdepth.layers.heads.conv import (
                _MMCV_DCN_AVAILABLE,
                _TORCHVISION_DCN_AVAILABLE,
            )

            # Determine if we have proper DCN support (torchvision or MMCV)
            # torchvision is preferred (no compiled extensions needed, like uniad/vadv2)
            has_dcn_support = _TORCHVISION_DCN_AVAILABLE or _MMCV_DCN_AVAILABLE

            # Only filter out conv_offset keys if we're using Conv2d fallback (no DCN at all)
            if not has_dcn_support:
                filtered_state_dict = {}
                dcn_keys_info = []

                for key, value in state_dict.items():
                    # DCN layers have conv_offset parameters that don't exist in regular Conv2d
                    if "conv_offset" in key:
                        # Store info about DCN keys
                        base_key = key.replace(".conv_offset.weight", "").replace(".conv_offset.bias", "")
                        if base_key not in dcn_keys_info:
                            dcn_keys_info.append(base_key)
                    else:
                        filtered_state_dict[key] = value

                if dcn_keys_info:
                    print(
                        f"\n⚠ Warning: Checkpoint contains DCN (Deformable Convolution) layers, but no DCN implementation is available."
                    )
                    print(f"   The model is using Conv2d fallback instead of DCN.")
                    print(f"   This means:")
                    print(f"   - DCN-specific weights (conv_offset) will be ignored")
                    print(f"   - Model accuracy may differ from the original trained model")
                    print(f"   - To get correct behavior, install torchvision (pip install torchvision)")
                    print(
                        f"   - Affected layers: {', '.join(dcn_keys_info[:3])}{'...' if len(dcn_keys_info) > 3 else ''}"
                    )

                state_dict_to_load = filtered_state_dict
            else:
                # We have DCN support (torchvision or MMCV), so keep all keys including conv_offset
                state_dict_to_load = state_dict
                if _TORCHVISION_DCN_AVAILABLE:
                    if _MMCV_DCN_AVAILABLE:
                        print(
                            f"\nℹ Using torchvision's DCN implementation (primary option, no compiled extensions needed)"
                        )
                    else:
                        print(f"\nℹ Using torchvision's DCN implementation (MMCV extensions not available)")
                    print(f"   All DCN weights including conv_offset will be loaded correctly.")
                elif _MMCV_DCN_AVAILABLE:
                    print(f"\nℹ Using MMCV's DCN implementation (torchvision not available)")
                    print(f"   All DCN weights including conv_offset will be loaded correctly.")

            # Load weights into model
            missing_keys, unexpected_keys = model.load_state_dict(state_dict_to_load, strict=False)

            print(f"✓ Weights loaded successfully!")
            if missing_keys:
                print(f"\n⚠ Missing keys ({len(missing_keys)}):")
                for key in list(missing_keys)[:10]:  # Show first 10
                    print(f"  - {key}")
                if len(missing_keys) > 10:
                    print(f"  ... and {len(missing_keys) - 10} more")

            if unexpected_keys:
                print(f"\n⚠ Unexpected keys ({len(unexpected_keys)}):")
                for key in list(unexpected_keys)[:10]:  # Show first 10
                    print(f"  - {key}")
                if len(unexpected_keys) > 10:
                    print(f"  ... and {len(unexpected_keys) - 10} more")

            # Set model to evaluation mode
            model.eval()

            print("\n" + "=" * 50)
            print("Testing forward pass with loaded weights...")
            print("=" * 50)

            # Create dummy inputs matching BEVDepth's expected format
            batch_size = 1
            num_sweeps = 2  # For 2-key model
            num_cameras = 6  # Typical for nuScenes (6 cameras)
            img_h, img_w = 256, 704

            # 1. Images
            imgs = torch.randn(batch_size, num_sweeps, num_cameras, 3, img_h, img_w)

            # 2. Transformation matrices (mats_dict)
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

            print(f"Input shapes:")
            print(f"  imgs: {imgs.shape}")
            for key, val in mats_dict.items():
                print(f"  {key}: {val.shape}")

            # Run forward pass
            with torch.no_grad():
                try:
                    output = model(imgs, mats_dict)
                    print("\n✓ Forward pass successful!")

                    # Print output structure
                    if isinstance(output, list):
                        print(f"\nOutput: List with {len(output)} elements")
                        for i, task_output in enumerate(output):
                            print(f"\n  Task {i}:")
                            if isinstance(task_output, list):
                                for j, item in enumerate(task_output):
                                    if isinstance(item, dict):
                                        print(f"    Item {j} (dict):")
                                        for key, val in item.items():
                                            if isinstance(val, torch.Tensor):
                                                print(f"      {key}: {val.shape}")
                                    elif isinstance(item, torch.Tensor):
                                        print(f"    Item {j} (tensor): {item.shape}")
                            elif isinstance(task_output, dict):
                                print(f"    Dict with keys: {task_output.keys()}")
                                for key, val in task_output.items():
                                    if isinstance(val, torch.Tensor):
                                        print(f"      {key}: {val.shape}")
                    elif isinstance(output, dict):
                        print(f"\nOutput: Dict with keys: {output.keys()}")
                        for key, val in output.items():
                            if isinstance(val, torch.Tensor):
                                print(f"  {key}: {val.shape}")
                    else:
                        print(f"\nOutput type: {type(output)}")

                    print("\n" + "=" * 50)
                    print("✓ Model with real weights works correctly!")
                    print("=" * 50)

                except Exception as e:
                    print(f"\n✗ Forward pass failed: {e}")
                    import traceback

                    traceback.print_exc()

        except Exception as e:
            print(f"\n✗ Failed to load weights: {e}")
            import traceback

            traceback.print_exc()
# if __name__ == '__main__':
#     run_cli(BEVDepthLightningModel,
#             'bev_depth_lss_r50_256x704_128x128_24e_2key')
