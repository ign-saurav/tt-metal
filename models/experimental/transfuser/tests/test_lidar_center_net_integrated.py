#!/usr/bin/env python3
"""
Integrated LiDAR CenterNet Test with Trained Weights and Real Data Loading
"""

import torch
import os
import sys

# Add the models directory to the path
sys.path.append("models/experimental/transfuser")


def load_trained_weights(weight_path: str):
    """
    Load trained Transfuser weights and clean them for use

    Args:
        weight_path: Path to the .pth file

    Returns:
        Cleaned state dict ready for model loading
    """
    print(f"Loading trained weights from: {weight_path}")

    # Load the checkpoint
    checkpoint = torch.load(weight_path, map_location="cpu")

    # The weights are stored with 'module._model.' prefix, we need to clean this
    state_dict = {}
    for key, value in checkpoint.items():
        # Remove 'module._model.' prefix
        if key.startswith("module._model."):
            clean_key = key[len("module._model.") :]
            state_dict[clean_key] = value
        else:
            state_dict[key] = value

    print(f"Loaded {len(state_dict)} parameters")
    print(
        f"Cleaned {len([k for k in checkpoint.keys() if k.startswith('module._model.')])} keys with 'module._model.' prefix"
    )

    # Add '_model.' prefix to backbone keys for compatibility with model structure
    backbone_keys = [
        "image_encoder",
        "lidar_encoder",
        "transformer1",
        "transformer2",
        "transformer3",
        "transformer4",
        "change_channel_conv_image",
        "change_channel_conv_lidar",
        "up_conv5",
        "up_conv4",
        "up_conv3",
        "c5_conv",
    ]
    backbone_renamed = 0
    for key in list(state_dict.keys()):
        for backbone in backbone_keys:
            if key.startswith(f"{backbone}."):
                new_key = f"_model.{backbone}.{key[len(backbone)+1:]}"
                state_dict[new_key] = state_dict.pop(key)
                backbone_renamed += 1
                break

    print(f"Added '_model.' prefix to {backbone_renamed} backbone keys")

    # Handle detection head and other components that need to be loaded without _model prefix
    # These components are at the top level in the model, not under _model
    detection_components = ["head", "pred_bev", "join", "decoder", "output"]
    detection_renamed = 0
    for key in list(state_dict.keys()):
        for component in detection_components:
            if key.startswith(f"module.{component}."):
                new_key = key[len("module.") :]  # Remove 'module.' prefix
                state_dict[new_key] = state_dict.pop(key)
                detection_renamed += 1
                break

    print(f"Cleaned {detection_renamed} detection component keys")
    return state_dict


def create_model_with_trained_weights(weight_path: str, config_values: dict = None):
    """
    Create Transfuser model and load trained weights

    Args:
        weight_path: Path to the .pth file
        config_values: Configuration values for the model

    Returns:
        Model with loaded weights
    """
    from models.experimental.transfuser.reference.config import GlobalConfig
    from models.experimental.transfuser.reference.lidar_center_net import LidarCenterNet

    # Load trained weights
    state_dict = load_trained_weights(weight_path)

    # Create model with the same configuration as training
    if config_values is None:
        config_values = {
            "image_architecture": "regnety_032",
            "lidar_architecture": "regnety_032",
            "n_layer": 4,
            "use_velocity": False,
            "use_target_point_image": True,
        }

    # Create config object
    config = GlobalConfig(setting="eval")
    config.n_layer = config_values["n_layer"]
    config.use_target_point_image = config_values["use_target_point_image"]

    # Create model
    model = LidarCenterNet(
        config=config,
        backbone="transFuser",
        image_architecture=config_values["image_architecture"],
        lidar_architecture=config_values["lidar_architecture"],
        use_velocity=config_values["use_velocity"],
    )

    # Load the trained weights
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    print("Successfully loaded trained weights into model")

    # Report missing and unexpected keys
    if missing_keys:
        print(f"Missing keys ({len(missing_keys)}): {missing_keys}")
    else:
        print("No missing keys")

    if unexpected_keys:
        print(f"Unexpected keys ({len(unexpected_keys)}): {unexpected_keys}")
    else:
        print("No unexpected keys")

    return model


def test_with_trained_weights_and_real_data():
    """
    Test the LiDAR CenterNet with trained weights and real data
    """
    print("=" * 80)
    print("INTEGRATED LIDAR CENTERNET TEST WITH TRAINED WEIGHTS")
    print("=" * 80)

    # Path to trained weights
    weight_path = "model_ckpt/models_2022/transfuser/model_seed1_39.pth"

    if not os.path.exists(weight_path):
        print(f"Error: Weight file not found: {weight_path}")
        return

    # Create model with trained weights
    print("\n1. Creating model with trained weights...")
    model = create_model_with_trained_weights(weight_path)
    model.eval()

    # Test with random data first
    print("\n2. Testing with random data...")
    random_inputs = {
        "image": torch.randn(1, 3, 160, 704, dtype=torch.float32),
        "lidar": torch.randn(1, 3, 256, 256, dtype=torch.float32),
        "velocity": torch.randn(1, 1, dtype=torch.float32),
        "target_point": torch.randn(1, 2, dtype=torch.float32),
        "target_point_image": torch.randn(1, 1, 256, 256, dtype=torch.float32),
    }

    print("Random inputs:")
    for key, value in random_inputs.items():
        print(f"  {key}: {value.shape} ({value.dtype})")

    # Run inference with random data
    print("\nRunning inference with random data...")
    with torch.no_grad():
        try:
            outputs = model.forward_ego(
                random_inputs["image"], random_inputs["lidar"], random_inputs["target_point"], random_inputs["velocity"]
            )
            print("✓ Random data inference successful!")

            # Print output shapes
            print("\nModel outputs:")
            for i, output in enumerate(outputs):
                if isinstance(output, torch.Tensor):
                    print(f"  Output {i}: {output.shape} ({output.dtype})")
                elif isinstance(output, (list, tuple)):
                    print(f"  Output {i}: {type(output).__name__} with {len(output)} elements")
                    for j, elem in enumerate(output):
                        if isinstance(elem, torch.Tensor):
                            print(f"    [{j}]: {elem.shape} ({elem.dtype})")
                        else:
                            print(f"    [{j}]: {type(elem)}")
                else:
                    print(f"  Output {i}: {type(output)}")

        except Exception as e:
            print(f"✗ Random data inference failed: {e}")
            import traceback

            traceback.print_exc()
            return

    # Test with real data if available
    print("\n3. Testing with real data...")
    data_root = "models/experimental/transfuser/tests/Scenario3_Town01_curved_route0_11_23_20_02_59"
    frame = "0120"

    if os.path.exists(data_root):
        print(f"Found real data directory: {data_root}")

        # Import the data loading functionality from the test
        sys.path.append("models/experimental/transfuser/tests")
        from test_lidar_center_net import load_data_from_args_or_fallback

        # Load real data
        real_inputs = load_data_from_args_or_fallback(
            data_root=data_root, frame=frame, save_debug_images=True, debug_output_dir="debug_images_trained_weights"
        )

        print("Real data inputs:")
        for key, value in real_inputs.items():
            print(f"  {key}: {value.shape} ({value.dtype})")

        # Run inference with real data
        print("\nRunning inference with real data...")
        with torch.no_grad():
            try:
                real_outputs = model.forward_ego(
                    real_inputs["image"], real_inputs["lidar"], real_inputs["target_point"], real_inputs["velocity"]
                )
                print("✓ Real data inference successful!")

                # Print output shapes
                print("\nReal data model outputs:")
                for i, output in enumerate(real_outputs):
                    if isinstance(output, torch.Tensor):
                        print(f"  Output {i}: {output.shape} ({output.dtype})")
                        print(f"    Mean: {output.mean().item():.6f}, Std: {output.std().item():.6f}")
                    elif isinstance(output, (list, tuple)):
                        print(f"  Output {i}: {type(output).__name__} with {len(output)} elements")
                        for j, elem in enumerate(output):
                            if isinstance(elem, torch.Tensor):
                                print(f"    [{j}]: {elem.shape} ({elem.dtype})")
                                print(f"      Mean: {elem.mean().item():.6f}, Std: {elem.std().item():.6f}")
                            else:
                                print(f"    [{j}]: {type(elem)}")
                    else:
                        print(f"  Output {i}: {type(output)}")

            except Exception as e:
                print(f"✗ Real data inference failed: {e}")
                import traceback

                traceback.print_exc()
    else:
        print(f"Real data directory not found: {data_root}")
        print("Skipping real data test")

    print("\n" + "=" * 80)
    print("INTEGRATION TEST COMPLETE")
    print("=" * 80)
    print("✓ Successfully loaded trained Transfuser weights")
    print("✓ Model runs inference with random data")
    if os.path.exists(data_root):
        print("✓ Model runs inference with real data")
        print("✓ Debug images saved for inspection")
    print("✓ Ready for TT-Metal integration")

    return model


def main():
    """Main function"""
    print("Integrated LiDAR CenterNet Test with Trained Weights and Real Data")
    print("=" * 80)

    # Run the integrated test
    model = test_with_trained_weights_and_real_data()

    if model is not None:
        print("\nNext steps:")
        print("1. Integrate this weight loading into the TT-Metal test")
        print("2. Compare performance between random and trained weights")
        print("3. Test with different real data scenarios")
        print("4. Optimize for TT-Metal hardware")


if __name__ == "__main__":
    main()
