#!/usr/bin/env python3
"""
Extract and save inputs to the second bottleneck block (b2) of each layer.
This runs the first bottleneck (b1) on the input features and saves its output.
"""

import torch
import os
from models.experimental.transfuser.reference.bottleneck import Bottleneck


def extract_bottleneck_state_dict(checkpoint, stage_name, bottleneck_num):
    """
    Extract state dict for a specific bottleneck block.

    Args:
        checkpoint: Full checkpoint dict
        stage_name: 's1', 's2', 's3', or 's4'
        bottleneck_num: 1 or 2

    Returns:
        State dict for that specific bottleneck
    """
    state_dict = checkpoint.get("state_dict", checkpoint)

    prefix = f"module._model.image_encoder.features.{stage_name}.b{bottleneck_num}."
    bottleneck_dict = {}

    for k, v in state_dict.items():
        if k.startswith(prefix):
            # Remove the prefix
            new_key = k[len(prefix) :]

            # Adjust downsample keys
            if new_key.startswith("downsample.conv"):
                new_key = new_key.replace("downsample.conv", "downsample.0")
            elif new_key.startswith("downsample.bn"):
                new_key = new_key.replace("downsample.bn", "downsample.1")

            bottleneck_dict[new_key] = v

    return bottleneck_dict


def load_input_features(layer_num):
    """Load input features for a layer."""
    feature_file = f"image_features_{layer_num}.pt"
    if os.path.exists(feature_file):
        features = torch.load(feature_file)
        print(f"Loaded {feature_file} with shape: {features.shape}")
        return features
    else:
        print(f"Warning: {feature_file} not found, skipping this layer")
        return None


def get_bottleneck_channels(stage_name):
    """Get input and output channels for each stage's first bottleneck."""
    channels = {
        "s1": (32, 72, 2),  # in_chs, out_chs, stride
        "s2": (72, 216, 2),
        "s3": (216, 576, 2),
        "s4": (576, 1512, 2),
    }
    return channels.get(stage_name, (32, 32, 1))


def save_bottleneck_inputs():
    """Save inputs to b2 for each layer."""

    # Load checkpoint
    checkpoint_path = "model_ckpt/models_2022/transfuser/model_seed1_39.pth"
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    print(f"Loaded checkpoint from: {checkpoint_path}")

    # Map stages to layers
    stage_to_layer = {
        "s1": "layer1",
        "s2": "layer2",
        "s3": "layer3",
        "s4": "layer4",
    }

    # Process each stage
    for stage_name, layer_num in stage_to_layer.items():
        print(f"\n{'='*60}")
        print(f"Processing {stage_name} -> {layer_num}")
        print(f"{'='*60}")

        # Load input features for this layer
        input_features = load_input_features(layer_num)
        if input_features is None:
            continue

        # Get channel configuration for this stage
        in_chs, out_chs, stride = get_bottleneck_channels(stage_name)

        # Extract state dict for b1
        b1_state_dict = extract_bottleneck_state_dict(checkpoint, stage_name, bottleneck_num=1)

        if not b1_state_dict:
            print(f"Warning: No b1 weights found for {stage_name}")
            continue

        print(f"Found {len(b1_state_dict)} parameters for {stage_name}.b1")

        # Create b1 model
        b1_model = Bottleneck(
            in_chs=in_chs,
            out_chs=out_chs,
            stride=stride,
            group_size=24,
        )

        # Load weights
        b1_model.load_state_dict(b1_state_dict, strict=True)
        b1_model.eval()

        # Run b1 on input features
        with torch.no_grad():
            output_b1 = b1_model(input_features)

        print(f"Input shape: {input_features.shape}")
        print(f"Output from b1: {output_b1.shape}")

        # Save the output (which is the input to b2)
        output_filename = f"image_features_{layer_num}_b2_input.pt"
        torch.save(output_b1, output_filename)
        print(f"✅ Saved: {output_filename}")

        # Also save info about this layer
        info = {
            "input_shape": tuple(input_features.shape),
            "output_shape": tuple(output_b1.shape),
            "stage_name": stage_name,
            "layer_num": layer_num,
            "in_chs": in_chs,
            "out_chs": out_chs,
            "stride": stride,
        }
        info_filename = f"image_features_{layer_num}_b2_input_info.txt"
        with open(info_filename, "w") as f:
            for key, value in info.items():
                f.write(f"{key}: {value}\n")
        print(f"✅ Saved info: {info_filename}")

    print(f"\n{'='*60}")
    print("✅ All bottleneck inputs saved successfully!")
    print(f"{'='*60}")


if __name__ == "__main__":
    save_bottleneck_inputs()
