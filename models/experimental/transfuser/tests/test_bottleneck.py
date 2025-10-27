import torch
import pytest
import ttnn
from models.experimental.transfuser.reference.bottleneck import Bottleneck as PyTorchBottleneck
from models.experimental.transfuser.tt.custom_preprocessing import create_custom_mesh_preprocessor
from models.experimental.transfuser.tt.bottleneck import TTRegNetBottleneck
from ttnn.model_preprocessing import (
    preprocess_model_parameters,
)
from tests.ttnn.utils_for_testing import check_with_pcc
from loguru import logger


# from models.experimental.transfuser.tt.ttn_bottleneck import TTNNBottleneck


def get_mesh_mappers(device):
    if device.get_num_devices() != 1:
        return (
            ttnn.ShardTensorToMesh(device, dim=0),
            None,
            ttnn.ConcatMeshToTensor(device, dim=0),
        )
    return None, None, None


def comp_pcc(golden, actual, pcc=0.99):
    """Compare tensors using PCC similar to codebase patterns."""
    golden_flat = golden.flatten()
    actual_flat = actual.flatten()

    correlation_matrix = torch.corrcoef(torch.stack([golden_flat, actual_flat]))
    pcc_value = correlation_matrix[0, 1].item()

    return pcc_value >= pcc, pcc_value


def preprocess_parameters_for_ttnn(torch_model, device):
    """Convert PyTorch parameters to TTNN tensors."""
    parameters = {}

    # Extract and convert all weights/biases to TTNN format
    conv1_weight = ttnn.from_torch(torch_model.conv1.conv.weight, device=device)
    conv1_bias = (
        ttnn.from_torch(torch_model.conv1.bn.bias, device=device) if torch_model.conv1.bn.bias is not None else None
    )

    conv2_weight = ttnn.from_torch(torch_model.conv2.conv.weight, device=device)
    conv2_bias = (
        ttnn.from_torch(torch_model.conv2.bn.bias, device=device) if torch_model.conv2.bn.bias is not None else None
    )

    conv3_weight = ttnn.from_torch(torch_model.conv3.conv.weight, device=device)
    conv3_bias = (
        ttnn.from_torch(torch_model.conv3.bn.bias, device=device) if torch_model.conv3.bn.bias is not None else None
    )

    # SE parameters (if exists)
    se_fc1_weight = se_fc1_bias = se_fc2_weight = se_fc2_bias = None
    if hasattr(torch_model.se, "fc1"):
        se_fc1_weight = ttnn.from_torch(torch_model.se.fc1.weight, device=device)
        se_fc1_bias = (
            ttnn.from_torch(torch_model.se.fc1.bias, device=device) if torch_model.se.fc1.bias is not None else None
        )
        se_fc2_weight = ttnn.from_torch(torch_model.se.fc2.weight, device=device)
        se_fc2_bias = (
            ttnn.from_torch(torch_model.se.fc2.bias, device=device) if torch_model.se.fc2.bias is not None else None
        )

    # Downsample parameters (if exists)
    downsample_weight = downsample_bias = None
    if torch_model.downsample is not None:
        downsample_weight = ttnn.from_torch(torch_model.downsample[0].weight, device=device)
        downsample_bias = (
            ttnn.from_torch(torch_model.downsample[1].bias, device=device)
            if torch_model.downsample[1].bias is not None
            else None
        )

    return {
        "conv1_weight": conv1_weight,
        "conv1_bias": conv1_bias,
        "conv2_weight": conv2_weight,
        "conv2_bias": conv2_bias,
        "conv3_weight": conv3_weight,
        "conv3_bias": conv3_bias,
        "se_fc1_weight": se_fc1_weight,
        "se_fc1_bias": se_fc1_bias,
        "se_fc2_weight": se_fc2_weight,
        "se_fc2_bias": se_fc2_bias,
        "downsample_weight": downsample_weight,
        "downsample_bias": downsample_bias,
    }


def extract_stage_bottleneck_image_encoder(ckpt_path, stage_name, bottleneck_num):
    """
    Extracts a specific stage.bottleneck block of image_encoder from a checkpoint,
    adjusts key names to match the model's state_dict.

    Args:
        ckpt_path (str): Path to the checkpoint.
        stage_name (str): Stage name (e.g., "s1", "s2", "s3", "s4").
        bottleneck_num (int): Bottleneck block number (e.g., 1, 2, 3, etc.).

    Returns:
        dict: Adjusted state_dict for stage.bottleneck of image_encoder.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)

    prefix = f"module._model.image_encoder.features.{stage_name}.b{bottleneck_num}."
    new_state_dict = {}

    for k, v in state_dict.items():
        if k.startswith(prefix):
            # Remove the prefix
            new_k = k[len(prefix) :]

            # Adjust downsample keys
            if new_k.startswith("downsample.conv"):
                new_k = new_k.replace("downsample.conv", "downsample.0")
            elif new_k.startswith("downsample.bn"):
                new_k = new_k.replace("downsample.bn", "downsample.1")

            new_state_dict[new_k] = v

    return new_state_dict


@pytest.mark.parametrize(
    "stage_name, bottleneck_num, in_chs, out_chs, stride",
    [
        ("s1", 1, 32, 72, 2),  # stage 1, bottleneck 1 (downsample)
        ("s1", 2, 72, 72, 1),  # stage 1, bottleneck 2 (no downsample, same channels)
        ("s2", 1, 72, 216, 2),  # stage 2, bottleneck 1 (downsample)
        ("s2", 2, 216, 216, 1),  # stage 2, bottleneck 2 (no downsample, same channels)
        ("s3", 1, 216, 576, 2),  # stage 3, bottleneck 1 (downsample)
        ("s3", 2, 576, 576, 1),  # stage 3, bottleneck 2 (no downsample, same channels)
        ("s4", 1, 576, 1512, 2),  # stage 4, bottleneck 1 (downsample)
        # Note: s4 does not have b2, so we only test b1
    ],
)
def test_regnet_bottleneck_pcc(stage_name, bottleneck_num, in_chs, out_chs, stride):
    """Test RegNet bottleneck with PCC assertion."""
    device = ttnn.open_device(device_id=0, l1_small_size=16384)

    try:
        # Map stage to layer number for loading correct features
        stage_to_layer = {"s1": "layer1", "s2": "layer2", "s3": "layer3", "s4": "layer4"}
        stage_to_shard = {
            "s1": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            "s2": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            "s3": ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            "s4": ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        }
        layer_num = stage_to_layer[stage_name]

        # Load actual features for the stage based on bottleneck number
        if bottleneck_num == 1:
            # For b1, use the original layer features
            feature_file = f"image_features_{layer_num}.pt"
            try:
                torch_input = torch.load(feature_file)
                logger.info(f"Loaded saved {layer_num} features (input to b1) with shape: {torch_input.shape}")
                print(f"{stage_name}.b{bottleneck_num} input shape: {torch_input.shape}")
            except FileNotFoundError:
                logger.warning(f"{feature_file} not found, using random input")
                input_sizes = {
                    "s1": (1, 32, 80, 352),
                    "s2": (1, 72, 40, 176),
                    "s3": (1, 216, 20, 88),
                    "s4": (1, 576, 10, 44),
                }
                torch_input = torch.randn(input_sizes.get(stage_name, (1, in_chs, 64, 64)))
                print(f"Using random input with shape: {torch_input.shape}")
        else:
            # For b2, use the output from b1
            feature_file = f"image_features_{layer_num}_b{bottleneck_num}_input.pt"
            try:
                torch_input = torch.load(feature_file)
                logger.info(f"Loaded saved input to {stage_name}.b{bottleneck_num} with shape: {torch_input.shape}")
                print(f"{stage_name}.b{bottleneck_num} input shape: {torch_input.shape}")
            except FileNotFoundError:
                logger.warning(f"{feature_file} not found, using random input")
                # Input to b2 matches output from b1 (same spatial size, channels match out_chs of stage)
                # Spatial dimensions reduced by stride from b1 input
                input_sizes_b2 = {
                    "s1": (1, 72, 40, 176),  # 80/2=40, 352/2=176
                    "s2": (1, 216, 20, 88),  # 40/2=20, 176/2=88
                    "s3": (1, 576, 10, 44),  # 20/2=10, 88/2=44
                    "s4": (1, 1512, 5, 22),  # 10/2=5, 44/2=22
                }
                torch_input = torch.randn(input_sizes_b2.get(stage_name, (1, out_chs, 32, 32)))
                print(f"Using random input with shape: {torch_input.shape}")

        # Determine if downsample is needed based on stride and channel change
        # For b2+, typically no downsample (stride=1, in_chs==out_chs)
        has_downsample = (stride != 1) or (in_chs != out_chs)

        # Create PyTorch model for reference
        torch_model = PyTorchBottleneck(
            in_chs=in_chs, out_chs=out_chs, stride=stride, group_size=24, downsample="conv1x1" if has_downsample else ""
        )
        checkpoint = torch.load("model_ckpt/models_2022/transfuser/model_seed1_39.pth", map_location="cpu")
        torch_model.eval()
        state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint

        # Filter only keys in image_encoder → features.{stage_name}.b{bottleneck_num}
        stage_bottleneck_image_encoder = {
            k: v for k, v in state_dict.items() if f"image_encoder.features.{stage_name}.b{bottleneck_num}." in k
        }

        # Save filtered checkpoint
        checkpoint_filename = f"checkpoint_image_encoder_{stage_name}_b{bottleneck_num}.pth"
        torch.save(stage_bottleneck_image_encoder, checkpoint_filename)

        # Extract and adjust state dict
        adjusted_state_dict = extract_stage_bottleneck_image_encoder(checkpoint_filename, stage_name, bottleneck_num)
        print(f"Extracted {len(adjusted_state_dict)} keys for {stage_name}.b{bottleneck_num}")
        print(f"Sample keys: {list(adjusted_state_dict.keys())[:5]}")

        # Optional: save adjusted
        adjusted_checkpoint_filename = f"checkpoint_image_encoder_{stage_name}_b{bottleneck_num}_adjusted.pth"
        torch.save(adjusted_state_dict, adjusted_checkpoint_filename)

        checkpoint = torch.load(adjusted_checkpoint_filename, map_location="cpu")

        torch_model.load_state_dict(checkpoint, strict=True)

        # # Reset BatchNorm statistics to default values for testing with random input
        # # This is necessary because the loaded checkpoint contains training statistics
        # # that don't match the random test input distribution
        # for module in torch_model.modules():
        #     if hasattr(module, "running_mean") and hasattr(module, "running_var"):
        #         module.running_mean.zero_()
        #         module.running_var.fill_(1.0)

        with torch.no_grad():
            torch_output = torch_model(torch_input)

        inputs_mesh_mapper, weights_mesh_mapper, output_mesh_composer = get_mesh_mappers(device)
        parameters = preprocess_model_parameters(
            initialize_model=lambda: torch_model,
            custom_preprocessor=create_custom_mesh_preprocessor(weights_mesh_mapper),
            device=None,
        )

        model_config = {
            "MATH_FIDELITY": ttnn.MathFidelity.LoFi,
            "WEIGHTS_DTYPE": ttnn.bfloat16,
            "ACTIVATIONS_DTYPE": ttnn.bfloat16,
        }
        # Determine if downsample is needed based on stride
        downsample = (stride == 2) or (in_chs != out_chs)
        print(f"downsample block bool check for {stage_name}.b{bottleneck_num}: {downsample}")

        bottle_ratio = 1.0
        group_size = 24
        bottleneck_chs = int(round(out_chs * bottle_ratio))
        groups = bottleneck_chs // group_size

        ttnn_model = TTRegNetBottleneck(
            parameters=parameters,
            model_config=model_config,
            stride=stride,
            downsample=downsample,
            groups=groups,
            shard_layout=stage_to_shard[stage_name],
        )
        tt_input = ttnn.from_torch(
            torch_input,
            # self.torch_image_input.permute(0, 2, 3, 1),
            dtype=ttnn.bfloat16,
            mesh_mapper=inputs_mesh_mapper,
        )
        tt_input = ttnn.to_device(tt_input, device)
        tt_input = ttnn.permute(tt_input, (0, 2, 3, 1))
        tt_output = ttnn_model(tt_input, device)
        tt_torch_output = ttnn.to_torch(
            tt_output,
            device=device,
            mesh_composer=output_mesh_composer,
        )
        expected_image_shape = torch_output.shape
        tt_torch_output = torch.reshape(
            tt_torch_output,
            (expected_image_shape[0], expected_image_shape[2], expected_image_shape[3], expected_image_shape[1]),
        )
        tt_torch_output = torch.permute(tt_torch_output, (0, 3, 1, 2))
        pcc_passed, pcc_message = check_with_pcc(torch_output, tt_torch_output, pcc=0.90)

        logger.info(f"Image Output PCC: {pcc_message}")
        assert pcc_passed, logger.error(f"PCC check failed - pcc_message: {pcc_message}")

        print(f"✓ RegNet bottleneck {stage_name}.b{bottleneck_num} TTNN implementation matches PyTorch with PCC > 0.90")

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    test_regnet_bottleneck_pcc()
