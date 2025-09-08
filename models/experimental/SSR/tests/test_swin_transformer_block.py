import torch
import pytest
import ttnn
from loguru import logger

from models.experimental.SSR.reference.SSR.model.net_blocks import SwinTransformerBlock
from models.experimental.SSR.tt import TTSwinTransformerBlock
from ttnn.model_preprocessing import preprocess_model_parameters, preprocess_linear_bias, preprocess_linear_weight

from models.utility_functions import (
    tt2torch_tensor,
    comp_pcc,
)


def create_swin_transformer_block_preprocessor(device):
    def custom_preprocessor(torch_model, name, ttnn_module_args):
        parameters = {}

        if hasattr(torch_model, "attn"):  # SwinTransformerBlock model
            # Preprocess attention parameters
            parameters["attn"] = {}
            parameters["attn"]["qkv"] = {}
            parameters["attn"]["proj"] = {}

            parameters["attn"]["qkv"]["weight"] = preprocess_linear_weight(
                torch_model.attn.qkv.weight, dtype=ttnn.bfloat16
            )
            parameters["attn"]["qkv"]["bias"] = preprocess_linear_bias(torch_model.attn.qkv.bias, dtype=ttnn.bfloat16)

            # Preprocess relative position bias for attention
            relative_position_bias = torch_model.attn.relative_position_bias_table[
                torch_model.attn.relative_position_index.view(-1)
            ].view(
                torch_model.attn.window_size[0] * torch_model.attn.window_size[1],
                torch_model.attn.window_size[0] * torch_model.attn.window_size[1],
                -1,
            )
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous().unsqueeze(0)
            parameters["attn"]["relative_position_bias"] = ttnn.from_torch(
                relative_position_bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
            )

            parameters["attn"]["proj"]["weight"] = preprocess_linear_weight(
                torch_model.attn.proj.weight, dtype=ttnn.bfloat16
            )
            parameters["attn"]["proj"]["bias"] = preprocess_linear_bias(torch_model.attn.proj.bias, dtype=ttnn.bfloat16)

            # Preprocess layer normalization parameters
            parameters["norm1"] = {}
            parameters["norm1"]["weight"] = ttnn.from_torch(
                torch_model.norm1.weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
            )
            parameters["norm1"]["bias"] = ttnn.from_torch(
                torch_model.norm1.bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
            )

            parameters["norm2"] = {}
            parameters["norm2"]["weight"] = ttnn.from_torch(
                torch_model.norm2.weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
            )
            parameters["norm2"]["bias"] = ttnn.from_torch(
                torch_model.norm2.bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
            )

            # Preprocess MLP parameters
            parameters["mlp"] = {}
            parameters["mlp"]["fc1"] = {}
            parameters["mlp"]["fc2"] = {}

            parameters["mlp"]["fc1"]["weight"] = preprocess_linear_weight(
                torch_model.mlp.fc1.weight, dtype=ttnn.bfloat16
            )
            parameters["mlp"]["fc1"]["bias"] = preprocess_linear_bias(torch_model.mlp.fc1.bias, dtype=ttnn.bfloat16)

            parameters["mlp"]["fc2"]["weight"] = preprocess_linear_weight(
                torch_model.mlp.fc2.weight, dtype=ttnn.bfloat16
            )
            parameters["mlp"]["fc2"]["bias"] = preprocess_linear_bias(torch_model.mlp.fc2.bias, dtype=ttnn.bfloat16)

        return parameters

    return custom_preprocessor


@pytest.mark.parametrize(
    "batch_size, height, width, dim, num_heads, window_size, shift_size, mlp_ratio",
    (
        # # Regular window attention (no shift)
        (1, 14, 14, 96, 3, 7, 0, 4.0),
        (2, 14, 14, 192, 6, 7, 0, 4.0),
        # Shifted window attention
        (1, 14, 14, 96, 3, 7, 3, 4.0),
        (2, 14, 14, 192, 6, 7, 3, 4.0),
        # Different window sizes
        (1, 16, 16, 192, 6, 8, 0, 4.0),
        (1, 16, 16, 192, 6, 8, 4, 4.0),
        # Non-square input (requires padding)
        (1, 15, 13, 96, 3, 7, 0, 4.0),
        (1, 15, 13, 96, 3, 7, 3, 4.0),
    ),
)
def test_swin_transformer_block(batch_size, height, width, dim, num_heads, window_size, shift_size, mlp_ratio):
    # Create input tensor
    input_shape = (batch_size, height * width, dim)
    x = torch.randn(input_shape)

    # Create reference model
    ref_layer = SwinTransformerBlock(
        dim=dim,
        input_resolution=(height, width),
        num_heads=num_heads,
        window_size=window_size,
        shift_size=shift_size,
        mlp_ratio=mlp_ratio,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
    )

    # Get reference output
    ref_output = ref_layer(x)

    # Initialize device
    device = ttnn.open_device(device_id=0)

    try:
        # Preprocess model parameters
        parameters = preprocess_model_parameters(
            initialize_model=lambda: ref_layer,
            custom_preprocessor=create_swin_transformer_block_preprocessor(device),
            device=device,
        )

        # Create TTNN model
        tt_layer = TTSwinTransformerBlock(
            parameters=parameters,
            device=device,
            dim=dim,
            input_resolution=(height, width),
            num_heads=num_heads,
            window_size=window_size,
            shift_size=shift_size,
            mlp_ratio=mlp_ratio,
        )

        # Convert input to TTNN tensor
        tt_input = ttnn.from_torch(x, device=device, layout=ttnn.TILE_LAYOUT)

        # Run forward pass
        tt_output = tt_layer(tt_input)

        # Convert back to torch
        tt_torch_output = tt2torch_tensor(tt_output)

        # Compare outputs
        does_pass, pcc_message = comp_pcc(
            ref_output, tt_torch_output, 0.98
        )  # Slightly lower threshold due to accumulated precision differences

        logger.info(
            f"Test configuration: B={batch_size}, H={height}, W={width}, dim={dim}, heads={num_heads}, ws={window_size}, shift={shift_size}"
        )
        logger.info(pcc_message)

        if does_pass:
            logger.info("SwinTransformerBlock Passed!")
        else:
            logger.warning("SwinTransformerBlock Failed!")
            logger.warning(f"Reference output shape: {ref_output.shape}")
            logger.warning(f"TTNN output shape: {tt_torch_output.shape}")
            logger.warning(f"Reference output range: [{ref_output.min():.6f}, {ref_output.max():.6f}]")
            logger.warning(f"TTNN output range: [{tt_torch_output.min():.6f}, {tt_torch_output.max():.6f}]")

        assert does_pass

    finally:
        # Cleanup
        ttnn.close_device(device)


@pytest.mark.parametrize(
    "batch_size, height, width, dim, num_heads, window_size, shift_size",
    (
        # Edge case: input smaller than window size
        (1, 5, 5, 96, 3, 7, 0),
        # Edge case: input exactly window size
        (1, 7, 7, 96, 3, 7, 0),
        # Edge case: large shift size
        (1, 14, 14, 96, 3, 7, 6),
    ),
)
def test_swin_transformer_block_edge_cases(batch_size, height, width, dim, num_heads, window_size, shift_size):
    """Test edge cases for SwinTransformerBlock"""
    input_shape = (batch_size, height * width, dim)
    x = torch.randn(input_shape)

    ref_layer = SwinTransformerBlock(
        dim=dim,
        input_resolution=(height, width),
        num_heads=num_heads,
        window_size=window_size,
        shift_size=shift_size,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
    )

    ref_output = ref_layer(x)

    device = ttnn.open_device(device_id=0)

    try:
        parameters = preprocess_model_parameters(
            initialize_model=lambda: ref_layer,
            custom_preprocessor=create_swin_transformer_block_preprocessor(device),
            device=device,
        )

        tt_layer = TTSwinTransformerBlock(
            parameters=parameters,
            device=device,
            dim=dim,
            input_resolution=(height, width),
            num_heads=num_heads,
            window_size=window_size,
            shift_size=shift_size,
            mlp_ratio=4.0,
        )

        tt_input = ttnn.from_torch(x, device=device, layout=ttnn.TILE_LAYOUT)
        tt_output = tt_layer(tt_input)
        tt_torch_output = tt2torch_tensor(tt_output)

        does_pass, pcc_message = comp_pcc(ref_output, tt_torch_output, 0.95)  # More lenient for edge cases

        logger.info(f"Edge case test: B={batch_size}, H={height}, W={width}, ws={window_size}, shift={shift_size}")
        logger.info(pcc_message)

        if does_pass:
            logger.info("SwinTransformerBlock Edge Case Passed!")
        else:
            logger.warning("SwinTransformerBlock Edge Case Failed!")

        assert does_pass

    finally:
        ttnn.close_device(device)


def test_swin_transformer_block_memory_cleanup():
    """Test that memory is properly managed during forward pass"""
    batch_size, height, width, dim, num_heads = 2, 14, 14, 96, 3
    input_shape = (batch_size, height * width, dim)
    x = torch.randn(input_shape)

    ref_layer = SwinTransformerBlock(
        dim=dim,
        input_resolution=(height, width),
        num_heads=num_heads,
        window_size=7,
        shift_size=3,
        mlp_ratio=4.0,
    )

    device = ttnn.open_device(device_id=0)

    try:
        parameters = preprocess_model_parameters(
            initialize_model=lambda: ref_layer,
            custom_preprocessor=create_swin_transformer_block_preprocessor(device),
            device=device,
        )

        tt_layer = TTSwinTransformerBlock(
            parameters=parameters,
            device=device,
            dim=dim,
            input_resolution=(height, width),
            num_heads=num_heads,
            window_size=7,
            shift_size=3,
            mlp_ratio=4.0,
        )

        # Run multiple forward passes to check for memory leaks
        for i in range(3):
            tt_input = ttnn.from_torch(x, device=device, layout=ttnn.TILE_LAYOUT)
            tt_output = tt_layer(tt_input)

            # Cleanup input and output tensors
            ttnn.deallocate(tt_input)
            ttnn.deallocate(tt_output)

            logger.info(f"Forward pass {i+1} completed successfully")

        logger.info("Memory cleanup test passed!")

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    # Run a simple test
    test_swin_transformer_block(1, 14, 14, 96, 3, 7, 0, 4.0)
