import torch
import pytest
import ttnn
from loguru import logger

# Import the reference models (adjust import paths as needed)
from models.experimental.SSR.reference.SSR.model.tile_refinement import TileRefinement
from models.experimental.SSR.tests.test_patch_embed_tile_selection import create_patch_embed_preprocessor_conv
from models.experimental.SSR.tests.test_RHAG import create_rhag_preprocessor
from models.experimental.SSR.tests.test_upsample import create_upsample_preprocessor
from models.experimental.SSR.tt.tile_refinement import TTTileRefinement
from models.experimental.SSR.tests.test_HAB import create_relative_position_index
from models.utility_functions import profiler

from ttnn.model_preprocessing import preprocess_model_parameters
from models.utility_functions import (
    tt2torch_tensor,
    comp_pcc,
)


def create_tile_refinement_preprocessor(device, forward_params, window_size, rpi_sa):
    """Custom preprocessor for TileRefinement model"""

    def custom_preprocessor(torch_model, name, ttnn_module_args):
        parameters = {}
        parameters["forward_params"] = forward_params
        if isinstance(torch_model, TileRefinement):
            # Preprocess conv layers using ttnn.prepare_conv_weights
            conv_layers = ["conv_first", "conv_after_body", "conv_last"]

            for conv_name in conv_layers:
                if hasattr(torch_model, conv_name):
                    conv_layer = getattr(torch_model, conv_name)
                    if hasattr(conv_layer, "weight"):  # Direct conv layer
                        conv_config = ttnn.Conv2dConfig(weights_dtype=ttnn.bfloat16)
                        parameters[conv_name] = {
                            "weight": ttnn.from_torch(conv_layer.weight, dtype=ttnn.bfloat16),
                            "bias": ttnn.from_torch(conv_layer.bias.reshape(1, 1, 1, -1), dtype=ttnn.bfloat16),
                        }

            # Handle conv_before_upsample (Sequential layer)
            if hasattr(torch_model, "conv_before_upsample") and torch_model.conv_before_upsample is not None:
                conv_layer = torch_model.conv_before_upsample[0]  # Conv2d layer
                conv_config = ttnn.Conv2dConfig(weights_dtype=ttnn.bfloat16)
                parameters["conv_before_upsample"] = {
                    "weight": ttnn.from_torch(conv_layer.weight, dtype=ttnn.bfloat16),  # ttnn.prepare_conv_weights(
                    "bias": ttnn.from_torch(  # ttnn.prepare_conv_bias(
                        conv_layer.bias.reshape(1, 1, 1, -1), dtype=ttnn.bfloat16
                    ),
                }

            # Preprocess layer norm
            if hasattr(torch_model, "norm"):
                dim = torch_model.norm.weight.size(0)  # 180
                padded_dim = ((dim + 31) // 32) * 32  # Round up to nearest multiple of 32 = 192

                norm_weight_padded = torch.nn.functional.pad(torch_model.norm.weight, (0, padded_dim - dim))
                norm_bias_padded = torch.nn.functional.pad(torch_model.norm.bias, (0, padded_dim - dim))

                norm_weight = norm_weight_padded.view(1, 1, padded_dim // 32, 32)
                norm_bias = norm_bias_padded.view(1, 1, padded_dim // 32, 32)

                parameters["norm"] = {
                    "weight": ttnn.from_torch(
                        norm_weight, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
                    ),
                    "bias": ttnn.from_torch(
                        norm_bias, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
                    ),
                }

            # Preprocess relative position indices
            if hasattr(torch_model, "relative_position_index_SA"):
                parameters["relative_position_index_SA"] = ttnn.from_torch(
                    torch_model.relative_position_index_SA, dtype=ttnn.int32
                )

            if hasattr(torch_model, "relative_position_index_OCA"):
                parameters["relative_position_index_OCA"] = ttnn.from_torch(
                    torch_model.relative_position_index_OCA, dtype=ttnn.int32
                )
            if hasattr(torch_model, "patch_embed"):
                patch_embed_params = preprocess_model_parameters(
                    initialize_model=lambda: torch_model.patch_embed,
                    custom_preprocessor=create_patch_embed_preprocessor_conv(device),
                    device=device,
                )
                parameters["patch_embed"] = patch_embed_params

            if hasattr(torch_model, "upsample"):
                upsample_params = preprocess_model_parameters(
                    initialize_model=lambda: torch_model.upsample,
                    custom_preprocessor=create_upsample_preprocessor(device),
                    device=device,
                )
                parameters["upsample"] = upsample_params

            if hasattr(torch_model, "layers"):
                for i in range(len(torch_model.layers)):
                    rhag_params = preprocess_model_parameters(
                        initialize_model=lambda: torch_model.layers[i],
                        custom_preprocessor=create_rhag_preprocessor(
                            device, depth=6, window_size=window_size, rpi_sa=rpi_sa
                        ),
                        device=device,
                    )
                    parameters[f"layers.{i}"] = rhag_params

        return parameters

    return custom_preprocessor


@pytest.mark.parametrize(
    "img_size, patch_size, embed_dim, depths, num_heads, window_size, mlp_ratio, upscale, input_shape",
    [
        # Test configuration - adjust based on your requirements
        # (64, 1, 96, (2, 2), (6, 6), 7, 4, 2, (1, 3, 64, 64)),
        # (64, 1, 180, (2, 2, 2), (6, 6, 6), 16, 2, 4, (1, 3, 64, 64)),
        (64, 1, 180, (6, 6, 6, 6, 6, 6), (6, 6, 6, 6, 6, 6), 16, 2, 4, (3, 3, 64, 64)),
    ],
)
def test_tile_refinement(
    img_size, patch_size, embed_dim, depths, num_heads, window_size, mlp_ratio, upscale, input_shape
):
    """Test TTTileRefinement model against PyTorch reference"""

    # Create input tensor
    x = torch.randn(input_shape)
    overlap_ratio = 0.5
    # Create reference PyTorch model
    ref_model = TileRefinement(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=3,
        embed_dim=embed_dim,
        depths=depths,
        num_heads=num_heads,
        window_size=window_size,
        compress_ratio=3,
        squeeze_factor=30,
        conv_scale=0.01,
        overlap_ratio=overlap_ratio,
        mlp_ratio=mlp_ratio,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        ape=False,
        patch_norm=True,
        use_checkpoint=False,
        upscale=upscale,
        img_range=1.0,
        upsampler="pixelshuffle",
        resi_connection="1conv",
    )

    rpi_sa = create_relative_position_index((window_size, window_size))

    # Create attention mask for shifted windows (simplified for testing)
    attn_mask = None

    # Create RPI for OCAB
    overlap_win_size = int(window_size * overlap_ratio) + window_size
    rpi_oca = torch.zeros((window_size * window_size, overlap_win_size * overlap_win_size), dtype=torch.long)

    # Create params dictionary
    params = {"rpi_sa": rpi_sa, "attn_mask": attn_mask, "rpi_oca": rpi_oca}

    device = ttnn.open_device(device_id=0, l1_small_size=32768)

    tt_rpi_sa = ttnn.from_torch(rpi_sa, device=device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint32)

    tt_rpi_oca = ttnn.from_torch(rpi_oca, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.uint32)

    tt_params = {"rpi_sa": tt_rpi_sa, "attn_mask": attn_mask, "rpi_oca": tt_rpi_oca}

    ref_model.eval()

    # Get reference output (both image and features)
    with torch.no_grad():
        ref_output, ref_features = ref_model(x)
        # ref_output = ref_model(x)

    # Open TTNN device

    try:
        # Preprocess model parameters
        parameters = preprocess_model_parameters(
            initialize_model=lambda: ref_model,
            # custom_preprocessor=create_tile_refinement_preprocessor(device, params),
            custom_preprocessor=create_tile_refinement_preprocessor(device, tt_params, window_size, rpi_sa),
            device=device,
        )
        # import pdb; pdb.set_trace()

        # Create TTNN model
        tt_model = TTTileRefinement(
            device=device,
            parameters=parameters,
            img_size=img_size,
            patch_size=patch_size,
            in_chans=3,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            compress_ratio=3,
            squeeze_factor=30,
            conv_scale=0.01,
            overlap_ratio=0.5,
            mlp_ratio=mlp_ratio,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.1,
            ape=False,
            patch_norm=True,
            upscale=upscale,
            img_range=1.0,
            upsampler="pixelshuffle",
            resi_connection="1conv",
        )

        # Convert input to TTNN tensor
        tt_input = ttnn.from_torch(x, device=device, layout=ttnn.TILE_LAYOUT)

        # Run TTNN model
        profiler.start("warmUpRun")
        tt_output, tt_features = tt_model(tt_input)
        profiler.end("warmUpRun")

        profiler.start("actualRun")
        tt_output, tt_features = tt_model(tt_input)
        profiler.end("actualRun")

        # Convert back to torch tensors
        tt_torch_output = tt2torch_tensor(tt_output)
        tt_torch_features = tt2torch_tensor(tt_features)

        warmup_run_time = profiler.get("warmUpRun")
        actual_run_time = profiler.get("actualRun")

        tt_torch_output = tt_torch_output.permute(0, 3, 1, 2)
        tt_torch_features = tt_torch_features.permute(0, 3, 1, 2)

        # Compare outputs
        output_pass, output_pcc_message = comp_pcc(ref_output, tt_torch_output, 0.90)
        features_pass, features_pcc_message = comp_pcc(ref_features, tt_torch_features, 0.90)

        logger.info(f"Output PCC: {output_pcc_message}")
        logger.info(f"Features PCC: {features_pcc_message}")

        logger.info(f"Warmup Run: {warmup_run_time} s")
        logger.info(f"Actual Run: {actual_run_time} s")
        logger.info(f"Throughput: {input_shape[0]/actual_run_time}")

        if output_pass and features_pass:
            # if output_pass:  # and features_pass:
            logger.info("TTTileRefinement Test Passed!")
        else:
            logger.warning("TTTileRefinement Test Failed!")

        assert output_pass, f"Output comparison failed: {output_pcc_message}"
        assert features_pass, f"Features comparison failed: {features_pcc_message}"

    finally:
        ttnn.close_device(device)
