import torch
import pytest
import ttnn
import math
from loguru import logger
from ttnn.model_preprocessing import preprocess_model_parameters, preprocess_linear_bias, preprocess_linear_weight
from models.experimental.SSR.reference.SSR.model.tile_selection import TileSelection
from models.experimental.SSR.tt.tile_selection import TTTileSelection
from models.experimental.SSR.tests.test_patch_embed import create_patch_embed_preprocessor
from models.experimental.SSR.tests.test_basic_block import create_basic_layer_preprocessor
from models.experimental.SSR.tests.test_mlp import create_mlp_preprocessor
from models.experimental.SSR.tests.test_mask_token_inference import create_mask_token_inference_preprocessor
from models.utility_functions import tt2torch_tensor, comp_pcc


def create_tile_selection_preprocessor(device, dim=96):
    def custom_preprocessor(torch_model, name, ttnn_module_args):
        parameters = {}

        # Handle mask token embedding
        if hasattr(torch_model, "mask_token"):
            parameters["mask_token"] = {}
            parameters["mask_token"]["weight"] = ttnn.from_torch(
                torch_model.mask_token.weight, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT
            )

        # Handle patch embedding - delegate to existing TTPatchEmbed preprocessor
        if hasattr(torch_model, "patch_embed"):
            patch_embed_params = preprocess_model_parameters(
                initialize_model=lambda: torch_model.patch_embed,
                custom_preprocessor=create_patch_embed_preprocessor(device),
                device=device,
            )
            parameters["patch_embed"] = patch_embed_params

        # Handle encoder layers - delegate to existing TTBasicLayer preprocessor
        if hasattr(torch_model, "layers"):
            for i, layer in enumerate(torch_model.layers):
                layer_dim = int(dim * 2**i)
                layer_params = preprocess_model_parameters(
                    initialize_model=lambda l=layer: l,
                    custom_preprocessor=create_basic_layer_preprocessor(device, layer_dim),
                    device=device,
                )
                parameters[f"layers.{i}"] = layer_params

        # Handle layer norms for different scales
        for norm_name in ["norm1", "norm2", "norm3", "mlp_norm1", "mlp_norm2", "mlp_norm3"]:
            if hasattr(torch_model, norm_name):
                norm_layer = getattr(torch_model, norm_name)
                parameters[norm_name] = {}
                parameters[norm_name]["weight"] = ttnn.from_torch(
                    norm_layer.weight, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT
                )
                parameters[norm_name]["bias"] = ttnn.from_torch(
                    norm_layer.bias, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT
                )

        # Handle MLPs - delegate to existing TTMlp preprocessor
        for mlp_name in ["fea_mlp1", "fea_mlp2", "fea_mlp3", "mlp1", "mlp2", "mlp3"]:
            if hasattr(torch_model, mlp_name):
                mlp = getattr(torch_model, mlp_name)
                mlp_params = preprocess_model_parameters(
                    initialize_model=lambda m=mlp: m,
                    custom_preprocessor=create_mlp_preprocessor(device),
                    device=device,
                )
                parameters[mlp_name] = mlp_params

        # Handle linear classification layers
        for linear_name in ["linear1", "linear2", "linear3"]:
            if hasattr(torch_model, linear_name):
                linear_layer = getattr(torch_model, linear_name)
                parameters[linear_name] = {}
                parameters[linear_name]["weight"] = preprocess_linear_weight(linear_layer.weight, dtype=ttnn.bfloat16)
                parameters[linear_name]["bias"] = preprocess_linear_bias(linear_layer.bias, dtype=ttnn.bfloat16)

        # Handle mask token inference modules
        for mask_name in ["mask_pre1", "mask_pre2", "mask_pre3"]:
            if hasattr(torch_model, mask_name):
                mask_module = getattr(torch_model, mask_name)
                mask_params = preprocess_model_parameters(
                    initialize_model=lambda m=mask_module: m,
                    custom_preprocessor=create_mask_token_inference_preprocessor(device),
                    device=device,
                )
                parameters[mask_name] = mask_params

        return parameters

    return custom_preprocessor


@pytest.mark.parametrize(
    "image_size, patch_size, token_size, num_cls",
    [
        (256, 2, 4, 1),  # Original configuration
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}])
def test_tile_selection(device, image_size, patch_size, token_size, num_cls):
    """Test TileSelection module against PyTorch reference for correctness"""

    # Create mock args object
    class Args:
        def __init__(self, imgsz, patchsz, token_size, dim):
            self.imgsz = imgsz
            self.patchsz = patchsz
            self.token_size = token_size
            self.dim = dim

    # Calculate dimensions
    num_layers = int(math.log2((image_size // patch_size) // token_size))
    dim = 96

    args = Args(image_size, patch_size, token_size, dim)

    # Create test input [B, C, H, W]
    batch_size = 3
    input_tensor = torch.randn(batch_size, 3, image_size, image_size)

    # Create PyTorch reference
    ref_layer = TileSelection(args, num_cls)
    ref_layer.eval()

    with torch.no_grad():
        ref_output = ref_layer(input_tensor)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: ref_layer,
        custom_preprocessor=create_tile_selection_preprocessor(device, dim),
        device=device,
    )

    # Create TTNN implementation
    tt_layer = TTTileSelection(device=device, parameters=parameters, args=args, num_cls=num_cls)

    # Convert input to TTNN
    tt_input = ttnn.from_torch(input_tensor, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)

    # Run TTNN implementation
    tt_output = tt_layer(tt_input)

    # Convert outputs back to torch for comparison
    tt_mask_3 = tt2torch_tensor(tt_output)

    # Compare outputs with appropriate PCC thresholds
    does_pass_3, pcc_message_3 = comp_pcc(ref_output[0], tt_mask_3, 0.98)

    logger.info(f"Scale 3 PCC: {pcc_message_3}")

    if does_pass_3:
        logger.info("TileSelection Passed!")
    else:
        logger.warning("TileSelection Failed!")

    assert does_pass_3, f"TileSelection test failed - Scale 3: {does_pass_3}"
