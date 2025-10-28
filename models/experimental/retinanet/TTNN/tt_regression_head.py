# models/experimental/retinanet/tt/regressionhead.py
import ttnn
from typing import List
from models.experimental.retinanet.TTNN.utils import Conv2dNormActivation


def ttnn_retinanet_regression_head(
    feature_maps: List[ttnn.Tensor],
    parameters: dict,
    device: ttnn.Device,
    in_channels: int = 256,
    num_anchors: int = 9,
    batch_size: int = 1,
    input_shapes: List[tuple] = None,
) -> ttnn.Tensor:
    """
    TTNN implementation of RetinaNet regression head with all 4 conv layers + GroupNorm + ReLU.

    Args:
        feature_maps: List of FPN feature tensors in NHWC format
        parameters: Dict containing 'conv' (list of 4 Conv2dNormActivation params) and 'bbox_reg'
        device: TTNN device
        in_channels: Number of input channels (256 for RetinaNet)
        num_anchors: Number of anchors per location (9 for RetinaNet)
        batch_size: Batch size
        input_shapes: List of (H, W) tuples for each FPN level

    Returns:
        Concatenated bbox regressions in shape (N, total_anchors, 4)
    """
    all_bbox_regression = []

    grid_size = ttnn.CoreGrid(y=8, x=8)

    input_mask_tensor = ttnn.create_group_norm_input_mask(in_channels, 32, grid_size.y)
    input_mask_tensor = ttnn.from_torch(
        input_mask_tensor,
        dtype=ttnn.DataType.BFLOAT8_B,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Initialize 4 Conv2dNormActivation blocks
    conv_blocks = []
    for conv_idx in range(4):
        conv_block = Conv2dNormActivation(
            parameters=parameters["conv"][conv_idx],
            device=device,
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            num_groups=32,
            grid_size=grid_size,
            input_mask=input_mask_tensor,
        )
        conv_blocks.append(conv_block)

    for level_idx, x in enumerate(feature_maps):
        H, W = input_shapes[level_idx]

        # Apply 4 conv blocks (Conv2d + GroupNorm + ReLU)
        for conv_block in conv_blocks:
            x = conv_block(x, batch_size=batch_size, input_height=H, input_width=W)

        # Final bbox_reg conv layer
        bbox_reg_slice_config = ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dDRAMSliceHeight, num_slices=4)

        bbox_regression = ttnn.conv2d(
            input_tensor=x,
            weight_tensor=parameters["bbox_reg"]["weight"],
            in_channels=in_channels,
            out_channels=num_anchors * 4,
            device=device,
            bias_tensor=parameters["bbox_reg"]["bias"],
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            batch_size=batch_size,
            input_height=H,
            input_width=W,
            slice_config=bbox_reg_slice_config,
        )

        # Reshape to (N, H*W*num_anchors, 4)
        N, H_final, W_final, C_final = bbox_regression.shape
        bbox_regression = ttnn.reshape(bbox_regression, (N, H_final, W_final, num_anchors, 4))
        bbox_regression = ttnn.reshape(bbox_regression, (N, H_final * W_final * num_anchors, 4))

        all_bbox_regression.append(bbox_regression)

    # Concatenate all FPN levels
    output = ttnn.concat(all_bbox_regression, dim=1)
    return output
