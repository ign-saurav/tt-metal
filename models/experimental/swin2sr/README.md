# Swin2SR TTNN Implementation

Swin2SR (Swin Transformer V2 for Super-Resolution) implementation for Tenstorrent Neural Network (TTNN) accelerator.

## Model Overview

Swin2SR is a state-of-the-art image super-resolution model based on Swin Transformer V2 architecture. This implementation provides a pure TTNN version optimized for Tenstorrent hardware accelerators.

### Architecture Details

**Model Configuration:**
- **Embedding Dimension**: 180
- **Number of Layers**: 6
- **Depths**: (6, 6, 6, 6, 6, 6) → 36 transformer blocks total
- **Number of Heads**: (6, 6, 6, 6, 6, 6)
- **Window Size**: 8
- **MLP Ratio**: 2.0
- **Patch Size**: 1
- **Input Channels**: 3 (RGB)
- **Upsampler**: PixelShuffle
- **Residual Connection**: 1conv or 3conv

**Model Components:**
1. **Shallow Feature Extraction**: Initial convolutional layer (`conv_first`)
2. **Deep Feature Extraction**:
   - Patch embedding
   - 6 Swin Transformer layers (each with 6 transformer blocks)
   - Layer normalization
3. **Reconstruction Module**:
   - Residual connection (`conv_after_body`)
   - Upsampling via PixelShuffle (`upsample`)
   - Final convolutional layer (`conv_last`)

**Total Parameters**: ~66MB (checkpoint size)

## Performance Metrics

### PCC (Pearson Correlation Coefficient) Values

The TTNN implementation achieves high accuracy compared to the PyTorch reference:

| Test Configuration | PCC Value | Status |
|-------------------|-----------|--------|
| **Full Network (Random Weights)** | | |
| - `resi_connection='1conv'` | **0.997591** (99.76%) | ✓ PASS |
| - `resi_connection='3conv'` | **0.999281** (99.93%) | ✓ PASS |
| **Full Network (Trained Checkpoint)** | | |
| - `resi_connection='1conv'` | **0.999222** (99.92%) | ✓ PASS |

**PCC Threshold**: ≥ 0.99

**Note**: Individual components achieve >0.99 PCC; accumulated precision loss is expected in deep bfloat16 models due to numerical precision differences.

## Features

- ✅ **Pure TTNN Implementation**: No PyTorch fallbacks, all operations use TTNN primitives
- ✅ **Tiled Processing**: Supports processing large images by splitting into tiles
- ✅ **DRAM Support**: Can handle larger direct inputs using DRAM slicing
- ✅ **Multiple Scale Factors**: Supports 2x, 3x, 4x, and 8x upscaling
- ✅ **Checkpoint Compatibility**: Works with pre-trained Swin2SR checkpoints

## Directory Structure

```
models/experimental/swin2sr/
├── README.md                    # This file
├── demo/
│   └── demo_tiled.py           # Demo script with tiled processing
├── resources/
│   ├── checkpoints/            # Model checkpoints (.pth files)
│   └── test_images/            # Test images
├── tt/                         # TTNN implementation
│   ├── tt_swin2sr.py          # Main model class
│   ├── tt_rstb.py             # Residual Swin Transformer Block
│   ├── tt_window_attention.py # Window attention mechanism
│   ├── tt_swin_transformer_block.py
│   ├── tt_patch_embed.py
│   ├── tt_upsample.py         # PixelShuffle upsampling
│   ├── tt_mlp.py
│   ├── tt_basic_layer.py
│   └── utils.py
├── reference/                  # PyTorch reference implementation
└── tests/                      # PCC tests
    └── pcc/
        └── test_ttnn_swin2sr.py  # Full network tests
```

## Prerequisites

- Tenstorrent device with TTNN drivers installed
- Python environment with TTNN and dependencies
- Model checkpoints (available in `resources/checkpoints/`)

## Running the Demo

### Basic Usage

```bash
# Run with default settings (2x upscale, tiled mode)
python models/experimental/swin2sr/demo/demo_tiled.py \
    --image models/experimental/swin2sr/resources/test_images/Set5/LR_bicubic/X2/babyx2.png \
    --scale 2

# Run with 4x upscale
python models/experimental/swin2sr/demo/demo_tiled.py \
    --image models/experimental/swin2sr/resources/test_images/Set5/LR_bicubic/X2/babyx2.png \
    --scale 4

# Specify custom output path
python models/experimental/swin2sr/demo/demo_tiled.py \
    --image <input_image_path> \
    --scale 2 \
    --output <output_image_path>
```

### Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--image` | str | **Required** | Path to input image |
| `--checkpoint` | str | Auto-selected | Path to model checkpoint |
| `--output` | str | Auto-generated | Path to save output image |
| `--scale` | int | 2 | Upscale factor (2, 3, 4, or 8) |
| `--tile-size` | int | 64 | Tile size (must be multiple of 8) |
| `--tile-overlap` | int | 32 | Overlap between tiles |
| `--device-id` | int | 0 | TTNN device ID |

### Processing Mode

The demo uses **tiled processing** to handle images of any size:
- Splits large images into overlapping tiles for processing
- Handles arbitrary image sizes
- Each tile is processed independently and results are stitched together
- Uses weighted averaging at tile boundaries for smooth transitions

### Examples

**Example 1: Process a test image with 2x upscale**
```bash
python models/experimental/swin2sr/demo/demo_tiled.py \
    --image models/experimental/swin2sr/resources/test_images/Set5/LR_bicubic/X2/babyx2.png \
    --scale 2 \
    --output output_2x.png
```

**Example 2: Process with 4x upscale and custom tile size**
```bash
python models/experimental/swin2sr/demo/demo_tiled.py \
    --image models/experimental/swin2sr/resources/test_images/Set5/LR_bicubic/X2/babyx2.png \
    --scale 4 \
    --tile-size 64 \
    --tile-overlap 32 \
    --output output_4x.png
```

**Example 3: Custom tile size and overlap**
```bash
python models/experimental/swin2sr/demo/demo_tiled.py \
    --image models/experimental/swin2sr/resources/test_images/Set5/LR_bicubic/X2/babyx2.png \
    --scale 2 \
    --tile-size 128 \
    --tile-overlap 64 \
    --output output_custom_tiles.png
```

## Available Checkpoints

Checkpoints are located in `resources/checkpoints/`:

- `Swin2SR_ClassicalSR_X2_64.pth` - 2x upscaling (66MB)
- `Swin2SR_ClassicalSR_X4_64.pth` - 4x upscaling (66MB)

The demo automatically selects the appropriate checkpoint based on the `--scale` argument.

## Testing

Run PCC tests to verify model accuracy:

```bash
# Run full network tests
pytest models/experimental/swin2sr/tests/pcc/test_ttnn_swin2sr.py -v

# Run specific test
pytest models/experimental/swin2sr/tests/pcc/test_ttnn_swin2sr.py::test_swin2sr_ttnn_vs_torch -v
pytest models/experimental/swin2sr/tests/pcc/test_ttnn_swin2sr.py::test_swin2sr_checkpoint -v
```

## Implementation Details

### Pixel Shuffle Replacement

The TTNN implementation replaces PyTorch's `nn.PixelShuffle` with pure TTNN operations:
- Uses `ttnn.reshape` and `ttnn.permute` to rearrange channels into spatial dimensions
- Maintains mathematical equivalence with PyTorch version

### Memory Management

- **L1 Memory**: Used for smaller operations and intermediate tensors
- **DRAM Memory**: Used for larger tensors and when `use_dram=True`
- **Sharding Strategies**: Automatically selected based on tensor sizes

### Padding

- Uses constant padding (zero-padding) via `ttnn.pad` instead of PyTorch's reflection padding
- Padding ensures input dimensions are multiples of `window_size`

## Limitations

1. **Fixed Input Dimensions**: TTNN model requires fixed dimensions matching initialization `img_size`
2. **Tiled Processing**: Large images are automatically split into tiles for processing
3. **Tile Size**: Must be a multiple of `window_size` (8)
4. **Memory**: Very large images may require smaller tile sizes or more overlap

## References

- Original Swin2SR Paper: [Swin2SR: SwinV2 Transformer for Compressed Image Super-Resolution and Restoration](https://arxiv.org/abs/2209.11345)
- Swin Transformer V2: [Swin Transformer V2: Scaling Up Capacity and Resolution](https://arxiv.org/abs/2111.09883)

## License

SPDX-License-Identifier: Apache-2.0

Copyright © 2025 Tenstorrent AI ULC
