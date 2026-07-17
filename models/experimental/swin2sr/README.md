# Swin2SR

**Platforms:** Wormhole (n150 and n300)
**Supported Input Resolution:** Variable (tiled processing for arbitrary sizes, default tile size: 64x64). Supports input sizes like 256×256 and 512×512 for scale factors 2× and 4× respectively.

## Introduction
Swin2SR (Swin Transformer V2 for Super-Resolution) is a state-of-the-art image super-resolution model based on Swin Transformer V2 architecture. This implementation provides a pure TTNN version optimized for Tenstorrent hardware accelerators.

This repository provides:
- A **reference PyTorch model** for correctness validation.
- A **TT-NN implementation** for Tenstorrent hardware (Wormhole).
- **Tests**, **demo**, and **resources** (sample images and checkpoints).

## Table of Contents
- [Prerequisites](#prerequisites)
- [Repository Layout](#repository-layout)
- [Weights](#weights)
- [Quickstart](#quickstart)
  - [Run Tests](#run-tests)
  - [Run the Demo](#run-the-demo)
  - [Custom Images](#custom-images)
- [Performance](#performance)
- [Configuration Notes](#configuration-notes)

## Prerequisites
- Clone the **tt-metal** repository (source code & toolchains):
  <https://github.com/tenstorrent/tt-metal>
- Install **TT-Metalium™ / TT-NN™**:
  Follow the official instructions: <https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md>

## Repository Layout
```
models/experimental/swin2sr/
├── README.md                    # This file
├── demo/
│   └── demo_tiled.py            # Demo script with tiled processing
├── resources/
│   └── test_images/             # Test images
├── tt/                          # TTNN implementation
│   ├── tt_swin2sr.py            # Main model class
│   ├── tt_rstb.py               # Residual Swin Transformer Block
│   ├── tt_window_attention.py   # Window attention mechanism
│   ├── tt_swin_transformer_block.py
│   ├── tt_patch_embed.py
│   ├── tt_upsample.py           # PixelShuffle upsampling
│   ├── tt_mlp.py
│   ├── tt_basic_layer.py
│   └── utils.py
├── reference/                   # PyTorch reference implementation
└── tests/                       # PCC tests
    ├── pcc/
    |   └── test_ttnn_swin2sr.py # Full network tests
    └── perf/
        ├── test_perf.py         # Performance tests
        └── test_perf_e2e.py
```

## Weights
Swin2SR checkpoints are available in `resources/checkpoints/`:
- `Swin2SR_ClassicalSR_X2_64.pth` - 2x upscaling (66MB)
- `Swin2SR_ClassicalSR_X4_64.pth` - 4x upscaling (66MB)

The demo automatically selects the appropriate checkpoint based on the `--scale` argument.

## Quickstart
### Run Tests
```
pytest models/experimental/swin2sr/tests/pcc/test_ttnn_swin2sr.py
```
This runs an end-to-end flow that:
  - Loads the Swin2SR PyTorch reference model,
  - Runs the TT-NN graph,
  - Compares results (PCC validation).

**PCC (Pearson Correlation Coefficient) Values:**

| Test Configuration | PCC Value | Status |
|-------------------|-----------|--------|
| **Full Network (Random Weights)** | | |
| - `resi_connection='1conv'` | **0.998151** (99.82%) | ✓ PASS |
| - `resi_connection='3conv'` | **0.999162** (99.92%) | ✓ PASS |
| **Full Network (Trained Checkpoint)** | | |
| - `resi_connection='1conv'` | **0.999217** (99.92%) | ✓ PASS |

**PCC Threshold**: ≥ 0.99

### Multi-Device:
To run multi-device test:
```
pytest models/experimental/swin2sr/tests/perf/test_perf_e2e.py::test_swin2sr_perf_multi_device -s
```

### Run the Demo
```
python3 models/experimental/swin2sr/demo/demo_tiled.py --image <input image path> --output <output image path> --scale 2
```

### Custom Images
Sample images are placed under:
```
models/experimental/swin2sr/resources/test_images/
```

**Example for 2× upscaling:**
```
python3 models/experimental/swin2sr/demo/demo_tiled.py \
    --image models/experimental/swin2sr/resources/test_images/Set5/LR_bicubic/X2/babyx2.png \
    --scale 2 \
    --output output_2x.png
```

**Example for 4× upscaling:**
```
python3 models/experimental/swin2sr/demo/demo_tiled.py \
    --image models/experimental/swin2sr/resources/test_images/Set5/LR_bicubic/X4/your_image.png \
    --scale 4 \
    --output output_4x.png
```

The model supports various input sizes including 256×256 and 512×512 pixels. For larger images, the demo automatically uses tiled processing to handle memory constraints.

## Performance
### Single Device (BS=1, img_size=64x64)(n150):
- end-2-end perf with 2CQ+Trace is `~2` FPS
- Device Performance is `~1.7` FPS

### Multi Device (BS=1, img_size=64x64)(n300):
- end-2-end perf with 2CQ+Trace is `~4` FPS

To run perf test:
```
pytest models/experimental/swin2sr/tests/perf/test_perf_e2e.py::test_swin2sr_perf_single_device -s
pytest models/experimental/swin2sr/tests/perf/test_perf_e2e.py::test_swin2sr_perf_multi_device -s
```

This test validates Swin2SR on single and multi-device setups using 2 command queues with trace.


## Configuration Notes
- Resolution: Variable input sizes supported via tiled processing (default tile size: 64x64). Tile size must be a multiple of window_size (8). The model supports input sizes like 256×256 for 2× upscaling and 512×512 for 4× upscaling, with automatic tiling for larger images.
- Device: The demo opens a Wormhole device (default id typically 0). If you need to change it, use `--device-id` argument.
- Batch Size: Demo/tests are written for BS=1. For larger BS you'll need to verify memory layouts and tile alignment.
- Memory Layouts: The TT-NN path uses ROW_MAJOR layout for resize ops and may pad channels to multiples of 32 to satisfy kernel/tile alignment.
- Pipeline Configuration: Both Single device and Multi-device supports 2 command queues with trace for optimal performance using `ShardTensorToMesh` for input distribution and `ConcatMeshToTensor` for output composition.
- Tiled Processing: Large images are automatically split into overlapping tiles for processing. Uses weighted averaging at tile boundaries for smooth transitions. Required due to transformer attention's quadratic memory growth and Wormhole L1 memory constraints.

# References
- **Original Paper**: [Swin2-SR](https://arxiv.org/abs/2209.11345)
- **Reference Implementation**: [Swin2-SR](https://github.com/mv-lab/swin2sr/tree/main) by Marcos V. Conde, Ui-Jin Choi, Maxime Burchi, Radu Timofte ( Apache-2.0 license)
