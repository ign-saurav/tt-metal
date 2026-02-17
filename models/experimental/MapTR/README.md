# MapTR - Map TRansformer for TTNN

MapTR is a Bird's-Eye-View (BEV) map detection model implemented for Tenstorrent Neural Network (TTNN) framework. This implementation supports inference-only mode and is optimized for running on Tenstorrent hardware.

## Overview

MapTR is a transformer-based model for detecting map elements (lane dividers, pedestrian crossings, and boundaries) from multi-view camera images. This implementation uses:

- **Backbone**: ResNet50
- **Encoder**: BEVFormer encoder
- **Framework**: TTNN (Tenstorrent Neural Network) for hardware acceleration
- **Dataset**: NuScenes
- **Input Resolution**: 1600x900 pixels (multi-view camera images from 6 cameras)

## Features
- **Comprehensive testing**: PCC (Pearson Correlation Coefficient) tests for all components
- **Demo visualization**: Full demo script with visualization capabilities
- **Modular architecture**: Clean separation of PyTorch reference and TTNN implementations

## Project Structure

```
MapTR/
├── demo/                           # Demo scripts
│   ├── demo.py                     # Main inference demo
│   └── processing.py                # Data generation utilities
├── figs/                           # Figures and visualizations
│   ├── car.png                     # Car visualization
│   └── lidar_car.png               # LiDAR car visualization
├── reference/                      # Reference implementations and configs
│   ├── config_maptr_tiny_r50_24e_bevformer.py  # Main config file
│   ├── maptr.py                    # MapTR detector implementation
│   ├── dependency.py               # Shared dependencies and utilities from external libraries
│   ├── datasets.py                 # Dataset builders and samplers
│   ├── datasets_nuscenes.py        # NuScenes dataset implementation
│   ├── datasets_nuscenes_map.py    # NuScenes map dataset implementation
│   ├── pipelines.py                # Data processing pipelines
│   ├── utils.py                    # Utility functions
│   ├── bevformer_base_layer.py     # BEVFormer base transformer layer
│   ├── bevformer_decoder.py        # BEVFormer decoder
│   ├── bevformer_deformable_attn.py # Deformable attention implementation
│   ├── bevformer_encoder.py        # BEVFormer encoder
│   ├── bevformer_spatial_attention.py  # Spatial cross attention
│   └── bevformer_temporal_attention.py # Temporal self attention
├── resources/                      # Resource utilities
│   ├── download_chkpoint.py        # Checkpoint download utility
│   └── nuScenes/                   # Sample data files
│       └── samples/                # Sample camera images
├── tests/                          # Test suite
│   ├── pcc/                        # PCC tests for numerical validation
│   │   ├── test_backbone.py        # ResNet50 backbone test
│   │   ├── test_encoder.py         # BEVFormer encoder test
│   │   ├── test_deocder.py         # MapTR decoder test
│   │   ├── test_head.py            # MapTR head test
│   │   ├── test_fpn.py             # FPN test
│   │   ├── test_mha.py             # Multi-head attention test
│   │   ├── test_transformer.py     # Transformer test
│   │   ├── test_maptr.py           # Full model end-to-end test
│   │   ├── test_custom_deformable_attention.py  # Custom deformable attention test
│   │   ├── test_spatial_cross_attention.py      # Spatial cross attention test
│   │   └── test_temporal_self_attention.py      # Temporal self attention test
│   └── perf/                       # Performance tests
│       ├── test_e2e_ttcnn_performant.py  # TT-CNN pipeline performance test
│       └── test_perf.py            # Device performance test
└── tt/                             # TTNN implementations
    ├── ttnn_backbone.py            # ResNet50 TTNN implementation
    ├── ttnn_bottleneck.py          # ResNet bottleneck TTNN implementation
    ├── ttnn_encoder.py             # BEVFormer encoder TTNN implementation
    ├── ttnn_decoder.py             # MapTR decoder TTNN implementation
    ├── ttnn_head.py                # MapTR head TTNN implementation
    ├── ttnn_transformer.py         # Transformer TTNN implementation
    ├── ttnn_maptr.py               # Full MapTR TTNN model
    ├── ttnn_fpn.py                 # FPN TTNN implementation
    ├── ttnn_mha.py                 # Multi-head attention TTNN implementation
    ├── ttnn_ffn.py                 # Feed-forward network TTNN implementation
    ├── ttnn_spatial_cross_attention.py  # Spatial cross attention TTNN
    ├── ttnn_temporal_self_attention.py # Temporal self attention TTNN
    ├── ttnn_custom_defrmble_attention.py  # Custom deformable attention TTNN
    ├── ttnn_detr_transformer_decoder_layer.py  # DETR decoder layer TTNN
    ├── model_preprocessing.py       # Model preprocessing utilities
    └── utils.py                    # TTNN utility functions
```

## Performance

### Single Device (BS=1, img_size=384x640) (N150):
- Device FPS: `~0.11`

### Multi Device (BS=1, img_size=384x640) (N300):
- Device FPS: `~0.24`
- E2E FPS (Direct TTNN): `~0.04`
- E2E FPS (2CQ + no trace): `~0.20`

## PCC (Pearson Correlation Coefficient) Values

| Output Component | PCC Value | Status |
|-----------------|-----------|--------|
| **End-to-End Model Outputs (Trained Checkpoint)** | | |
| - `bev_embed` | **0.995472** (99.55%) | ✓ PASS |
| - `all_cls_scores` | **0.998778** (99.88%) | ✓ PASS |
| - `all_bbox_preds` | **0.999790** (99.98%) | ✓ PASS |
| - `all_pts_preds` | **0.999871** (99.99%) | ✓ PASS |


## Installation

### Prerequisites

- Clone the **tt-metal** repository (source code & toolchains):
  <https://github.com/tenstorrent/tt-metal>
- Install **TT-Metalium™ / TT-NN™**:
  Follow the official instructions: <https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md>

### Setup

1. Ensure you're in the TTNN environment.

2. Install required Python packages:
```bash
pip install pyquaternion nuscenes-devkit gdown
```

3. The checkpoint will be automatically downloaded when you run the demo or tests.

## Quick Start

### Running the Demo

The demo script automatically downloads the checkpoint if it's missing:

```bash
# Using default checkpoint (auto-downloads if missing)
python \
    models/experimental/MapTR/demo/demo.py \
    models/experimental/MapTR/reference/config_maptr_tiny_r50_24e_bevformer.py

# Using custom checkpoint
python \
    models/experimental/MapTR/demo/demo.py \
    models/experimental/MapTR/reference/config_maptr_tiny_r50_24e_bevformer.py \
    path/to/your/checkpoint.pth

# With custom options
python \
    models/experimental/MapTR/demo/demo.py \
    models/experimental/MapTR/reference/config_maptr_tiny_r50_24e_bevformer.py \
    --score-thresh 0.5 \
    --show-dir ./output \
    --device-params '{"l1_small_size": 32768}'
```

### Demo Options

- `config`: Path to the configuration file (required)
- `checkpoint`: Path to checkpoint file (optional, defaults to auto-downloaded weights)
- `--score-thresh`: Score threshold for predictions (default: 0.4)
- `--show-dir`: Directory to save visualizations (default: `./work_dirs/...`)
- `--show-cam`: Show camera images in visualization
- `--gt-format`: Ground truth visualization format (default: `fixed_num_pts`)
- `--device-params`: TTNN device parameters as JSON string

### Running Tests

All tests automatically download the checkpoint if needed:

```bash
# Run all PCC tests
pytest models/experimental/MapTR/tests/pcc/

# Run specific test
pytest models/experimental/MapTR/tests/pcc/test_tt_maptr.py

# Run with verbose output
pytest models/experimental/MapTR/tests/pcc/ -v

# Run performance tests
pytest models/experimental/MapTR/tests/perf/
```

### Test Files

- `test_backbone.py`: Tests ResNet50 backbone implementation
- `test_encoder.py`: Tests BEVFormer encoder implementation
- `test_deocder.py`: Tests MapTR decoder implementation
- `test_head.py`: Tests MapTR head implementation
- `test_fpn.py`: Tests Feature Pyramid Network implementation
- `test_mha.py`: Tests Multi-head Attention implementation
- `test_spatial_cross_attention.py`: Tests Spatial Cross Attention
- `test_temporal_self_attention.py`: Tests Temporal Self Attention
- `test_transformer.py`: Tests MapTRPerceptionTransformer
- `test_maptr.py`: End-to-end full model test

## Configuration

The main configuration file is located at:
```
models/experimental/MapTR/projects/configs/maptr/maptr_tiny_r50_24e_bevformer.py
```

Key configuration parameters:

- `bev_h_`, `bev_w_`: BEV feature map dimensions (default: 200x100)
- `point_cloud_range`: 3D point cloud range for detection
- `num_vec`: Number of map vectors to predict
- `num_pts_per_vec`: Number of points per vector
- `num_classes`: Number of map classes (divider, ped_crossing, boundary)
- `embed_dims`: Embedding dimensions

## Model Architecture

### Components

1. **Backbone (ResNet50)**: Extracts features from multi-view camera images
2. **FPN**: Feature Pyramid Network for multi-scale feature extraction
3. **BEVFormer Encoder**: Transforms image features to BEV representation
4. **MapTR Decoder**: Decodes BEV features to map elements
5. **MapTR Head**: Final prediction head for map element detection

### Input/Output

- **Input**: Multi-view camera images (6 cameras: front, front-left, front-right, back, back-left, back-right)
- **Output**: Map elements (lane dividers, pedestrian crossings, boundaries) in BEV space

## Checkpoint Management

The checkpoint is automatically managed:

- **Location**: `models/experimental/MapTR/chkpt/downloaded_weights.pth`
- **Auto-download**: Checkpoints are automatically downloaded when running demo or tests

The checkpoint download utility:
- Checks if checkpoint exists before downloading
- Installs `gdown` automatically if needed
- Downloads from Google Drive
- Creates necessary directories

## Troubleshooting

### Checkpoint Download Issues

If checkpoint download fails:

1. Check internet connection
2. Verify `gdown` is installed (should be in requirements-dev.txt)
3. Check that the checkpoint directory exists and is writable

### Test Failures

If PCC tests fail:

1. Check that checkpoint is downloaded correctly
2. Verify TTNN device is properly initialized
3. Check device parameters match hardware capabilities
4. Review test logs for specific component failures

### Demo Issues

If demo fails:

1. Verify configuration file path is correct (should be `models/experimental/MapTR/reference/config_maptr_tiny_r50_24e_bevformer.py`)
2. Check that dataset paths in config are valid
3. Ensure TTNN device parameters are appropriate for your hardware
4. Check image input format matches expected dimensions
5. Verify you're using `python_env/bin/python` or have the correct Python environment activated

## License

SPDX-License-Identifier: Apache-2.0

Copyright © 2026 Tenstorrent AI ULC

## Source Code Implementation and licenses
- **MapTR**: https://github.com/hustvl/MapTR (MIT License)
-  - MapTR: [Original MapTR Paper](https://arxiv.org/abs/2208.14437)
-  - BEVFormer: [BEVFormer Paper](https://arxiv.org/abs/2203.17270)
- **MMCV**: https://github.com/open-mmlab/mmcv/tree/v1.4.0/mmcv (Apache License 2.0)
- **MMSegmentation**: https://github.com/open-mmlab/mmsegmentation/tree/v0.14.1/mmseg (Apache License 2.0)
- **MMDetection3D**: https://github.com/open-mmlab/mmdetection3d/tree/v0.17.1/mmdet3d (Apache License 2.0)
- **MMDetection**: https://github.com/open-mmlab/mmdetection/tree/v2.14.0/mmdet (Apache License 2.0)
- **MMEngine**: https://github.com/open-mmlab/mmengine/blob/main/mmengine (Apache License 2.0)


Original work Copyright (c) OpenMMLab. Licensed under the Apache License, Version 2.0.
