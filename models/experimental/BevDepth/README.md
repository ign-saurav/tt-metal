# BEVDepth

- **Platforms:** Wormhole (n150)
- **Supported Input Resolution:** `(256, 704)` = (Height, Width)
- **Supported Model Configuration:** [bev_depth_lss_r50_256x704_128x128_24e_2key.py](https://github.com/Megvii-BaseDetection/BEVDepth/blob/main/bevdepth/exps/nuscenes/mv/bev_depth_lss_r50_256x704_128x128_24e_2key.py)

## Introduction

BEVDepth is a multi-view 3D object detection model that acquires reliable depth information for accurate Bird's Eye View (BEV) perception. The model uses a **Lift-Splat-Shoot (LSS)** architecture with a **ResNet-50** backbone and **SECONDFPN** neck to process multi-camera inputs and generate 3D object detections in BEV space.

This implementation adapts **BEVDepth** for Tenstorrent hardware using the TT-NN and TT-Metalium stack, optimized for throughput and low-latency inference on Wormhole devices. The implementation supports the [bev_depth_lss_r50_256x704_128x128_24e_2key](https://github.com/Megvii-BaseDetection/BEVDepth/blob/main/bevdepth/exps/nuscenes/mv/bev_depth_lss_r50_256x704_128x128_24e_2key.py) configuration with 6-camera inputs.

This repository provides:
- A **reference PyTorch model** (from [Megvii-BaseDetection/BEVDepth](https://github.com/Megvii-BaseDetection/BEVDepth)) for correctness validation.
- A **TT-NN implementation** for Tenstorrent hardware (Wormhole).
- **tests**, **demo**, and **resources** (sample nuScenes data).

## Table of Contents
- [Prerequisites](#prerequisites)
- [Repository Layout](#repository-layout)
- [Weights](#weights)
- [Quickstart](#quickstart)
  - [Run Tests](#run-tests)
  - [Run the Demo](#run-the-demo)
- [Model Architecture](#model-architecture)
- [Configuration Notes](#configuration-notes)
- [References](#references)

## Prerequisites
- Clone the **tt-metal** repository (source code & toolchains):
  <https://github.com/tenstorrent/tt-metal>
- Install **TT-Metalium™ / TT-NN™**:
  Follow the official instructions: <https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md>
- Install additional dependencies:
  ```bash
  pip3 install -r models/experimental/BevDepth/reference/requirements.txt
  pip3 install pytorch-lightning
  pip3 install pyquaternion
  pip3 install nuscenes-devkit
  ```

## Repository Layout
```
models/
└── experimental/
    └── BevDepth/
        ├── resources/
        │   ├── checkpoints/
        │   │   └── bevdepth_weights.pth      # Model checkpoint (auto-downloaded)
        │   └── nuScenes/
        │       └── samples/                   # Sample camera images (6 cameras)
        │           ├── CAM_BACK/
        │           ├── CAM_BACK_LEFT/
        │           ├── CAM_BACK_RIGHT/
        │           ├── CAM_FRONT/
        │           ├── CAM_FRONT_LEFT/
        │           ├── CAM_FRONT_RIGHT/
        │           └── LIDAR_TOP/
        │
        ├── reference/
        │   ├── base_bev_depth.py              # Base BEVDepth model
        │   ├── base_exp.py                    # Experiment base class
        │   ├── base_lss_fpn.py                # LSS FPN base
        │   ├── base_points.py                 # Point cloud utilities
        │   ├── bbox_3d.py                     # 3D bounding box utilities
        │   ├── bev_depth_head.py              # Detection head
        │   ├── bev_depth_lss_r50_256x704_128x128_24e_2key.py  # Model config
        │   ├── builder.py                     # Model builder
        │   ├── centerpoint_head.py            # CenterPoint head
        │   ├── conv.py                        # Convolution utilities
        │   ├── deform_conv.py                 # Deformable convolution
        │   ├── det3d_data_sample.py           # 3D detection data sample
        │   ├── gaussian.py                    # Gaussian utilities
        │   ├── norm.py                        # Normalization layers
        │   ├── point_data.py                  # Point data utilities
        │   ├── registry.py                    # Model registry
        │   ├── res_layer.py                   # ResNet layers
        │   ├── resnet.py                      # ResNet backbone
        │   ├── second_fpn.py                  # SECONDFPN neck
        │   └── requirements.txt               # Reference model dependencies
        │
        ├── tt/
        │   ├── custom_preprocessing.py        # Model preprocessing utilities
        │   ├── ttnn_bevdepth.py               # Main TTNN model wrapper (TtBEVDepth)
        │   ├── ttnn_bevdepth_backbone.py      # Backbone (ResNet + LSS FPN)
        │   ├── ttnn_bevdepth_head.py          # Detection head
        │   ├── ttnn_depthnet.py               # Depth estimation network
        │   ├── ttnn_resnet50_backbone.py      # ResNet-50 backbone
        │   ├── ttnn_secondfpn.py              # SECONDFPN neck
        │   └── utils.py                       # Utility functions (config creation, etc.)
        │
        ├── demo/
        │   ├── demo.py                        # Demo script with visualization
        │   └── processing.py                  # Post-processing utilities (decoding, NMS)
        │
        ├── tests/
        │   ├── pcc/                           # Pearson Correlation Coefficient tests
        │   │   ├── test_bevdepth_e2e.py       # End-to-end test
        │   │   ├── test_bevdepth_backbone.py  # Backbone test
        │   │   ├── test_bevdepth_head.py      # Head test
        │   │   ├── test_depthnet.py           # DepthNet test
        │   │   ├── test_resnet50_backbone.py  # ResNet-50 test
        │   │   └── test_secondfpn.py          # SECONDFPN test
        │   └── perf/                          # Performance tests
        │       └── test_bevdepth_perf.py      # Device performance test
        │
        ├── common.py                          # Common utilities (inference, weight loading)
        └── README.md                          # This file
```

## Weights

BEVDepth pretrained weights are automatically downloaded when running the model. The weights are from the official BEVDepth repository:

- **Model:** `bev_depth_lss_r50_256x704_128x128_24e_2key`
- **Download URL:** <https://github.com/Megvii-BaseDetection/BEVDepth/releases/download/v0.0.2/bev_depth_lss_r50_256x704_128x128_24e_2key.pth>
- **Checkpoint Location:** `/tmp/bevdepth_weights.pth` (auto-downloaded) or `resources/checkpoints/bev_depth_lss_r50_256x704_128x128_24e_2key.pth`

The weights are trained on the nuScenes dataset.

## Quickstart

### Run Tests

#### End-to-End Test
```bash
pytest models/experimental/BevDepth/tests/pcc/test_bevdepth_e2e.py
```
This runs a full end-to-end flow that:
- Loads the BEVDepth reference model from PyTorch
- Runs the TT-NN implementation
- Compares results using PCC (Pearson Correlation Coefficient) validation
- Validates all 6 task heads (heatmap, reg, height, dim, rot, vel)

#### Component Tests
```bash
# Test ResNet-50 backbone
pytest models/experimental/BevDepth/tests/pcc/test_resnet50_backbone.py

# Test SECONDFPN neck
pytest models/experimental/BevDepth/tests/pcc/test_secondfpn.py

# Test DepthNet
pytest models/experimental/BevDepth/tests/pcc/test_depthnet.py

# Test BEVDepth head
pytest models/experimental/BevDepth/tests/pcc/test_bevdepth_head.py

# Test full backbone (ResNet + LSS)
pytest models/experimental/BevDepth/tests/pcc/test_bevdepth_backbone.py
```

### Run the Demo

The demo script processes sample nuScenes data and visualizes 3D object detections:

```bash
python3 models/experimental/BevDepth/demo/demo.py --mode ttnn --output bevdepth_output.png
```

**Options:**
- `--mode`: Choose `torch`, `ttnn`, `both`, or `precomputed` (default: `both`)
- `--output`: Output visualization path (default: `bevdepth_demo_output.png`)
- `--threshold`: Detection score threshold (default: 0.3)
- `--show-range`: Visualization range in meters (default: 60.0)

The demo will:
1. Load sample images from `resources/nuScenes/samples/`
2. Run inference (PyTorch and/or TTNN)
3. Visualize 3D bounding boxes in BEV space
4. Save the visualization to the output path

## Performance
### Single Device (BS=1)(n150):
- Device perf is `3.8` FPS

To run perf test:
```
pytest models/experimental/BevDepth/tests/perf/test_bevdepth_perf.py -s
```

## Configuration Notes

### Supported Configuration
- **Model Config**: [bev_depth_lss_r50_256x704_128x128_24e_2key.py](https://github.com/Megvii-BaseDetection/BEVDepth/blob/main/bevdepth/exps/nuscenes/mv/bev_depth_lss_r50_256x704_128x128_24e_2key.py)
- **Input Resolution**: `(256, 704)` (Height, Width)
- **Number of Cameras**: 6 (CAM_FRONT_LEFT, CAM_FRONT, CAM_FRONT_RIGHT, CAM_BACK_RIGHT, CAM_BACK, CAM_BACK_LEFT)

### Device Configuration
- **Device**: The demo/tests open a Wormhole device (default id typically 0)
- **Batch Size**: Tests are written for BS=1. For larger batch sizes, verify memory layouts and tile alignment

## References
### Paper
- **BEVDepth: Acquisition of Reliable Depth for Multi-view 3D Object Detection**
  - Authors: Yinhao Li, Zheng Ge, Guanyi Yu, et al.
  - arXiv: <https://arxiv.org/pdf/2206.10092>
  - Year: 2022

### Code Repository
- **Official BEVDepth Implementation**: <https://github.com/Megvii-BaseDetection/BEVDepth>
- **License**: MIT License
