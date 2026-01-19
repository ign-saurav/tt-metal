# BEVDepth

**Platforms:** Wormhole (n150)
**Supported Input Resolution:** `(256, 704)` = (Height, Width)

## Introduction
BEVDepth is a multi-view 3D object detection model that acquires reliable depth information for accurate Bird's Eye View (BEV) perception. The model uses a **Lift-Splat-Shoot (LSS)** architecture with a **ResNet-50** backbone and **SECONDFPN** neck to process multi-camera inputs and generate 3D object detections in BEV space.

This implementation adapts **BEVDepth** for Tenstorrent hardware using the TT-NN and TT-Metalium stack, optimized for throughput and low-latency inference on Wormhole devices. The implementation supports the [bev_depth_lss_r50_256x704_128x128_24e_2key](https://github.com/Megvii-BaseDetection/BEVDepth/blob/main/bevdepth/exps/nuscenes/mv/bev_depth_lss_r50_256x704_128x128_24e_2key.py) configuration with 6-camera inputs.

This repository provides:
- A **reference PyTorch model** (from [Megvii-BaseDetection/BEVDepth](https://github.com/Megvii-BaseDetection/BEVDepth)) for correctness validation.
- A **TT-NN implementation** for Tenstorrent hardware (Wormhole).
- **Tests**, **demo**, and **resources** (sample nuScenes data).

## Table of Contents
- [Prerequisites](#prerequisites)
- [Repository Layout](#repository-layout)
- [Weights](#weights)
- [Quickstart](#quickstart)
  - [Run Tests](#run-tests)
  - [Run the Demo](#run-the-demo)
- [Performance](#performance)
- [Configuration Notes](#configuration-notes)
- [References](#references)

## Prerequisites
- Clone the **tt-metal** repository (source code & toolchains):
  <https://github.com/tenstorrent/tt-metal>
- Install **TT-Metalium™ / TT-NN™**:
  Follow the official instructions: <https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md>
- Install additional dependencies for BevDepth as mentioned in "tt_metal/python_env/requirements-dev.txt" if not already present.

## Repository Layout
```
models/
└── experimental/
    └── BevDepth/
        ├── resources/
        │   └── bevdepth_weights.pth           # Model checkpoint (auto-downloaded)
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
        │   └── utils.py                       # Utility functions
        │
        ├── demo/
        │   ├── demo.py                        # Demo script with visualization
        │   └── processing.py                  # Post-processing utilities
        │
        ├── tests/
        │   ├── pcc/                           # Pearson Correlation Coefficient tests
        │   │   ├── test_bevdepth.py           # End-to-end functional test
        │   │   ├── test_bevdepth_backbone.py  # Backbone test
        │   │   ├── test_bevdepth_head.py      # Head test
        │   │   ├── test_depthnet.py           # DepthNet test
        │   │   ├── test_resnet50_backbone.py  # ResNet-50 test
        │   │   └── test_secondfpn.py          # SECONDFPN test
        │   └── perf/                          # Performance tests
        │       ├── test_bevdepth_perf.py      # Device performance test
        │       └── test_e2e_performant.py     # End-to-end performance test with pipeline
        │
        ├── common.py                          # Common utilities
        └── README.md
```

## Weights
BEVDepth pretrained weights are automatically downloaded when running the model. The weights are from the official BEVDepth repository:

- **Model:** `bev_depth_lss_r50_256x704_128x128_24e_2key`
- **Download URL:** <https://github.com/Megvii-BaseDetection/BEVDepth/releases/download/v0.0.2/bev_depth_lss_r50_256x704_128x128_24e_2key.pth>
- **Checkpoint Location:** `resources/bevdepth_weights.pth` (auto-downloaded)

Note: The weights are trained on the nuScenes dataset.

## Quickstart
### Run Tests
```
pytest models/experimental/BevDepth/tests/pcc/test_bevdepth.py
```
This runs an end-to-end flow that:
  - Loads the BEVDepth reference model from PyTorch,
  - Runs the TT-NN implementation,
  - Compares results (PCC validation),
  - Validates all 6 task heads (heatmap, reg, height, dim, rot, vel).

**Note:**
- **DeformConv2d**: Torchvision's DeformConv2d is currently used as TTNN support is not yet available. A ticket has been raised for TTNN DeformConv2d support (https://github.com/tenstorrent/tt-metal/issues/34509).
- **SecondFPN Backbone Conv2d**: Torch fallback is used for Conv2d operations in the SecondFPN backbone. With TTNN implementation, the PCC drops to 0.82, while using PyTorch fallback achieves PCC >0.99, ensuring better accuracy.
- **Running TTNN**: To run the TTNN implementation (instead of torch fallback), set the environment variable `export FALLBACK_ON_SECONDFPN=0` before running tests or demo.

### Component Tests
```
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
```
python3 models/experimental/BevDepth/demo/demo.py --mode ttnn --output bevdepth_demo_output.png
```

**Options:**
- `--mode`: Choose `ttnn`, `both` (default: `ttnn`)
- `--output`: Output visualization path (default: `bevdepth_demo_output.png`)
- `--threshold`: Detection score threshold (default: 0.3)
- `--show-range`: Visualization range in meters (default: 60.0)

The demo uses the pipeline API (1CQ, no trace) and processes sample nuScenes data to visualize 3D object detections in BEV space.

## Performance
### Single Device (BS=1)(n150):
- Device perf is `3.8` FPS
- E2E perf (with 1CQ, no trace) is `0.17` FPS

### Run Device Performance Test
```
pytest models/experimental/BevDepth/tests/perf/test_bevdepth_perf.py -s
```

### Run End-to-End Performance Test
```
pytest models/experimental/BevDepth/tests/perf/test_e2e_performant.py -s
```

The e2e_performant test uses the pipeline API with 1 command queue and trace disabled, providing realistic end-to-end performance measurements.

**Note:** The test is configured with 1CQ (single command queue) without trace due to:
- 2CQ requires sharded inputs, which conflicts with BevDepth's L1 memory requirements
- Trace is not supported due to deformable convolution operations

## Configuration Notes
- Resolution: (H, W) = (256, 704) is supported end-to-end.
- Device: The demo/tests open a Wormhole device (default id typically 0). If you need to change it, adjust the device open call in the demo.
- Batch Size: Tests are written for BS=1. For larger BS you'll need to verify memory layouts and tile alignment.
- Number of Cameras: 6 cameras (CAM_FRONT_LEFT, CAM_FRONT, CAM_FRONT_RIGHT, CAM_BACK_RIGHT, CAM_BACK, CAM_BACK_LEFT).
- Model Config: [bev_depth_lss_r50_256x704_128x128_24e_2key.py](https://github.com/Megvii-BaseDetection/BEVDepth/blob/main/bevdepth/exps/nuscenes/mv/bev_depth_lss_r50_256x704_128x128_24e_2key.py)
- Weights: Auto-downloaded from the official BEVDepth repository if not cached locally.

## References
### Paper
- **BEVDepth: Acquisition of Reliable Depth for Multi-view 3D Object Detection**
  - Authors: Yinhao Li, Zheng Ge, Guanyi Yu, et al.
  - arXiv: <https://arxiv.org/pdf/2206.10092>
  - Year: 2022

### Code Repository and licensing
- **Official BEVDepth Implementation**: <https://github.com/Megvii-BaseDetection/BEVDepth>
- **License**: MIT License

This TT-Metal implementation is adapted from the original BEVDepth repository (MIT License) and is licensed under Apache-2.0 by Tenstorrent AI ULC. The reference implementations in the `reference/` directory retain their original MIT license headers with proper attribution to the source repositories.
