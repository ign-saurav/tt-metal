# PointPillars

**Platforms:** Wormhole (n150 and n300)

**Supported Input:** LiDAR Point Cloud (`.bin` format), processed to `(496, 432)` feature map

## Introduction
PointPillars is a real-time 3D object detection model designed for autonomous driving applications. It efficiently converts sparse 3D point clouds into a dense 2D pseudo-image representation using vertical pillars, enabling fast inference with standard 2D convolutional networks while maintaining high detection accuracy.

This implementation adapts **PointPillars** for Tenstorrent hardware using the TT-NN and TT-Metalium stack, optimized for throughput and low-latency inference on Wormhole devices.

This repository provides:
- A **reference PyTorch model** for correctness.
- A **TT-NN implementation** for Tenstorrent hardware (Wormhole).
- **Tests**, **demo**, and **resources** (sample point clouds).

## Table of Contents
- [Prerequisites](#prerequisites)
- [Repository Layout](#repository-layout)
- [Weights](#weights)
- [Quickstart](#quickstart)
  - [Run Tests](#run-tests)
  - [Run the Demo](#run-the-demo)
  - [Custom Point Clouds](#custom-point-clouds)
- [Performance](#performance)
- [Configuration Notes](#configuration-notes)

## Prerequisites
- Clone the **tt-metal** repository (source code & toolchains):
  <https://github.com/tenstorrent/tt-metal>
- Install **TT-Metalium™ / TT-NN™**:
  Follow the official instructions: <https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md>

## Repository Layout
```
models/
└── experimental/
    └── pointpillars/
        │
        ├── reference/
        │   ├── pointpillars.py                   # Main PointPillars model
        │   └── anchors.py                        # Anchor generation
        │   ├── voxel_module.py                   # Voxelization operations
        │   └── iou3d_module.py                   # 3D IoU operations
        │   ├── io.py                             # I/O utilities
        │   ├── process.py                        # Post-processing utilities
        │   └── vis_o3d.py                        # 3D visualization
        │
        ├── tt/
        │   ├── pointpillars.py                   # Main TTNN PointPillars model
        │   ├── pillar_encoder.py                 # TTNN PillarEncoder
        │   ├── backbone.py                       # TTNN Backbone
        │   ├── neck.py                           # TTNN Neck
        │   ├── head.py                           # TTNN Head
        │   ├── custom_preprocessor.py            # Custom weight preprocessing
        │   └── utils.py                          # TTNN utilities
        │
        ├── demo/
        │   └── demo.py                           # Main demo script
        │
        ├── runner/
        │   └── performant_runner_infra.py
        │
        ├── tests/
        │   ├── pcc/
        │   │   ├── test_pointpillars.py          # End-to-end pytest
        │   │   ├── test_backbone.py              # Backbone component test
        │   │   ├── test_neck.py                  # Neck component test
        │   │   ├── test_head.py                  # Head component test
        │   │   ├── test_pillar_encoder.py        # Pillar encoder test
        │   │   └── test_conv_transpose_split.py  # Conv transpose test
        │   ├── perf/
        │   │   └── test_pointpillars_perf_e2e.py # End-to-end performance
        │   │   └── test_pointpillars_device_perf.py # Device performance
        │   └── test_stability.py                 # Stability test
        └── common.py
        └── README.md
```

## Weights
PointPillars weights will be downloaded automatically:
- Downloaded from: [PointPillars Pretrained Weights](https://github.com/zhulf0804/PointPillars/blob/main/pretrained/epoch_160.pth)
- Saved as: `epoch_160.pth` in the working directory or `models/experimental/pointpillars/resources/checkpoint`

Note: The weights are pretrained on KITTI dataset for 3 classes (Car, Pedestrian, Cyclist).

## Quickstart
### Run Tests
```
pytest models/experimental/pointpillars/tests/pcc/test_pointpillars.py
```
This runs an end-to-end flow that:
  - Loads the PointPillars PyTorch reference,
  - Runs the TT-NN graph,
  - Compares results (PCC validation).

### Multi-Device:
To run multi-device test:
```
pytest models/experimental/pointpillars/tests/pcc/test_pointpillars.py --device-params '{"l1_small_size": 79104}'
```

### Run the Demo
```
python3 models/experimental/pointpillars/demo/demo.py
```
The demo automatically downloads sample data and weights on first run.

### Custom Point Clouds
Sample data is automatically downloaded from:
- [PointPillars Demo Data](https://github.com/zhulf0804/PointPillars/tree/main/pointpillars/dataset/demo_data)

Files are placed under:
```
models/experimental/pointpillars/resources/
```

## Performance
### Single Device (BS=1, n150)
- End-to-end perf (trace enabled, 2CQ): `19.7` FPS
- Device perf: `21` FPS

### Multi-Device (BS=2, n300)
- End-to-end perf (trace enabled, 2CQ): `40` FPS

To run performance tests:
```
pytest models/experimental/pointpillars/tests/perf/test_pointpillars_perf_e2e.py::test_pointpillars_perf_single_device -s
pytest models/experimental/pointpillars/tests/perf/test_pointpillars_perf_e2e.py::test_pointpillars_perf_multi_device -s

```

## Configuration Notes
- **Input**: LiDAR point cloud (`.bin` format) with (N, 4) array [x, y, z, intensity]
- **Point Cloud Range**: [0, -39.68, -3, 69.12, 39.68, 1] meters
- **Voxel Size**: [0.16, 0.16, 4] meters
- **Feature Map Size**: (H, W) = (496, 432) after pillar encoding, (248, 216) output
- **Device**: The demo opens a Wormhole device (default ID is 0). Adjust with `--device_id` flag.
- **Batch Size**: Demo and tests are written for BS=1. For larger batch sizes, verify memory layouts and tile alignment.
- **Classes**: 3 classes (Car, Pedestrian, Cyclist) for KITTI dataset

## References
- **Original Paper**: [PointPillars: Fast Encoders for Object Detection from Point Clouds](https://arxiv.org/abs/1812.05784)
- **Reference Implementation**: [PointPillars PyTorch](https://github.com/zhulf0804/PointPillars) by zhulf0804 (MIT License)
