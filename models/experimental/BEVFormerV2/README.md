# BEVFormerV2

**Platforms:** Wormhole (n150)
**Supported Input Resolution:** `(256, 704)` = (Height, Width)

## Introduction

BEVFormerV2 is a transformer-based approach for multi-view 3D object detection that generates Bird's Eye View (BEV) representations from multi-camera inputs. The model uses a **Perception Transformer** architecture with a **ResNet-50** backbone and **FPN** neck to process multi-camera inputs and generate 3D object detections in BEV space.

This implementation adapts **BEVFormerV2** for Tenstorrent hardware using the TT-NN and TT-Metalium stack, optimized for throughput and low-latency inference on Wormhole devices. The implementation supports 6-camera inputs with 256×704 resolution.

This repository provides:
- A **reference PyTorch model** (from [fundamentalvision/BEVFormer](https://github.com/fundamentalvision/BEVFormer)) for correctness validation.
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

- Clone the **tt-metal** repository (source code & toolchains): [https://github.com/tenstorrent/tt-metal](https://github.com/tenstorrent/tt-metal)
- Install **TT-Metalium™ / TT-NN™**: Follow the official instructions: [https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md)
- Install additional dependencies for BEVFormerV2 as mentioned in "tt_metal/python_env/requirements-dev.txt" if not already present.

## Repository Layout

```
models/
└── experimental/
    └── BEVFormerV2/
        ├── demo/
        │   ├── demo_data/samples/
        │   |    ├── CAM_BACK/
        │   |    ├── CAM_BACK_LEFT/
        │   |    ├── CAM_BACK_RIGHT/
        │   |    ├── CAM_FRONT/
        │   |    ├── CAM_FRONT_LEFT/
        │   |    ├── CAM_FRONT_RIGHT/
        │   ├── demo_data_loader.py # Demo data loader
        │   ├── test.py # Test to generate predictions
        ├── reference/
        │   ├── bevformer_v2.py # Main BEVFormerV2 model
        │   ├── ... # Reference model components
        ├── tt/
        │   ├── model_preprocessing.py # Model preprocessing utilities
        │   ├── ttnn_bevformer_v2.py # Main TTNN model wrapper (TtBevFormerV2)
        │   ├── ttnn_encoder.py # Encoder implementation
        │   ├── ttnn_decoder.py # Decoder implementation
        │   ├── ttnn_perception_transformer.py # Perception transformer
        │   ├── ttnn_bevformer_head.py # Detection head
        │   ├── ttnn_backbone.py # ResNet-50 backbone
        │   ├── ttnn_fpn.py # FPN neck
        │   ├── utils.py # Utility functions
        ├── tests/
        │   ├── pcc/ # Pearson Correlation Coefficient tests
        │   │   ├── test_bevformer_v2.py # End-to-end functional test
        │   │   ├── test_perception_transformer.py # Perception transformer test
        │   │   ├── test_bevformer_head.py # Head test
        │   │   ├── test_decoder_layer.py # Decoder layer test
        │   │   └── test_ffn.py # FFN test
        │   ├── perf/ # Performance tests
        │       ├── test_device_perf.py # Device performance test
        |       ├── test_e2e_performant.py # End to end performance test with 1cq+no trace
        |       ├── test_e2e_performant_2cq_no_trace.py # End to end performance test with 2cq+no trace
        ├── common.py # Common utilities
        ├── README.md
```

## Weights

BEVFormerV2 pretrained weights are automatically downloaded when running the model. The weights are from the official BEVFormer repository:

- **Model:** BEVFormerV2 (ResNet-50 backbone, 6 encoder + 6 decoder layers)
- **Checkpoint Location:** Auto-downloaded via `common.py` on first use

Note: The weights are trained on the nuScenes dataset.

## Quickstart

### Run Tests

```bash
pytest models/experimental/BEVFormerV2/tests/pcc/test_bevformer_v2.py
```

This runs an end-to-end flow that:
- Loads the BEVFormerV2 reference model from PyTorch,
- Runs the TT-NN implementation,
- Compares results (PCC validation),
- Validates outputs (bev_embed, all_cls_scores, all_bbox_preds).

### Component Tests

```bash
# Test Backbone
pytest models/experimental/BEVFormerV2/tests/pcc/test_backbone.py

# Test FPN
pytest models/experimental/BEVFormerV2/tests/pcc/test_fpn.py

# Test Head
pytest models/experimental/BEVFormerV2/tests/pcc/test_bevformer_head.py

# Test Full Network
pytest models/experimental/BEVFormerV2/tests/pcc/test_bevformer_v2.py
```

### Run the Demo

```bash
python3 models/experimental/BEVFormerV2/demo/test.py --data-root models/experimental/BEVFormerV2/demo/demo_data --sample-idx 0 --out models/experimental/BEVFormerV2/demo/outputs/results.json
```

**Options:**
- `--data-root`: Path to demo data directory (default: `models/experimental/BEVFormerV2/demo/demo_data`)
- `--sample-idx`: Sample index to process (default: 0, use -1 for all samples)
- `--out`: Output JSON file path (default: `models/experimental/BEVFormerV2/demo/outputs/results.json`)

The demo processes sample nuScenes data and outputs 3D object detections in JSON format.

## Performance

### Single Device (BS=1)(n150):

- Device perf is **0.105** FPS
- E2E perf with 2 command queue is **0.08** FPS (without trace)

### Run Device Performance Test

```bash
pytest models/experimental/BEVFormerV2/tests/perf/test_device_perf.py -s
```
### Run End to End Performance Test

```bash
pytest models/experimental/BEVFormerV2/tests/perf/test_e2e_performant_2cq_no_trace.py -s
```

**PCC Scores:**
| Output | PCC Score |
|--------|-----------|
| all_cls_scores | 0.99 |
| all_bbox_preds | 0.99 |
| bev_embed | 0.99 |

All tests use PCC validation with threshold: 0.99.

## Configuration Notes

- **Resolution:** (H, W) = (256, 704) is supported end-to-end.
- **Device:** The demo/tests open a Wormhole device (default id typically 0). If you need to change it, adjust the device open call in the demo.
- **Batch Size:** Tests are written for BS=1. For larger BS you'll need to verify memory layouts and tile alignment.
- **Number of Cameras:** 6 cameras (CAM_FRONT, CAM_FRONT_LEFT, CAM_FRONT_RIGHT, CAM_BACK, CAM_BACK_LEFT, CAM_BACK_RIGHT).
- **Transformer Layers:** 6 encoder + 6 decoder layers.
- **BEV Resolution:** 100×100 (bev_h × bev_w) (memory constraints).
- **Weights:** Auto-downloaded via `common.py` on first use.

## References

### Paper

- **BEVFormer v2: Adapting Modern Image Backbones to Bird's-Eye-View Recognition via Perspective Supervision**
  - Authors: Zhiqi Li, Wenhai Chen, Hongyang Li, et al.
  - arXiv: [https://arxiv.org/abs/2211.10439](https://arxiv.org/abs/2211.10439)
  - Year: 2023

### Source Code implementation and licenses
- ***BEVFormerV2**: https://github.com/fundamentalvision/BEVFormer(Apache License 2.0)
- **MMCV**: https://github.com/open-mmlab/mmcv/tree/v1.4.0/mmcv (Apache License 2.0)
- **MMSegmentation**: https://github.com/open-mmlab/mmsegmentation/tree/v0.14.1/mmseg (Apache License 2.0)
- **MMDetection3D**: https://github.com/open-mmlab/mmdetection3d/tree/v0.17.1/mmdet3d (Apache License 2.0)
- **MMDetection**: https://github.com/open-mmlab/mmdetection/tree/v2.14.0/mmdet (Apache License 2.0)
- **MMEngine**: https://github.com/open-mmlab/mmengine/blob/main/mmengine (Apache License 2.0)
