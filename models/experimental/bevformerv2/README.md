# BEVFormerV2 (TT-NN)

**Platforms:** Wormhole (n150)
## Introduction

BEVFormerV2 (BEVFormer v2) is a state-of-the-art camera-based 3D object detection method that adapts modern image backbones to Bird's-Eye-View (BEV) recognition via perspective supervision. It builds upon the original BEVFormer framework, which learns unified BEV representations with spatiotemporal transformers to support multiple autonomous driving perception tasks.

This repository provides:
- A **reference PyTorch model** for correctness (ResNet-50 backbone + FPN).
- A **TT-NN implementation** for Tenstorrent hardware (Wormhole).
- **PCC tests** for component-level and end-to-end validation.
- **Performance benchmarks** and **resources** (weights + sample assets).

## Table of Contents
- [Prerequisites](#prerequisites)
- [Repository Layout](#repository-layout)
- [Weights](#weights)
- [Quickstart](#quickstart)
  - [Run PCC Tests](#run-pcc-tests)
  - [Component Testing](#component-testing)
  - [Performance Testing](#performance-testing)
- [Performance](#performance)
- [Configuration Notes](#configuration-notes)

## Prerequisites
- Clone the **tt-metal** repository (source code & toolchains):
  <https://github.com/tenstorrent/tt-metal>
- Install **TT-Metalium™ / TT-NN™**:
  Follow the official instructions: <https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md>
- (Optional, for profiling) Build with profiler enabled:
  ```bash
  ./build_metal.sh --enable-profiler
  ```

## Repository Layout
```
models/
└── experimental/
    └── bevformerv2/
        ├── resources/
        │   ├── resnet50_backbone.pth    # ResNet-50 backbone weights
        │   └── fpn_weights.pth          # FPN weights
        ├── reference/
        │   ├── resnet.py                # Reference ResNet-50 implementation (MMDetection style)
        │   └── fpn.py                   # Reference FPN implementation
        ├── tt/
        │   ├── tt_resnet.py             # TT-NN ResNet-50 implementation
        │   ├── tt_fpn.py                # TT-NN FPN implementation
        │   ├── tt_bottleneck.py         # TT-NN ResNet bottleneck block
        │   ├── utils.py                 # TT-NN utility functions (create_conv2d_configuration, etc.)
        │   └── model_configs.py         # Configuration helpers for BEVFormerV2
        ├── common.py                    # Weight loading and common utilities
        ├── README.md
        └── tests/
            ├── perf/
            │   └── test_perf.py         # Performance benchmarks
            └── pcc/
                ├── test_tt_resnet.py    # ResNet-50 PCC tests
                ├── test_tt_fpn.py       # FPN PCC tests
                └── test_bevformerv2.py  # End-to-end tests (if available)
```

## Weights
The model expects pretrained weights in:

```
models/experimental/bevformerv2/resources/resnet50_backbone.pth
models/experimental/bevformerv2/resources/fpn_weights.pth
```

These weights are used for:
- **ResNet-50 backbone**: MMDetection-style ResNet-50 pretrained on ImageNet/COCO
- **FPN**: Feature Pyramid Network weights compatible with the MMDetection FPN implementation

Note: The weights should be compatible with the MMDetection framework's ResNet-50 and FPN implementations.

## Quickstart

### Run PCC Tests

PCC (Pearson Correlation Coefficient) tests validate the numerical correctness of the TT-NN implementation against the reference PyTorch model.

#### Component Testing

**Test ResNet-50 Backbone:**
```bash
pytest models/experimental/bevformerv2/tests/pcc/test_tt_resnet.py::test_bevformerv2_resnet_matches_reference -v
```

This test:
- Loads the reference ResNet-50 model with pretrained weights
- Runs inference on both PyTorch and TT-NN implementations
- Compares outputs at C3, C4, C5 feature levels
- Validates PCC ≥ 0.99 for each level

**Test FPN:**
```bash
pytest models/experimental/bevformerv2/tests/pcc/test_tt_fpn.py::test_bevformerv2_fpn_matches_reference -v
```

This test:
- Loads the reference FPN model with pretrained weights
- Uses ResNet-50 backbone to generate input features (C3, C4, C5)
- Runs inference on both PyTorch and TT-NN FPN implementations
- Compares outputs at P3, P4, P5, P6, P7 levels
- Validates PCC ≥ 0.99 for each level

**Test FPN with Synthetic Features:**
```bash
pytest models/experimental/bevformerv2/tests/pcc/test_tt_fpn.py::test_bevformerv2_fpn_pretrained_weights -v
pytest models/experimental/bevformerv2/tests/pcc/test_tt_fpn.py::test_bevformerv2_fpn_random_weights -v
```

These tests use synthetic input features instead of a backbone, useful for isolating FPN behavior.

#### Run All PCC Tests
```bash
pytest models/experimental/bevformerv2/tests/pcc/ -v
```

### Performance Testing

To run performance benchmarks:
```bash
pytest models/experimental/bevformerv2/tests/perf/test_perf.py -v
```

This runs device performance tests and generates performance reports.

To collect perf reports with the profiler, build with `--enable-profiler`:
```bashd
./build_metal.sh --enable-profiler
```

## Performance

### Single Device (BS=2):
- ResNet-50 backbone performance: See `test_perf.py` for current benchmarks
- FPN performance: See `test_perf.py` for current benchmarks

To run perf test:
```bash
pytest models/experimental/bevformerv2/tests/perf/test_perf.py
```

## Configuration Notes

- **Resolution**: Default test resolution is (256, 256) for batch size 2. The implementation supports configurable input resolutions.

- **Device**: Tests open a Wormhole device (default id typically 0). If you need to change it, adjust the device open call in the test files.

- **Batch Size**: Tests are written for BS=2. For different batch sizes, verify memory layouts and tile alignment.

- **Memory Layouts**: The TT-NN path uses:
  - `HEIGHT_SHARDED` layout for most convolutions
  - `BLOCK_SHARDED` layout for specific layers (configurable via `BevFormerV2ModelConfig`)
  - `DRAM_MEMORY_CONFIG` for intermediate activations

- **Weights**: The loader supports:
  - Automatic state_dict extraction from checkpoint files
  - Prefix stripping for compatibility with different checkpoint formats
  - Flexible checkpoint loading via `load_resnet50_backbone_weights()` and `load_fpn_weights()`

- **Model Configuration**: Use `BevFormerV2ModelConfig` to customize:
  - Activation and weight dtypes (bfloat16, bfloat8_b, etc.)
  - Shard layouts per layer
  - Math fidelity settings
  - Activation deallocation policies

- **PCC Threshold**: Default PCC threshold is 0.99 for all component tests. This ensures high numerical accuracy between PyTorch and TT-NN implementations.

## Architecture

The BEVFormerV2 TT-NN implementation consists of:

1. **ResNet-50 Backbone** (`TtResNet50_MMD_C345`):
   - MMDetection-style ResNet-50
   - Returns C3, C4, C5 feature maps (1/8, 1/16, 1/32 spatial resolution)
   - Uses optimized bottleneck blocks with configurable sharding

2. **FPN** (`TtFPN`):
   - Feature Pyramid Network for multi-scale feature fusion
   - Lateral connections from backbone features
   - Top-down pathway with nearest-neighbor upsampling
   - Extra levels (P6, P7) for detection heads

## References

- **BEVFormer Paper**: [BEVFormer: Learning Bird's-Eye-View Representation from Multi-Camera Images via Spatiotemporal Transformers](http://arxiv.org/abs/2203.17270), ECCV 2022
- **BEVFormerV2 Paper**: [BEVFormer v2: Adapting Modern Image Backbones to Bird's-Eye-View Recognition via Perspective Supervision](https://arxiv.org/abs/2211.10439)
- **Original Repository**: <https://github.com/fundamentalvision/BEVFormer>
