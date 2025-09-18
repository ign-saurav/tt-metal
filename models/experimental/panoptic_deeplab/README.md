# Panoptic-DeepLab (TT-NN)

**Platforms:** Wormhole (n150)
**Supported Input Resolution:** `(512, 1024)` = (Height, Width)

## Introduction
Panoptic-DeepLab is a state-of-the-art bottom-up method for panoptic segmentation, where the goal is to assign semantic labels (e.g., person, dog, cat and so on) to every pixel in the input image as well as instance labels (e.g. an id of 1, 2, 3, etc) to pixels belonging to thing classes.

This repository provides:
- A **reference PyTorch model** for correctness.
- A **TT-NN implementation** for Tenstorrent hardware (Wormhole).
- A **demo pipeline**, **tests**, and **resources** (weights + sample assets).

## Table of Contents
- [Prerequisites](#prerequisites)
- [Repository Layout](#repository-layout)
- [Weights](#weights)
- [Quickstart](#quickstart)
  - [Run Tests](#run-tests)
  - [Run the Demo](#run-the-demo)
  - [Custom Images](#custom-images)
- [Performance (Trace + 2CQ)](#performance-trace--2cq)
- [Configuration Notes](#configuration-notes)

## Prerequisites
- Clone the **tt-metal** repository (source code & toolchains):
  <https://github.com/tenstorrent/tt-metal>
- Install **TT-Metalium™ / TT-NN™**:
  Follow the official instructions: <https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md>
- (Optional, for profiling) Build with profiler enabled:
  ```bash
  ./build_metal.sh --enable-profiler

## Repository Layout
```
models/
└── experimental/
    └── panoptic_deeplab/
        ├── resources/
        │   ├── test_inputs/
        │   │   └── input_torch_input.pt # generated and stored during runtime
        │   ├── input.png
        │   ├── Panoptic_Deeplab_R52.pkl # downloaded during runtime if not present in the directory
        │   └── panoptic_deeplab_weights_download.sh
        ├── reference/
        │   ├── aspp.py
        │   ├── decoder.py
        │   ├── head.py
        │   ├── panoptic_deeplab.py        # TorchPanopticDeepLab (reference)
        │   ├── res_block.py
        │   ├── resnet52_backbone.py
        │   ├── resnet52_bottleneck.py
        │   ├── resnet52_stem.py
        │   └── utils.py
        ├── tt/
        │   ├── aspp.py
        │   ├── backbone.py
        │   ├── bottleneck.py
        │   ├── custom_peprocessing.py
        │   ├── decoder.py
        │   ├── head.py
        │   ├── panoptic_deeplab.py
        │   ├── res_block.py
        │   ├── stem.py
        │   └── utils.py
        ├── runner/
        │   └── runner.py
        ├── common.py
        ├── README.md
        ├── demo/
        │   ├── config.py
        │   ├── post_proessing.py
        │   └── panoptic_deeplab_demo.py    # CLI demo
        └── tests/
          ├── perf/
          │   ├── test_perf.py
          └── pcc/
              └── test_panoptic_deeplab.py    # end-to-end pytest
              └── test_aspp.py
              └── test_decoder.py
              └── test_head.py
              └── test_residual_block.py
              └── test_resnet52_backbone.py
              └── test_resnet52_bottleneck.py
              └── test_resnet52_stem.py
```

## Weights
The default model expects Panoptic_Deeplab_R52.pkl in:

```
models/experimental/panoptic_deeplab/resources/Panoptic_Deeplab_R52.pkl
```
If missing, the code will attempt to run:
```
models/experimental/panoptic_deeplab/resources/panoptic_deeplab_weights_download.sh
```

## Quickstart
### Run Tests
```
models/experimental/panoptic_deeplab/tests/pcc/test_panoptic_deeplab.py
```
This runs an end-to-end flow that:

  - Loads the Torch reference,

  - Runs the TT-NN graph,

  - Post-processes outputs,

  - Optionally compares results and saves artifacts.

### Run the Demo
```
python models/experimental/panoptic_deeplab/demo/panoptic_deeplab_demo.py \
  --input  <path/to/image.png> \
  --output <path/to/output_dir>
```
### Custom Images
You can place your image(s) under:
```
models/experimental/panoptic_deeplab/resources/
```
Then re-run either the demo:
```
python models/experimental/panoptic_deeplab/demo/panoptic_deeplab_demo.py
-i models/experimental/panoptic_deeplab/resources/input.png
-o models/experimental/panoptic_deeplab/resources
```

For visualizing heads comparison of PyTorch and TTNN implementation, enable save_comparison in demo/config.


## Performance
### Single Device (BS=1):

- end-2-end perf is `12.81` FPS

To run perf test:
```
pytest models/experimental/panoptic_deeplab/tests/perf/test_perf.py
```

To collect perf reports with the profiler, build with `--enable-profiler`

## Configuration Notes

- Resolution: (H, W) = (512, 1024) is supported end-to-end.

- Device: The demo opens a Wormhole device (default id typically 0). If you need to change it, adjust the DemoConfig or the device open call in the demo.

- Batch Size: Demo/tests are written for BS=1. For larger BS you’ll need to verify memory layouts and tile alignment.

- Memory Layouts: The TT-NN path uses ROW_MAJOR layout for resize ops and may pad channels to multiples of 32 to satisfy kernel/tile alignment.

- Weights: The loader maps Detectron/PDL keys → internal module keys. It will auto-download weights if missing via the included script.
