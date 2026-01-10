# CenterNet

### Platforms: Wormhole (n150)
### Supported Input Resolution:** `(512, 512)` = (Height, Width)

## Introduction
CenterNet with DLA-34 backbone (without deformable convolutions) is a real-time object detection model that represents objects as center points and performs detection in a single forward pass. The model uses a Deep Layer Aggregation (DLA-34) backbone with standard convolutions and upsampling layers to detect object centers and regress bounding box properties.

## Prerequisites
- Cloned [tt-metal repository](https://github.com/tenstorrent/tt-metal) for source code
- Installed: [TT-Metalium™ / TT-NN™](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md)

## Setup

### Download Weights
To use trained weights:
1. Download trained weigths from Google Drive link in [CenterNet MODEL ZOO](https://github.com/xingyizhou/CenterNet/blob/master/readme/MODEL_ZOO.md). Make sure to download the currently supported `ctdet_coco_dlav0_1x.pth`
2. Load weights into the PyTorch model before transferring to TTNN


## Repository Layout
```
models/
└── experimental/
    └── centernet/
        ├── resources/
        │   └── sample_input/
        │       └── 16004479832_a748d55f21_k.jpg
        ├── reference/
        │   ├── model.py                  # CenterNet (reference)
        │   ├── network/
        │   │   ├── __init__.py
        │   │   └── dlav0.py              # DLA-34 backbone
        │   └── utils/
        │       ├── __init__.py
        │       ├── ddd_utils.py
        │       ├── debugger.py
        │       ├── decode.py
        │       ├── image.py
        │       ├── post_process.py
        │       └── utils.py
        ├── tt/
        │   ├── basic_block.py            # Basic block (TTNN)
        │   ├── custom_preprocessor.py
        │   ├── dla_seg.py                # DLA segment (TTNN)
        │   ├── dla.py                    # DLA backbone (TTNN)
        │   ├── dlaup.py                  # DLA upsampling (TTNN)
        │   ├── root.py                   # Root block (TTNN)
        │   ├── tree.py                   # Tree structure (TTNN)
        │   └── utils.py                  # convtranspose utils
        ├── demo/
        │   └── demo.py                   # CLI demo
        ├── tests/
        │   ├── pcc/
        │   │   ├── test_basic_block.py
        │   │   ├── test_dla_seg.py
        │   │   ├── test_dla.py
        │   │   ├── test_dlaup.py
        │   │   ├── test_root.py
        │   │   └── test_tree.py
        │   └── perf/
        │       ├── test_centernet_e2e_perf.py
        │       ├── test_perf.py
        │       └── performant_infra.py   # CenterNetPerformantTestInfra class
        ├── ctdet_coco_dlav0_1x.pth       # Trained weights
        └── README.md
```

## Details

- The entry point to the TTNN CenterNet model is `TtDLASeg` in `models/experimental/centernet/tt/dla_seg.py`. The model uses random weights from the PyTorch reference implementation.
- Performance test infrastructure is encapsulated in `CenterNetPerformantTestInfra` class in `models/experimental/centernet/tests/perf/performant_infra.py`.

## How to Run

### Run the Full Model Test
```bash
# From tt-metal root directory
pytest models/experimental/centernet/tests/pcc/test_dla_seg.py
```

### Performance
### Single Device (BS=1):
- Expected throughput: `83.26` FPS

### Run Device Performance Test
```bash
# Test full model performance
pytest models/experimental/centernet/tests/perf/test_centernet_e2e_perf.py
```

### Run the Demo
```bash
# Process a single image
python3 models/experimental/centernet/demo/demo.py --input_image <path_to_image> --weights <path_to_weights>

```

Example:
```bash
python models/experimental/centernet/demo/demo.py --input models/experimental/centernet/resources/sample_input/16004479832_a748d55f21_k.jpg --weights models/experimental/centernet/ctdet_coco_dlav0_1x.pth
```

### Demo Output Files

The demo generates output files for each processed image:
- `ttnn.png`: TTNN detection results with bounding boxes and labels.
- `pytroch.png`: TTNN detection results with bounding boxes and labels.
- Output is saved to a newly generated directory named `outputs` in demo directory or to the path specified by `--output_path`

## Configuration Notes
- Resolution: (H, W) = (512, 512) is supported end-to-end.
- Device: The demo opens a Wormhole device (default id typically 0). If you need to change it, adjust the DemoConfig or the device open call in the demo.
- Batch Size: Demo/tests are written for BS=1. For larger BS you’ll need to verify memory layouts and tile alignment.
