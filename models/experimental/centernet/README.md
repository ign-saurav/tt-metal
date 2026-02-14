# CenterNet

### Platforms: Wormhole (n150)
### Supported Input Resolution:** `(512, 512)` = (Height, Width)

## Introduction
CenterNet with DLA-34 backbone (without deformable convolutions) is a real-time object detection model that represents objects as center points and performs detection in a single forward pass. The model uses a Deep Layer Aggregation (DLA-34) backbone with standard convolutions and upsampling layers to detect object centers and regress bounding box properties.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          CenterNet DLA-34                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                        INPUT IMAGE                                │  │
│  │                     (512 x 512 x 3)                               │  │
│  └─────────────────────────────┬─────────────────────────────────────┘  │
│                                │                                        │
│                                ▼                                        │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                    DLA-34 BACKBONE                                │  │
│  │                                                                   │  │
│  │  ┌─────────────┐                                                 │  │
│  │  │ Base Layer  │  Conv7x7 → BN → ReLU                            │  │
│  │  │   (ch=16)   │                                                 │  │
│  │  └──────┬──────┘                                                 │  │
│  │         │                                                         │  │
│  │         ▼                                                         │  │
│  │  ┌─────────────┐                                                 │  │
│  │  │   Level 0   │  Conv layers (ch=16)                            │  │
│  │  └──────┬──────┘                                                 │  │
│  │         │                                                         │  │
│  │         ▼                                                         │  │
│  │  ┌─────────────┐                                                 │  │
│  │  │   Level 1   │  Conv layers (ch=32, stride=2)                  │  │
│  │  └──────┬──────┘                                                 │  │
│  │         │                                                         │  │
│  │         ▼                                                         │  │
│  │  ┌─────────────┐                                                 │  │
│  │  │   Level 2   │  Tree structure (ch=64, stride=2)               │  │
│  │  │             │  BasicBlock × 1 level                           │  │
│  │  └──────┬──────┘                                                 │  │
│  │         │                                                         │  │
│  │         ▼                                                         │  │
│  │  ┌─────────────┐                                                 │  │
│  │  │   Level 3   │  Tree structure (ch=128, stride=2)              │  │
│  │  │             │  BasicBlock × 2 levels                          │  │
│  │  └──────┬──────┘                                                 │  │
│  │         │                                                         │  │
│  │         ▼                                                         │  │
│  │  ┌─────────────┐                                                 │  │
│  │  │   Level 4   │  Tree structure (ch=256, stride=2)              │  │
│  │  │             │  BasicBlock × 2 levels                          │  │
│  │  └──────┬──────┘                                                 │  │
│  │         │                                                         │  │
│  │         ▼                                                         │  │
│  │  ┌─────────────┐                                                 │  │
│  │  │   Level 5   │  Tree structure (ch=512, stride=2)              │  │
│  │  │             │  BasicBlock × 1 level                           │  │
│  │  └──────┬──────┘                                                 │  │
│  │         │                                                         │  │
│  └─────────┼─────────────────────────────────────────────────────────┘  │
│            │                                                            │
│            ▼                                                            │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                    DLA UPSAMPLING (DLAUp)                         │  │
│  │                                                                   │  │
│  │  Iterative upsampling and aggregation (IDAUp modules)            │  │
│  │  - Projects features to common dimension (64 channels)           │  │
│  │  - Upsamples using ConvTranspose2d                               │  │
│  │  - Aggregates multi-scale features with 3x3 convolutions         │  │
│  │                                                                   │  │
│  │  Output: 128 x 128 x 64 feature map                              │  │
│  └─────────────────────────────┬─────────────────────────────────────┘  │
│                                │                                        │
│                                ▼                                        │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                    DETECTION HEADS                                │  │
│  │                                                                   │  │
│  │  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐         │  │
│  │  │   Heatmap    │   │  Width/Height│   │    Offset    │         │  │
│  │  │     (hm)     │   │     (wh)     │   │     (reg)    │         │  │
│  │  │              │   │              │   │              │         │  │
│  │  │ Conv3x3(256) │   │ Conv3x3(256) │   │ Conv3x3(256) │         │  │
│  │  │    → ReLU    │   │    → ReLU    │   │    → ReLU    │         │  │
│  │  │ Conv1x1(80)  │   │ Conv1x1(2)   │   │ Conv1x1(2)   │         │  │
│  │  │              │   │              │   │              │         │  │
│  │  │ 128x128x80   │   │ 128x128x2    │   │ 128x128x2    │         │  │
│  │  └──────────────┘   └──────────────┘   └──────────────┘         │  │
│  │                                                                   │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**Key architectural details:**
- **DLA-34 Backbone**: Deep Layer Aggregation with 6 hierarchical levels using BasicBlock
- **Tree Structure**: Hierarchical feature aggregation with residual connections (levels 2-5)
- **DLA Upsampling**: Progressive upsampling with IDAUp modules for multi-scale feature fusion
- **Detection Heads**: Three parallel heads for center heatmap (80 classes), bounding box size, and center offset
- **Output Resolution**: 128×128 (down_ratio=4 from 512×512 input)

## Prerequisites
- Cloned [tt-metal repository](https://github.com/tenstorrent/tt-metal) for source code
- Installed: [TT-Metalium™ / TT-NN™](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md)

## Setup

### Download Weights

CenterNet weightswill be automatically downloaded while running the demo and will be saved:

- **Source**: [CenterNet MODEL ZOO](https://drive.google.com/drive/folders/1S3NnppRgXea_IG4WeyquJcnOB3I6G-LX)
- **Model**: `ctdet_coco_dlav0_1x.pth` (DLA-34 backbone trained on COCO, 211.6 MB)
- **Location**: `models/experimental/centernet/ctdet_coco_dlav0_1x.pth`

**Download Options:**

**Option 1 - Using wget (no additional installation required):**
```bash
cd models/experimental/centernet
wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1pl_-ael8wERdUREEnaIfqOV_VF2bEVRT' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1pl_-ael8wERdUREEnaIfqOV_VF2bEVRT" -O ctdet_coco_dlav0_1x.pth && rm -rf /tmp/cookies.txt
```

**Option 2 - Using gdown (simpler, but requires pip install):**
```bash
pip install gdown
cd models/experimental/centernet
gdown 1pl_-ael8wERdUREEnaIfqOV_VF2bEVRT -O ctdet_coco_dlav0_1x.pth
```

**Option 3 - Manual download from browser:**
1. Visit: https://drive.google.com/file/d/1pl_-ael8wERdUREEnaIfqOV_VF2bEVRT/view
2. Click the 'Download' button
3. Save the file as `ctdet_coco_dlav0_1x.pth` in `models/experimental/centernet/`


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
        │   │   ├── test_detection_heads.py
        │   │   └── test_tree.py
        │   └── perf/
        │       ├── test_centernet_e2e_perf.py
        │       ├── test_perf.py
        │       └── performant_infra.py   # CenterNetPerformantTestInfra class
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
- Expected throughput: `91` FPS

### Run End To End Application Performance Test
```bash
# Test full model performance
pytest models/experimental/centernet/tests/perf/test_centernet_e2e_perf.py
```
### Run Device Level Performance Test
- Expected throughput: `115` FPS
```bash
# Test full model performance
pytest models/experimental/centernet/tests/perf/test_perf.py
```

### Run the Demo
```bash
# Process a single image (weights auto-detected)
python3 models/experimental/centernet/demo/demo.py --input <path_to_image>

# Or specify weights explicitly
python3 models/experimental/centernet/demo/demo.py --input <path_to_image> --weights <path_to_weights>
```

Example:
```bash
# Auto-detect weights
python3 models/experimental/centernet/demo/demo.py --input models/experimental/centernet/resources/sample_input/16004479832_a748d55f21_k.jpg

# Or specify weights
python3 models/experimental/centernet/demo/demo.py --input models/experimental/centernet/resources/sample_input/16004479832_a748d55f21_k.jpg --weights models/experimental/centernet/ctdet_coco_dlav0_1x.pth
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

## References

- Original Paper: Objects as Points — https://arxiv.org/abs/1904.07850
- Reference Implementation: CenterNet by Xingyi Zhou (MIT License) — https://github.com/xingyizhou/CenterNet
