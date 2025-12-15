# PointPillars Model

PointPillars is a real-time 3D object detection model optimized for Tenstorrent hardware using TTNN (Tenstorrent Neural Network). This implementation provides both PyTorch reference and TTNN-accelerated versions of the PointPillars architecture for LiDAR point cloud processing.

## Overview

PointPillars is a state-of-the-art 3D object detection model designed for autonomous driving applications. It efficiently converts sparse 3D point clouds into a dense 2D pseudo-image representation using vertical pillars, enabling fast inference with standard 2D convolutional networks while maintaining high detection accuracy.

### Key Features

- **Efficient Architecture**: Converts 3D point clouds to 2D pillar representation for fast processing
- **Real-time Detection**: Optimized for low-latency inference on Tenstorrent hardware
- **Multi-class Detection**: Detects 3 object classes (Car, Pedestrian, Cyclist) for KITTI dataset
- **TTNN Acceleration**: Optimized for Tenstorrent hardware with bfloat16 precision
- **High Accuracy**: Achieves competitive performance on KITTI validation set

## Model Architecture

The PointPillars model consists of five main components:

1. **PillarLayer**: Converts raw 3D point clouds into pillar representation
   - Input: Raw LiDAR point cloud (N points with x, y, z, intensity)
   - Output: Pillars, coordinates, and point counts per pillar
   - Execution: PyTorch (Host)

2. **PillarEncoder**: Encodes pillar features using 1D convolutions
   - Input: Pillars with point coordinates
   - Output: Feature maps (batch, 64, 496, 432)
   - Architecture: Conv1D + BatchNorm + ReLU + MaxPool
   - Execution: Hybrid (Indexing on host, encoding on device)

3. **Backbone**: Multi-scale feature extraction with 2D convolutions
   - Input: Pillar features (batch, 64, 496, 432)
   - Output: Multi-scale features at 3 levels
     - Level 1: (batch, 64, 248, 216)
     - Level 2: (batch, 128, 124, 108)
     - Level 3: (batch, 256, 62, 54)
   - Execution: **TTNN**

4. **Neck**: Feature fusion using transposed convolutions for upsampling
   - Input: Multi-scale features from backbone
   - Output: Fused features (batch, 384, 248, 216)
   - Architecture: Transposed convolutions with stride [1, 2, 4]
   - Execution: **TTNN**

5. **Head**: Detection outputs (classification, regression, direction)
   - Input: Fused features from neck
   - Output: Three detection outputs
     - Classification: (batch, n_anchors*3, 248, 216)
     - Regression: (batch, n_anchors*7, 248, 216)
     - Direction: (batch, n_anchors*2, 248, 216)
   - Execution: **TTNN**

### Model Specifications

- **Input Format**: LiDAR point cloud (.bin files)
- **Voxel Size**: [0.16, 0.16, 4] meters
- **Point Cloud Range**: [0, -39.68, -3, 69.12, 39.68, 1] meters
- **Max Points per Pillar**: 32
- **Max Voxels**: (16000, 40000) for training/inference
- **Number of Classes**: 3 (Car, Pedestrian, Cyclist)
- **Feature Map Size**: 496×432 (after pillar encoding)
- **Output Feature Map Size**: 248×216
- **Anchor Scales**: 3 scales per class
- **Anchor Rotations**: [0, π/2] radians

## References

- **Original Paper**: [PointPillars: Fast Encoders for Object Detection from Point Clouds](https://arxiv.org/abs/1812.05784)

## Performance (on N150)

### Inference Performance

##### FPS (Frames Per Second)

- **Batch Size 1, Single Device (N150)**: 36 FPS
- **Batch Size 2, Multi Device (N300)**: 70 FPS

##### Device Time

- **Single Device (N150)**: ~27.8 ms per frame (end-to-end with trace enabled and 2CQ)
- **Multi Device (N300)**: ~28.6 ms per frame (end-to-end with trace enabled and 2CQ)

**Measurement Configuration:**
- Input: LiDAR point cloud
- Batch size: 1 (single device) / 2 (multi device)
- Device: Tenstorrent (N150/N300)
- Precision: bfloat16

## Directory Structure

```
models/experimental/pointpillars/
├── README.md                          # This file
│
├── demo/                              # Demo application
│   ├── demo.py                       # Main demo script
│   └── output/                        # Output directory for visualization images
│
├── reference/                         # PyTorch reference implementation
│   ├── model/                        # Model components
│   │   ├── pointpillars.py          # Main PointPillars model
│   │   └── anchors.py                # Anchor generation
│   ├── ops/                          # Operations
│   │   ├── voxel_module.py          # Voxelization operations
│   │   └── iou3d_module.py          # 3D IoU operations
│   └── utils/                        # Utility functions
│       ├── io.py                     # I/O utilities
│       ├── process.py                # Post-processing utilities
│       └── vis_o3d.py                # 3D visualization
│
├── runner/                            # Performance runner infrastructure
│   └── performant_runner_infra.py    # Runner infrastructure
│
├── tests/                             # Test suite
│   ├── pcc/                          # PCC (Pearson Correlation Coefficient) tests
│   │   ├── test_pointpillars.py     # Main integration test
│   │   ├── test_backbone.py         # Backbone component test
│   │   ├── test_neck.py             # Neck component test
│   │   ├── test_head.py             # Head component test
│   │   ├── test_pillar_encoder.py   # Pillar encoder test
│   │   └── test_conv_transpose_split.py  # Conv transpose test
│   ├── perf/                         # Performance tests
│   │   └── test_pointpillars_perf_e2e.py  # End-to-end performance test
│   └── test_stability.py             # Stability test
│
└── tt/                                # TTNN implementation
    ├── pointpillars.py               # Main TTNN PointPillars model
    ├── pillar_encoder.py             # TTNN PillarEncoder implementation
    ├── backbone.py                   # TTNN Backbone implementation
    ├── neck.py                       # TTNN Neck implementation
    ├── head.py                       # TTNN Head implementation
    ├── pillar_layer.py               # Pillar layer utilities
    ├── custom_preprocessor.py        # Custom weight preprocessing
    └── utils.py                      # TTNN utility functions
```

## Setup

### Download Model Weights and Data

The model weights and sample data are required for inference:

**Model Weights:**
- Download from: [PointPillars Pretrained Weights](https://github.com/zhulf0804/PointPillars/blob/main/pretrained/epoch_160.pth)
- Save as: `epoch_160.pth` in the working directory

**Sample Data:**
- Download sample point cloud, image, and calibration data from: [PointPillars Demo Data](https://github.com/zhulf0804/PointPillars/tree/main/pointpillars/dataset/demo_data)
- Create a resource directory  `models/experimental/pointpillars/resources/` and place files in them.

## Usage

### Running the Demo

The demo script demonstrates 3D object detection on LiDAR point clouds:

```bash


# With calibration and image for visualization
python models/experimental/pointpillars/demo/demo.py \
    --ckpt models/experimental/pointpillars/resources//epoch_160.pth \
    --pc_path models/experimental/pointpillars/resources/000134.bin \
    --calib_path models/experimental/pointpillars/resources/000134.txt \
    --img_path models/experimental/pointpillars/resources/000134.png
```

**Demo Arguments:**
- `--ckpt`: Path to checkpoint file (default: `epoch_160.pth`)
- `--pc_path`: Path to point cloud file (.bin)
- `--calib_path`: Path to calibration file (.txt)
- `--img_path`: Path to image file (.png/.jpg)
- `--output`: Output directory for visualization images (default: `models/experimental/pointpillars/resources/output`)
- `--device_id`: Device ID to use (default: 0)

The demo will:
1. Load and preprocess the input point cloud
2. Run inference on both PyTorch reference and TTNN models
3. Post-process outputs to get 3D bounding boxes
4. Display detection summary (count of detections for PyTorch and TTNN)
5. Visualize detections and save output images (if calibration and image are provided)

### Running Tests

Run the test suite to verify model correctness:

```bash
# Run all PCC tests
pytest models/experimental/pointpillars/tests/pcc/ -v

# Run specific test
pytest models/experimental/pointpillars/tests/pcc/test_pointpillars.py -v

# Run with specific device parameters
pytest models/experimental/pointpillars/tests/pcc/test_pointpillars.py \
    --device-params '{"l1_small_size": 79104}' -v
```

**Test Coverage:**
- `test_pointpillars.py`: Full model integration test
- `test_backbone.py`: Backbone component test
- `test_neck.py`: Neck component test
- `test_head.py`: Head component test
- `test_pillar_encoder.py`: Pillar encoder component test
- `test_conv_transpose_split.py`: Conv transpose component test

### Performance Testing

To run performance tests:

```bash
# Single device (N150)
pytest models/experimental/pointpillars/tests/perf/test_pointpillars_perf_e2e.py::test_pointpillars_perf_single_device -s

# Multi device (N300)
pytest models/experimental/pointpillars/tests/perf/test_pointpillars_perf_e2e.py::test_pointpillars_perf_multi_device -s
```

## Model Components

### PillarLayer

Converts raw 3D point clouds into pillar representation:
- **Input**: List of point cloud tensors `[(N1, 4), (N2, 4), ...]` with x, y, z, intensity
- **Output**:
  - Pillars: `(total_pillars, max_points, 4)`
  - Coordinates: `(total_pillars, 4)` [batch_idx, x, y, z]
  - Point counts: `(total_pillars,)`
- **Voxelization**: Groups points into vertical pillars based on x-y coordinates

### PillarEncoder

Encodes pillar features using 1D convolutions:
- **Input**: Pillars, coordinates, point counts
- **Output**: `(batch, 64, 496, 432)` feature maps
- **Features**:
  - Calculates offsets to point center and pillar center
  - Encodes 9 features per point (4 original + 3 point offsets + 2 pillar offsets)
  - Applies Conv1D, BatchNorm, ReLU, and MaxPool
  - Scatters features to 2D feature map based on pillar coordinates

### Backbone

Multi-scale feature extraction with 2D convolutions:
- **Input**: `(batch, 64, 496, 432)` pillar features
- **Output**: Three feature maps at different scales:
  - Level 1: `(batch, 64, 248, 216)` - 1/2 scale
  - Level 2: `(batch, 128, 124, 108)` - 1/4 scale
  - Level 3: `(batch, 256, 62, 54)` - 1/8 scale
- **Architecture**: Three blocks with [3, 5, 5] layers each, stride 2 downsampling

### Neck

Feature fusion using transposed convolutions:
- **Input**: Multi-scale features from backbone
- **Output**: `(batch, 384, 248, 216)` fused features
- **Architecture**:
  - Upsamples each level to same resolution (248×216)
  - Concatenates features from all three levels
  - Output channels: [128, 128, 128] per level

### Head

Produces detection outputs:
- **Input**: `(batch, 384, 248, 216)` fused features
- **Output**: Three detection outputs:
  - **Classification**: `(batch, n_anchors*3, 248, 216)` - class logits for 3 classes
  - **Regression**: `(batch, n_anchors*7, 248, 216)` - bounding box parameters (x, y, z, w, l, h, yaw)
  - **Direction**: `(batch, n_anchors*2, 248, 216)` - direction classification (forward/backward)
- **Architecture**: Separate convolutional heads for each output

## Input/Output Format

### Input

- **Format**: LiDAR point cloud files (.bin format)
- **Preprocessing**:
  - Load point cloud: `(N, 4)` array with [x, y, z, intensity]
  - Filter points within range: [0, -39.68, -3, 69.12, 39.68, 1]
  - Voxelize into pillars: max 32 points per pillar
  - Encode pillar features: 9 features per point

### Output

The model returns three outputs:

1. **Classification Output**: `(batch, n_anchors*3, 248, 216)`
   - Class logits for 3 classes (Car, Pedestrian, Cyclist)
   - Apply softmax to get probabilities

2. **Regression Output**: `(batch, n_anchors*7, 248, 216)`
   - Bounding box parameters: [x, y, z, w, l, h, yaw]
   - Used to transform anchor boxes to predicted boxes

3. **Direction Output**: `(batch, n_anchors*2, 248, 216)`
   - Direction logits (forward/backward)
   - Used to resolve direction ambiguity in yaw angle

### Post-processing

To get final detections:
1. Generate anchor boxes for feature map locations
2. Transform anchors using regression outputs
3. Apply softmax to classification outputs
4. Filter detections by score threshold (default: 0.1)
5. Apply 3D Non-Maximum Suppression (NMS) with IoU threshold (default: 0.01)
6. Transform boxes to LiDAR or camera coordinates (if calibration provided)

### Model Statistics

- **Total Parameters**: ~4.7M (PointPillars)
- **Inference Speed**: Optimized for Tenstorrent hardware acceleration
- **Input Resolution**: Variable (point cloud), processed to 496×432 feature map
- **Output Resolution**: 248×216 feature map

## Troubleshooting

### Common Issues

1. **Weights not found**
   - Download weights from: [PointPillars Pretrained Weights](https://github.com/zhulf0804/PointPillars/blob/main/pretrained/epoch_160.pth)
   - Place `epoch_160.pth` in the working directory

2. **Device initialization errors**
   - Check device availability: `ttnn.list_devices()`
   - Verify device parameters (l1_small_size: 79104)

3. **Memory errors**
   - Reduce max_voxels parameter
   - Check available device memory
   - Ensure proper device configuration

4. **Point cloud file not found**
   - Verify point cloud file path is correct
   - Check file format (.bin)
   - Ensure point cloud has correct format: (N, 4) with [x, y, z, intensity]

5. **PCC below threshold**
   - Verify model weights are loaded correctly
   - Check input preprocessing matches reference
   - Ensure device is properly configured
   - Note: PCC threshold is typically set to 0.92 for component tests

## Implementation Notes

- **PillarLayer** and **PillarEncoder** indexing operations are performed on host (PyTorch) due to complex scatter operations
- **Backbone** and **Head** modules leverage the [`tt_cnn` builder pattern](../../tt_cnn/tt/builder.py) for efficient Conv2D operations
- **Neck** module uses custom transposed convolution implementations, as `ConvTranspose2D` operations are not currently available in the builder utilities
- The model processes point clouds in batches, with pillar encoding happening on-device after initial voxelization

## License

The reference implementation is based on [PointPillars PyTorch](https://github.com/zhulf0804/PointPillars) by zhulf0804, which is licensed under MIT. See the SPDX license headers in source files for details.
