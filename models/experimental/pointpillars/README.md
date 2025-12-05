# 🧩 MobileNetV3

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![Supported Device](https://img.shields.io/badge/device-Wormhole%20(n150)-blue)
![Precision](https://img.shields.io/badge/precision-BF16%2FFP16-green)
![Input Resolution](https://img.shields.io/badge/input-224x224-lightgrey)
![Status](https://img.shields.io/badge/status-Stable-brightgreen)

---

## 🔍 Introduction

**PointPillars** are a real-time 3D object detection model designed for LiDAR point clouds, widely used in autonomous driving applications. It efficiently converts sparse 3D point clouds into a dense 2D pseudo-image representation using vertical pillars, enabling fast inference with standard 2D convolutional networks while maintaining high detection accuracy.

The model consists of a PillarEncoder that converts point cloud pillars into feature maps, a Backbone for multi-scale feature extraction, and a Neck that fuses features through upsampling. The Head produces final detection outputs including object classification, bounding box regression, and direction prediction.

---

## 📘 Overview

This implementation adapts **PointPillars** for **Tenstorrent hardware**, optimized for throughput and low-latency inference on **Wormhole** device.
The model is validated using internal test suites under `tests/`.

---

## :heavy_check_mark: Prerequisites
- Clone the **tt-metal** repository (source code & toolchains):
  <https://github.com/tenstorrent/tt-metal>
- Install **TT-Metalium™ / TT-NN™**:
  Follow the official instructions: <https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md>
- (Optional, for profiling) Build with profiler enabled:
  ```bash
  ./build_metal.sh --enable-profiler

---

## 🗂️ Repository Layout

| Directory | Purpose |
|------------|----------|
| `tt/` | Core Tenstorrent native modules of **Pointpillars** |
| `demo/` | Demo scripts and visualization |
| `resources/` | Sample images for testing |
| `tests/` | Validation(PCC) and Performance test scripts |



The `pointpillars/` directory plugs into this structure, exposing inference, profiling, and test utilities consistent with other models in the repo.

---

## 🚀 Quickstart: Run Pointpillars

### Set Up
#### Download Weights and data (Optional)
The demo currently uses saved weights from [reference model](https://github.com/zhulf0804/PointPillars/blob/main/pretrained/epoch_160.pth)
A sample of data (pointcloud,image,calibration data) needed for demo can be taken from [data](https://github.com/zhulf0804/PointPillars/tree/main/pointpillars/dataset/demo_data)

### Run Tests
```
models/experimental/pointpillars/tests/test_pointpillars.py
```
This runs an end-to-end flow that:

  - Loads the Torch reference from Torchvision,

  - Runs the TT-NN graph,

  - Post-processes outputs,

  - Optionally compares results and saves artifacts.

  - FPS ~ 31

### Run the Demo
```
python models/experimental/pointpillars/demo/demo.py \
        --pc_path  <path/to/pointcloud.bin> \
        --calib_path  <path/to/calibdata.txt> \
        --img_path  <path/to/image.png>
```
### Custom Images
You can place your image(s),pointcloud file under:
```
models/experimental/mobileNetV3/resources/
```
Then re-run either the demo:

Expected output:
```
Demo completed successfully!
Predicted classification label overlaid and image/s saved in output directory[resources/output/]
```

## Note
- PointPillars consists of five main stages:

| Stage | Description | Execution |
|-------|-------------|-----------|
| **PillarLayer** | Point cloud voxelization - converts raw 3D points into pillar representation | PyTorch (CPU/GPU) |
| **PillarEncoder** | Encodes pillar features using Conv1D + BatchNorm + ReLU + MaxPool | **TTNN** |
| **Backbone** | Multi-scale feature extraction with 2D convolutions | **TTNN** |
| **Neck** | Feature fusion using transposed convolutions for upsampling | **TTNN** |
| **Head** | Detection outputs (classification, regression, direction) | **TTNN** |

- The **Backbone** and **Head** modules leverage the [`tt_cnn` builder pattern](../../tt_cnn/tt/builder.py) for efficient Conv2D operations. The **Neck** module uses custom transposed convolution implementations, as `ConvTranspose2D` operations are not currently available in the builder utilities.

---

## 🔗 References

- [PyTorch Pointpillars ](https://github.com/zhulf0804/PointPillars/tree/main)
- [Tenstorrent Developer SDK Docs](https://tenstorrent.com/developer-docs)
- [TTNN API Reference](../../docs/ttnn_reference.md)

---
