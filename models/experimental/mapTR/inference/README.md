# MapTR Inference Pipeline

This directory contains the full inference pipeline for MapTR (Map Transformer) model.

## Quick Start

### 1. Download Sample Data

First, download sample data for testing:

```bash
cd /path/to/tt-metal

# Download sample images and create calibration
python models/experimental/mapTR/resources/download_data.py --sample

# (Optional) Download model weights
python models/experimental/mapTR/resources/download_data.py --weights

# List available downloads
python models/experimental/mapTR/resources/download_data.py --list
```

### 2. Demo Mode (No Weights Required)

Run inference with randomly initialized model and dummy data:

```bash
python models/experimental/mapTR/inference/run_inference.py --demo
```

### 3. Run with Sample Data

Run inference on the downloaded sample images:

```bash
python models/experimental/mapTR/inference/run_inference.py \
    --image_dir models/experimental/mapTR/resources/data/sample/images/ \
    --calibration models/experimental/mapTR/resources/data/sample/calibration.json \
    --output models/experimental/mapTR/resources/data/outputs/results.json
```

### With Real Images

Run inference on actual images:

```bash
python models/experimental/mapTR/inference/run_inference.py \
    --images cam_front.jpg cam_front_left.jpg cam_front_right.jpg \
             cam_back.jpg cam_back_left.jpg cam_back_right.jpg \
    --checkpoint /path/to/checkpoint.pth
```

Or from a directory:

```bash
python models/experimental/mapTR/inference/run_inference.py \
    --image_dir /path/to/images/ \
    --checkpoint /path/to/checkpoint.pth
```

### With Calibration

Provide camera calibration:

```bash
python models/experimental/mapTR/inference/run_inference.py \
    --images cam_front.jpg cam_front_left.jpg ... \
    --calibration /path/to/calibration.json \
    --checkpoint /path/to/checkpoint.pth
```

## Configuration

### Model Configurations

- `--config small`: Smaller model for testing (default)
  - Image: 224x400
  - BEV: 50x50
  - 20 vectors, 10 points per vector

- `--config nuScenes`: Full nuScenes configuration
  - Image: 900x1600
  - BEV: 200x100
  - 50 vectors, 20 points per vector

### Command Line Options

```
--checkpoint PATH    Path to model checkpoint (optional)
--images PATH...     Paths to input camera images
--image_dir PATH     Directory containing camera images
--calibration PATH   Path to calibration JSON file
--output PATH        Path to save results as JSON
--config {small,nuScenes}  Model configuration
--device {auto,cuda,cpu}   Device to use
--demo               Run demo with dummy data
```

## Checkpoint Loading

### From MapTR Official Checkpoints

The inference pipeline can load weights from official MapTR checkpoints:

```python
from models.experimental.mapTR.inference import MapTRInference, MapTRConfig

config = MapTRConfig.from_nuScenes()
inference = MapTRInference(
    config=config,
    checkpoint_path="/path/to/maptr_tiny_r50_24e.pth"
)
```

### Weight Conversion

If you have MMCV-format checkpoints, convert them first:

```python
from models.experimental.mapTR.inference.run_inference import convert_mmcv_checkpoint

convert_mmcv_checkpoint(
    checkpoint_path="/path/to/mmcv_checkpoint.pth",
    output_path="/path/to/pytorch_checkpoint.pth"
)
```

## Calibration Format

The calibration JSON file should have the following structure:

```json
{
    "lidar2img": [[...], [...], ...],
    "camera2ego": [[...], [...], ...],
    "camera_intrinsics": [[...], [...], ...],
    "img_aug_matrix": [[...], [...], ...],
    "lidar2ego": [[...], ...]
}
```

Each matrix should be a 4x4 transformation matrix. For multi-camera setups, provide arrays of shape (num_cameras, 4, 4).

## Programmatic Usage

```python
import torch
from pathlib import Path
from models.experimental.mapTR.inference import (
    MapTRConfig,
    MapTRInference,
    ImageProcessor,
)

# Create configuration
config = MapTRConfig(
    img_height=900,
    img_width=1600,
    num_cameras=6,
    num_classes=3,
    num_vec=50,
    num_pts_per_vec=20,
)

# Initialize inference pipeline
inference = MapTRInference(
    config=config,
    checkpoint_path="/path/to/checkpoint.pth",
)

# Option 1: Predict from image paths
results = inference.predict_from_paths([
    "cam_front.jpg",
    "cam_front_left.jpg",
    "cam_front_right.jpg",
    "cam_back.jpg",
    "cam_back_left.jpg",
    "cam_back_right.jpg",
])

# Option 2: Predict from tensor
image_processor = ImageProcessor(img_height=900, img_width=1600)
images = image_processor.generate_dummy_images(num_cameras=6)
calibration = inference.calibration.create_dummy_calibration(900, 1600)
img_metas = inference.create_img_metas(calibration)
results = inference.predict(images, img_metas)

# Format and print results
inference.print_results(results)
formatted = inference.format_results(results)
```

## Output Format

Results are returned as a list of dictionaries:

```python
{
    "num_detections": 50,
    "detections": [
        {
            "sample_idx": 0,
            "class": "divider",
            "class_id": 0,
            "score": 0.95,
            "bbox": [x1, y1, x2, y2],
            "points": [[x1, y1], [x2, y2], ...]
        },
        ...
    ]
}
```

## Classes

MapTR detects 3 classes of map elements:

| Class ID | Name | Description |
|----------|------|-------------|
| 0 | divider | Lane dividers |
| 1 | ped_crossing | Pedestrian crossings |
| 2 | boundary | Road boundaries |

## Notes

- The model expects 6 camera images in the nuScenes camera order:
  1. CAM_FRONT
  2. CAM_FRONT_LEFT
  3. CAM_FRONT_RIGHT
  4. CAM_BACK
  5. CAM_BACK_LEFT
  6. CAM_BACK_RIGHT

- For temporal fusion (video mode), use `video_test_mode=True` in the model configuration

- GPU acceleration is automatically used if available. Force CPU with `--device cpu`
