# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import List, Optional
import numpy as np


@dataclass
class DemoConfig:
    """Configuration class for demo parameters"""

    # Model configuration
    model_type: str = "PanopticDeepLab"
    backbone: str = "ResNet-52"
    num_classes: int = 19
    weights_path: Optional[str] = None

    # Input configuration
    input_height: int = 512
    input_width: int = 1024
    crop_enabled: bool = False
    normalize_enabled: bool = True
    mean: List[float] = None
    std: List[float] = None

    # thing instances
    center_threshold: float = 0.01
    nms_kernel: int = 7
    top_k_instances: int = 400
    instance_score_threshold: float = 0.1

    # stuff classes
    stuff_area_threshold: int = 100
    min_stuff_area: int = 100

    # For all detections
    label_divisor: int = 256
    min_instance_area: int = 300
    max_distance: int = 150

    # Device configuration
    device_id: int = 0
    math_fidelity: str = "LoFi"
    weights_dtype: str = "bfloat8_b"
    activations_dtype: str = "bfloat8_b"

    # Output configuration
    save_results: bool = True
    save_comparison: bool = False  # for different heads visualization
    dual_pipeline: bool = False  # for saving results in different pipelines

    # Dataset configuration (Cityscapes default)
    thing_classes: List[int] = None
    stuff_classes: List[int] = None
    class_names: List[str] = None

    def __post_init__(self):
        """Initialize default values after dataclass creation"""
        if self.mean is None:
            self.mean = [0.485, 0.456, 0.406]
        if self.std is None:
            self.std = [0.229, 0.224, 0.225]
        if self.thing_classes is None:
            self.thing_classes = [11, 12, 13, 14, 15, 16, 17, 18]  # Cityscapes things
        if self.stuff_classes is None:
            self.stuff_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # Cityscapes stuff
        if self.class_names is None:
            self.class_names = [
                "road",
                "sidewalk",
                "building",
                "wall",
                "fence",
                "pole",
                "traffic_light",
                "traffic_sign",
                "vegetation",
                "terrain",
                "sky",
                "person",
                "rider",
                "car",
                "truck",
                "bus",
                "train",
                "motorcycle",
                "bicycle",
            ]

    def _get_cityscapes_colors(self) -> np.ndarray:
        """Get Cityscapes color palette with enhanced visibility"""
        return np.array(
            [
                [128, 64, 128],  # road (stuff)
                [244, 35, 232],  # sidewalk (stuff)
                [70, 70, 70],  # building (stuff)
                [102, 102, 156],  # wall
                [190, 153, 153],  # fence
                [153, 153, 153],  # pole
                [250, 170, 30],  # traffic light
                [220, 220, 0],  # traffic sign
                [107, 142, 35],  # vegetation
                [152, 251, 152],  # terrain
                [70, 130, 180],  # sky
                [220, 20, 60],  # person (thing)
                [255, 0, 0],  # rider (thing)
                [0, 0, 142],  # car (thing)
                [0, 0, 70],  # truck (thing)
                [0, 60, 100],  # bus (thing)
                [0, 80, 100],  # train (thing)
                [0, 0, 230],  # motorcycle (thing)
                [119, 11, 32],  # bicycle (thing)
            ],
            dtype=np.uint8,
        )
