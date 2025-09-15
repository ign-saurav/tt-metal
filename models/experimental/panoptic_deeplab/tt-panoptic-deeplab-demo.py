# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import argparse
import time
import yaml
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from loguru import logger
import json
from dataclasses import dataclass, asdict
import pickle
import torch
import torchvision.transforms as transforms
import ttnn
from models.experimental.panoptic_deeplab.tt.panoptic_deeplab import TTPanopticDeepLab
from models.experimental.panoptic_deeplab.tt.custom_preprocessing import create_custom_mesh_preprocessor
from models.experimental.panoptic_deeplab.reference.panoptic_deeplab import TorchPanopticDeepLab
from post_processing import PostProcessing


def map_single_key(checkpoint_key):
    """
    Map checkpoint keys to model keys.
    """

    if not checkpoint_key:
        return ""

    key = checkpoint_key

    # BACKBONE MAPPINGS
    if key.startswith("backbone."):
        # Stem batch norm mappings
        key = key.replace("backbone.stem.conv1.norm.", "backbone.stem.bn1.")
        key = key.replace("backbone.stem.conv2.norm.", "backbone.stem.bn2.")
        key = key.replace("backbone.stem.conv3.norm.", "backbone.stem.bn3.")

        # Layer mapping: res2/3/4/5 -> layer1/2/3/4
        key = key.replace("backbone.res2.", "backbone.layer1.")
        key = key.replace("backbone.res3.", "backbone.layer2.")
        key = key.replace("backbone.res4.", "backbone.layer3.")
        key = key.replace("backbone.res5.", "backbone.layer4.")

        # Batch norm mapping: conv1/2/3.norm -> bn1/2/3
        key = key.replace(".conv1.norm.", ".bn1.")
        key = key.replace(".conv2.norm.", ".bn2.")
        key = key.replace(".conv3.norm.", ".bn3.")

        # Downsample mapping: shortcut -> downsample
        key = key.replace(".shortcut.norm.", ".downsample.1.")
        # Handle shortcut.weight
        if ".shortcut." in key and ".shortcut.norm." not in checkpoint_key:
            key = key.replace(".shortcut.", ".downsample.0.")

        return key

    # SEMANTIC HEAD MAPPINGS
    elif key.startswith("sem_seg_head."):
        # Replace base prefix
        key = key.replace("sem_seg_head.", "semantic_decoder.")

        # Head mappings
        if ".predictor." in key:
            key = key.replace(".predictor.", ".head_1.conv3.0.")
        elif ".head.pointwise." in key:
            if ".head.pointwise.norm." in key:
                key = key.replace(".head.pointwise.norm.", ".head_1.conv2.1.")
            else:
                key = key.replace(".head.pointwise.", ".head_1.conv2.0.")
        elif ".head.depthwise." in key:
            if ".head.depthwise.norm." in key:
                key = key.replace(".head.depthwise.norm.", ".head_1.conv1.1.")
            else:
                key = key.replace(".head.depthwise.", ".head_1.conv1.0.")

        # ASPP mappings (res5 -> aspp)
        elif ".decoder.res5.project_conv." in key:
            # Special case for ASPP_3_Depthwise
            if ".convs.3.depthwise.norm." in key:
                key = key.replace(".decoder.res5.project_conv.convs.3.depthwise.norm.", ".aspp.ASPP_3_Depthwise.1.")
            elif ".convs.3.depthwise." in key:
                key = key.replace(".decoder.res5.project_conv.convs.3.depthwise.", ".aspp.ASPP_3_Depthwise.0.")

            # ASPP_0_Conv
            elif ".convs.0.norm." in key:
                key = key.replace(".decoder.res5.project_conv.convs.0.norm.", ".aspp.ASPP_0_Conv.1.")
            elif ".convs.0." in key:
                key = key.replace(".decoder.res5.project_conv.convs.0.", ".aspp.ASPP_0_Conv.0.")

            # ASPP_1 Depthwise and Pointwise
            elif ".convs.1.depthwise.norm." in key:
                key = key.replace(".decoder.res5.project_conv.convs.1.depthwise.norm.", ".aspp.ASPP_1_Depthwise.1.")
            elif ".convs.1.depthwise." in key:
                key = key.replace(".decoder.res5.project_conv.convs.1.depthwise.", ".aspp.ASPP_1_Depthwise.0.")
            elif ".convs.1.pointwise.norm." in key:
                key = key.replace(".decoder.res5.project_conv.convs.1.pointwise.norm.", ".aspp.ASPP_1_pointwise.1.")
            elif ".convs.1.pointwise." in key:
                key = key.replace(".decoder.res5.project_conv.convs.1.pointwise.", ".aspp.ASPP_1_pointwise.0.")

            # ASPP_2 Depthwise and Pointwise
            elif ".convs.2.depthwise.norm." in key:
                key = key.replace(".decoder.res5.project_conv.convs.2.depthwise.norm.", ".aspp.ASPP_2_Depthwise.1.")
            elif ".convs.2.depthwise." in key:
                key = key.replace(".decoder.res5.project_conv.convs.2.depthwise.", ".aspp.ASPP_2_Depthwise.0.")
            elif ".convs.2.pointwise.norm." in key:
                key = key.replace(".decoder.res5.project_conv.convs.2.pointwise.norm.", ".aspp.ASPP_2_pointwise.1.")
            elif ".convs.2.pointwise." in key:
                key = key.replace(".decoder.res5.project_conv.convs.2.pointwise.", ".aspp.ASPP_2_pointwise.0.")

            # ASPP_3 Pointwise
            elif ".convs.3.pointwise.norm." in key:
                key = key.replace(".decoder.res5.project_conv.convs.3.pointwise.norm.", ".aspp.ASPP_3_pointwise.1.")
            elif ".convs.3.pointwise." in key:
                key = key.replace(".decoder.res5.project_conv.convs.3.pointwise.", ".aspp.ASPP_3_pointwise.0.")

            # ASPP_4_Conv
            elif ".convs.4." in key:
                key = key.replace(".decoder.res5.project_conv.convs.4.1.", ".aspp.ASPP_4_Conv_1.0.")

            # ASPP project
            elif ".project.norm." in key:
                key = key.replace(".decoder.res5.project_conv.project.norm.", ".aspp.ASPP_project.1.")
            elif ".project." in key:
                key = key.replace(".decoder.res5.project_conv.project.", ".aspp.ASPP_project.0.")

        # Decoder res3 mappings
        elif ".decoder.res3." in key:
            if ".project_conv.norm." in key:
                key = key.replace(".decoder.res3.project_conv.norm.", ".res3.conv1.1.")
            elif ".project_conv." in key:
                key = key.replace(".decoder.res3.project_conv.", ".res3.conv1.0.")
            elif ".fuse_conv.depthwise.norm." in key:
                key = key.replace(".decoder.res3.fuse_conv.depthwise.norm.", ".res3.conv2.1.")
            elif ".fuse_conv.depthwise." in key:
                key = key.replace(".decoder.res3.fuse_conv.depthwise.", ".res3.conv2.0.")
            elif ".fuse_conv.pointwise.norm." in key:
                key = key.replace(".decoder.res3.fuse_conv.pointwise.norm.", ".res3.conv3.1.")
            elif ".fuse_conv.pointwise." in key:
                key = key.replace(".decoder.res3.fuse_conv.pointwise.", ".res3.conv3.0.")

        # Decoder res2 mappings
        elif ".decoder.res2." in key:
            if ".project_conv.norm." in key:
                key = key.replace(".decoder.res2.project_conv.norm.", ".res2.conv1.1.")
            elif ".project_conv." in key:
                key = key.replace(".decoder.res2.project_conv.", ".res2.conv1.0.")
            elif ".fuse_conv.depthwise.norm." in key:
                key = key.replace(".decoder.res2.fuse_conv.depthwise.norm.", ".res2.conv2.1.")
            elif ".fuse_conv.depthwise." in key:
                key = key.replace(".decoder.res2.fuse_conv.depthwise.", ".res2.conv2.0.")
            elif ".fuse_conv.pointwise.norm." in key:
                key = key.replace(".decoder.res2.fuse_conv.pointwise.norm.", ".res2.conv3.1.")
            elif ".fuse_conv.pointwise." in key:
                key = key.replace(".decoder.res2.fuse_conv.pointwise.", ".res2.conv3.0.")

        return key

    # INSTANCE HEAD MAPPINGS
    elif key.startswith("ins_embed_head."):
        # Replace base prefix
        key = key.replace("ins_embed_head.", "instance_decoder.")

        # Center head mappings
        if ".center_head.0.norm." in key:
            key = key.replace(".center_head.0.norm.", ".head_2.conv1.1.")
        elif ".center_head.0." in key:
            key = key.replace(".center_head.0.", ".head_2.conv1.0.")
        elif ".center_head.1.norm." in key:
            key = key.replace(".center_head.1.norm.", ".head_2.conv2.1.")
        elif ".center_head.1." in key:
            key = key.replace(".center_head.1.", ".head_2.conv2.0.")
        elif ".center_predictor." in key:
            key = key.replace(".center_predictor.", ".head_2.conv3.0.")

        # Offset head mappings
        elif ".offset_head.depthwise.norm." in key:
            key = key.replace(".offset_head.depthwise.norm.", ".head_1.conv1.1.")
        elif ".offset_head.depthwise." in key:
            key = key.replace(".offset_head.depthwise.", ".head_1.conv1.0.")
        elif ".offset_head.pointwise.norm." in key:
            key = key.replace(".offset_head.pointwise.norm.", ".head_1.conv2.1.")
        elif ".offset_head.pointwise." in key:
            key = key.replace(".offset_head.pointwise.", ".head_1.conv2.0.")
        elif ".offset_predictor." in key:
            key = key.replace(".offset_predictor.", ".head_1.conv3.0.")

        # ASPP mappings (res5 -> aspp)
        elif ".decoder.res5.project_conv." in key:
            # Special case for ASPP_3_Depthwise
            if ".convs.3.depthwise.norm." in key:
                key = key.replace(".decoder.res5.project_conv.convs.3.depthwise.norm.", ".aspp.ASPP_3_Depthwise.1.")
            elif ".convs.3.depthwise." in key:
                key = key.replace(".decoder.res5.project_conv.convs.3.depthwise.", ".aspp.ASPP_3_Depthwise.0.")

            # ASPP_0_Conv
            elif ".convs.0.norm." in key:
                key = key.replace(".decoder.res5.project_conv.convs.0.norm.", ".aspp.ASPP_0_Conv.1.")
            elif ".convs.0." in key:
                key = key.replace(".decoder.res5.project_conv.convs.0.", ".aspp.ASPP_0_Conv.0.")

            # ASPP_1 Depthwise and Pointwise
            elif ".convs.1.depthwise.norm." in key:
                key = key.replace(".decoder.res5.project_conv.convs.1.depthwise.norm.", ".aspp.ASPP_1_Depthwise.1.")
            elif ".convs.1.depthwise." in key:
                key = key.replace(".decoder.res5.project_conv.convs.1.depthwise.", ".aspp.ASPP_1_Depthwise.0.")
            elif ".convs.1.pointwise.norm." in key:
                key = key.replace(".decoder.res5.project_conv.convs.1.pointwise.norm.", ".aspp.ASPP_1_pointwise.1.")
            elif ".convs.1.pointwise." in key:
                key = key.replace(".decoder.res5.project_conv.convs.1.pointwise.", ".aspp.ASPP_1_pointwise.0.")

            # ASPP_2 Depthwise and Pointwise
            elif ".convs.2.depthwise.norm." in key:
                key = key.replace(".decoder.res5.project_conv.convs.2.depthwise.norm.", ".aspp.ASPP_2_Depthwise.1.")
            elif ".convs.2.depthwise." in key:
                key = key.replace(".decoder.res5.project_conv.convs.2.depthwise.", ".aspp.ASPP_2_Depthwise.0.")
            elif ".convs.2.pointwise.norm." in key:
                key = key.replace(".decoder.res5.project_conv.convs.2.pointwise.norm.", ".aspp.ASPP_2_pointwise.1.")
            elif ".convs.2.pointwise." in key:
                key = key.replace(".decoder.res5.project_conv.convs.2.pointwise.", ".aspp.ASPP_2_pointwise.0.")

            # ASPP_3 Pointwise
            elif ".convs.3.pointwise.norm." in key:
                key = key.replace(".decoder.res5.project_conv.convs.3.pointwise.norm.", ".aspp.ASPP_3_pointwise.1.")
            elif ".convs.3.pointwise." in key:
                key = key.replace(".decoder.res5.project_conv.convs.3.pointwise.", ".aspp.ASPP_3_pointwise.0.")

            # ASPP_4_Conv
            elif ".convs.4." in key:
                key = key.replace(".decoder.res5.project_conv.convs.4.1.", ".aspp.ASPP_4_Conv_1.0.")

            # ASPP project
            elif ".project.norm." in key:
                key = key.replace(".decoder.res5.project_conv.project.norm.", ".aspp.ASPP_project.1.")
            elif ".project." in key:
                key = key.replace(".decoder.res5.project_conv.project.", ".aspp.ASPP_project.0.")

        # Decoder res3 mappings
        elif ".decoder.res3." in key:
            if ".project_conv.norm." in key:
                key = key.replace(".decoder.res3.project_conv.norm.", ".res3.conv1.1.")
            elif ".project_conv." in key:
                key = key.replace(".decoder.res3.project_conv.", ".res3.conv1.0.")
            elif ".fuse_conv.depthwise.norm." in key:
                key = key.replace(".decoder.res3.fuse_conv.depthwise.norm.", ".res3.conv2.1.")
            elif ".fuse_conv.depthwise." in key:
                key = key.replace(".decoder.res3.fuse_conv.depthwise.", ".res3.conv2.0.")
            elif ".fuse_conv.pointwise.norm." in key:
                key = key.replace(".decoder.res3.fuse_conv.pointwise.norm.", ".res3.conv3.1.")
            elif ".fuse_conv.pointwise." in key:
                key = key.replace(".decoder.res3.fuse_conv.pointwise.", ".res3.conv3.0.")

        # Decoder res2 mappings
        elif ".decoder.res2." in key:
            if ".project_conv.norm." in key:
                key = key.replace(".decoder.res2.project_conv.norm.", ".res2.conv1.1.")
            elif ".project_conv." in key:
                key = key.replace(".decoder.res2.project_conv.", ".res2.conv1.0.")
            elif ".fuse_conv.depthwise.norm." in key:
                key = key.replace(".decoder.res2.fuse_conv.depthwise.norm.", ".res2.conv2.1.")
            elif ".fuse_conv.depthwise." in key:
                key = key.replace(".decoder.res2.fuse_conv.depthwise.", ".res2.conv2.0.")
            elif ".fuse_conv.pointwise.norm." in key:
                key = key.replace(".decoder.res2.fuse_conv.pointwise.norm.", ".res2.conv3.1.")
            elif ".fuse_conv.pointwise." in key:
                key = key.replace(".decoder.res2.fuse_conv.pointwise.", ".res2.conv3.0.")

        return key

    return ""


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
    show_instance_labels: bool = True

    # Inference configuration
    center_threshold: float = 0.1
    nms_kernel: int = 7
    top_k_instances: int = 200
    stuff_area_threshold: int = 2048

    # Device configuration
    device_id: int = 0
    math_fidelity: str = "HiFi4"
    weights_dtype: str = "bfloat16"
    activations_dtype: str = "bfloat16"

    # Output configuration
    save_semantic: bool = True
    save_instance: bool = True
    save_panoptic: bool = True
    save_visualization: bool = True
    save_comparison: bool = True

    # Pipeline configuration
    run_torch_pipeline: bool = True
    run_ttnn_pipeline: bool = True
    compare_outputs: bool = True
    pcc_threshold: float = 0.97

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

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "DemoConfig":
        """Load configuration from YAML file"""
        with open(yaml_path, "r") as f:
            config_dict = yaml.safe_load(f)

        # Flatten nested config structure to match dataclass fields
        flattened = {}

        # Model section
        if "MODEL" in config_dict:
            model_cfg = config_dict["MODEL"]
            flattened.update(
                {
                    "model_type": model_cfg.get("TYPE", "PanopticDeepLab"),
                    "backbone": model_cfg.get("BACKBONE", "ResNet-52"),
                    "num_classes": model_cfg.get("NUM_CLASSES", 19),
                    "weights_path": model_cfg.get("WEIGHTS", None),
                }
            )

        # Input section
        if "INPUT" in config_dict:
            input_cfg = config_dict["INPUT"]
            if "SIZE_TRAIN" in input_cfg or "SIZE_TEST" in input_cfg:
                size = input_cfg.get("SIZE_TEST", input_cfg.get("SIZE_TRAIN", [1024, 512]))
                flattened.update(
                    {
                        "input_height": size[0],
                        "input_width": size[1],
                    }
                )
            flattened.update(
                {
                    "crop_enabled": input_cfg.get("CROP", {}).get("ENABLED", False),
                    "normalize_enabled": input_cfg.get("NORMALIZE", True),
                    "mean": input_cfg.get("PIXEL_MEAN", [0.485, 0.456, 0.406]),
                    "std": input_cfg.get("PIXEL_STD", [0.229, 0.224, 0.225]),
                }
            )

        # Postprocessing section
        if "POST_PROCESSING" in config_dict:
            pp_cfg = config_dict["POST_PROCESSING"]
            flattened.update(
                {
                    "center_threshold": pp_cfg.get("CENTER_THRESHOLD", 0.1),
                    "nms_kernel": pp_cfg.get("NMS_KERNEL", 7),
                    "top_k_instances": pp_cfg.get("TOP_K_INSTANCES", 200),
                    "stuff_area_threshold": pp_cfg.get("STUFF_AREA_THRESHOLD", 2048),
                }
            )

        # Device section
        if "DEVICE" in config_dict:
            device_cfg = config_dict["DEVICE"]
            flattened.update(
                {
                    "device_id": device_cfg.get("ID", 0),
                    "math_fidelity": device_cfg.get("MATH_FIDELITY", "HiFi4"),
                    "weights_dtype": device_cfg.get("WEIGHTS_DTYPE", "bfloat16"),
                    "activations_dtype": device_cfg.get("ACTIVATIONS_DTYPE", "bfloat16"),
                }
            )

        # Demo-specific sections
        if "DEMO" in config_dict:
            demo_cfg = config_dict["DEMO"]
            flattened.update(
                {
                    "run_torch_pipeline": demo_cfg.get("RUN_TORCH", True),
                    "run_ttnn_pipeline": demo_cfg.get("RUN_TTNN", True),
                    "compare_outputs": demo_cfg.get("COMPARE_OUTPUTS", True),
                    "pcc_threshold": demo_cfg.get("PCC_THRESHOLD", 0.95),
                }
            )

        # Classes section
        if "CLASSES" in config_dict:
            classes_cfg = config_dict["CLASSES"]
            flattened.update(
                {
                    "thing_classes": classes_cfg.get("THING", [11, 12, 13, 14, 15, 16, 17, 18]),
                    "stuff_classes": classes_cfg.get("STUFF", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
                    "class_names": classes_cfg.get("NAMES", None),
                }
            )

        return cls(**flattened)

    def to_yaml(self, yaml_path: str):
        """Save configuration to YAML file"""
        config_dict = {
            "MODEL": {
                "TYPE": self.model_type,
                "BACKBONE": self.backbone,
                "NUM_CLASSES": self.num_classes,
                "WEIGHTS": self.weights_path,
            },
            "INPUT": {
                "SIZE_TEST": [self.input_height, self.input_width],
                "CROP": {"ENABLED": self.crop_enabled},
                "NORMALIZE": self.normalize_enabled,
                "PIXEL_MEAN": self.mean,
                "PIXEL_STD": self.std,
            },
            "POST_PROCESSING": {
                "CENTER_THRESHOLD": self.center_threshold,
                "NMS_KERNEL": self.nms_kernel,
                "TOP_K_INSTANCES": self.top_k_instances,
                "STUFF_AREA_THRESHOLD": self.stuff_area_threshold,
            },
            "DEVICE": {
                "ID": self.device_id,
                "MATH_FIDELITY": self.math_fidelity,
                "WEIGHTS_DTYPE": self.weights_dtype,
                "ACTIVATIONS_DTYPE": self.activations_dtype,
            },
            "DEMO": {
                "RUN_TORCH": self.run_torch_pipeline,
                "RUN_TTNN": self.run_ttnn_pipeline,
                "COMPARE_OUTPUTS": self.compare_outputs,
                "PCC_THRESHOLD": self.pcc_threshold,
            },
            "CLASSES": {
                "THING": self.thing_classes,
                "STUFF": self.stuff_classes,
                "NAMES": self.class_names,
            },
        }

        os.makedirs(os.path.dirname(yaml_path), exist_ok=True)
        with open(yaml_path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)

        logger.info(f"Configuration saved to: {yaml_path}")


class DualPipelineDemo:
    """Enhanced demo supporting both PyTorch and TTNN pipelines with comparison"""

    def __init__(self, config: DemoConfig):
        self.config = config
        self.torch_model = None
        self.ttnn_model = None
        self.ttnn_device = None

        self.preprocess = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=config.mean, std=config.std)
                if config.normalize_enabled
                else transforms.Lambda(lambda x: x),
            ]
        )

        # Color palette for visualization
        self.colors = self._get_cityscapes_colors()

        # Mesh mappers for TTNN
        self.inputs_mesh_mapper = None
        self.weights_mesh_mapper = None
        self.output_mesh_composer = None

    def _get_cityscapes_colors(self) -> np.ndarray:
        """Get Cityscapes color palette"""
        return np.array(
            [
                [128, 64, 128],  # road
                [244, 35, 232],  # sidewalk
                [70, 70, 70],  # building
                [102, 102, 156],  # wall
                [190, 153, 153],  # fence
                [153, 153, 153],  # pole
                [250, 170, 30],  # traffic light
                [220, 220, 0],  # traffic sign
                [107, 142, 35],  # vegetation
                [152, 251, 152],  # terrain
                [70, 130, 180],  # sky
                [220, 20, 60],  # person
                [255, 0, 0],  # rider
                [0, 0, 142],  # car
                [0, 0, 70],  # truck
                [0, 60, 100],  # bus
                [0, 80, 100],  # train
                [0, 0, 230],  # motorcycle
                [119, 11, 32],  # bicycle
            ]
        )

    def initialize_torch_model(self):
        """Initialize PyTorch model"""
        if not self.config.run_torch_pipeline:
            return

        logger.info("Initializing PyTorch Panoptic DeepLab model...")

        self.torch_model = TorchPanopticDeepLab().eval()

        # Load weights if provided
        if self.config.weights_path and os.path.exists(self.config.weights_path):
            logger.info(f"Loading PyTorch weights from: {self.config.weights_path}")

            # Test manual mapping first
            logger.debug(f"Testing manual mapping with: {self.config.weights_path}")

            try:
                # Load checkpoint
                with open(self.config.weights_path, "rb") as f:
                    checkpoint = pickle.load(f, encoding="latin1")

                # Get state dict
                if "model_state_dict" in checkpoint:
                    state_dict = checkpoint["model_state_dict"]
                    logger.info("Using 'model_state_dict' key")
                elif "model" in checkpoint:
                    state_dict = checkpoint["model"]
                    logger.info("Using 'model' key")
                else:
                    state_dict = checkpoint
                    logger.info("Using checkpoint directly as state dict")

                # Convert numpy arrays to torch tensors
                converted_count = 0
                for k, v in state_dict.items():
                    if isinstance(v, np.ndarray):
                        state_dict[k] = torch.from_numpy(v)
                        converted_count += 1
                logger.debug(f"Converted {converted_count} numpy arrays to torch tensors")

                # Get model keys
                model_dict = self.torch_model.state_dict()
                model_keys = set(model_dict.keys())
                checkpoint_keys = set(state_dict.keys())
                # Create comprehensive mapping
                logger.info("Creating comprehensive key mapping...")

                logger.info("Mapping keys...")
                key_mapping = {}
                for checkpoint_key in checkpoint_keys:  # pickle key
                    mapped_key = map_single_key(checkpoint_key)
                    if mapped_key in model_keys:  # torch keys
                        key_mapping[checkpoint_key] = mapped_key
                    else:
                        logger.debug(f"No mapping for mapped key: {mapped_key} (checkpoint key: {checkpoint_key})")

                logger.info(f"Model_keys - {len(model_keys)} , checkpoint_keys - {len(checkpoint_keys)}")

                # Apply mappings
                mapped_state_dict = {}
                for checkpoint_key, model_key in key_mapping.items():
                    mapped_state_dict[model_key] = state_dict[checkpoint_key]

                # Try loading
                try:
                    self.torch_model.load_state_dict(mapped_state_dict, strict=True)
                    logger.info(f"Successfully loaded all {len(mapped_state_dict)} mapped weights with strict=True")

                except RuntimeError as e:
                    logger.warning(f"Strict loading failed:")
                    logger.info("Attempting partial loading...")

                    # Partial loading
                    loaded_keys = []
                    skipped_keys = []

                    for model_key, checkpoint_tensor in mapped_state_dict.items():
                        if model_key in model_dict:
                            if model_dict[model_key].shape == checkpoint_tensor.shape:
                                model_dict[model_key] = checkpoint_tensor
                                loaded_keys.append(model_key)
                            else:
                                skipped_keys.append(f"{model_key}: shape mismatch")
                        else:
                            skipped_keys.append(f"{model_key}: not found in model")

                    # Load the updated model dict
                    self.torch_model.load_state_dict(model_dict)

                    total_model_params = len(model_dict)
                    load_ratio = len(loaded_keys) / total_model_params

                    logger.info(f"Loaded {len(loaded_keys)}/{total_model_params} model parameters ({load_ratio:.1%})")

                    if skipped_keys:
                        logger.warning(f"Skipped {len(skipped_keys)} incompatible weights (showing first 10):")
                        for skip_msg in skipped_keys[:10]:
                            logger.warning(f"  - {skip_msg}")

                    if load_ratio >= 0.7:
                        logger.info(f"Successfully loaded {load_ratio:.1%} of model weights - excellent coverage!")
                    elif load_ratio >= 0.5:
                        logger.warning(f"Loaded {load_ratio:.1%} of model weights - decent coverage")
                    else:
                        logger.error(f"Only loaded {load_ratio:.1%} of model weights - poor coverage")

                # Print sample weight values for a few loaded checkpoint keys and their mapped model keys
                logger.info("Sample loaded weights (checkpoint key -> model key):")

                # Verify sample parameters were updated
                sample_params = list(self.torch_model.parameters())[:3]
                if all(torch.any(p != 0) for p in sample_params):
                    logger.info("Weight verification passed - parameters contain non-zero values")
                else:
                    logger.warning("Weight verification failed - found zero parameters")

            except Exception as e:
                logger.error(f"Failed to load weights file: {str(e)}")
                logger.warning("Falling back to random initialization")

        else:
            logger.warning("No weights provided - using random initialization")

        logger.info("PyTorch model initialization completed")
        logger.info("PyTorch model initialized")

    def initialize_ttnn_model(self):
        """Initialize TTNN model"""
        if not self.config.run_ttnn_pipeline:
            return

        logger.info("Initializing TTNN Panoptic DeepLab model...")

        # Initialize TT device
        self.ttnn_device = ttnn.open_device(device_id=self.config.device_id, l1_small_size=16384)

        # Setup mesh mappers
        self._setup_mesh_mappers()

        # Create reference model for parameter extraction
        if self.torch_model is not None:
            reference_model = self.torch_model
        else:
            reference_model = TorchPanopticDeepLab().eval()

            if self.config.weights_path and os.path.exists(self.config.weights_path):
                logger.info(f"Loading PyTorch weights from: {self.config.weights_path}")

                # Test manual mapping first
                logger.debug(f"Testing manual mapping with: {self.config.weights_path}")

                try:
                    # Load checkpoint
                    with open(self.config.weights_path, "rb") as f:
                        checkpoint = pickle.load(f, encoding="latin1")

                    # Get state dict
                    if "model_state_dict" in checkpoint:
                        state_dict = checkpoint["model_state_dict"]
                        logger.info("Using 'model_state_dict' key")
                    elif "model" in checkpoint:
                        state_dict = checkpoint["model"]
                        logger.info("Using 'model' key")
                    else:
                        state_dict = checkpoint
                        logger.info("Using checkpoint directly as state dict")

                    # Convert numpy arrays to torch tensors
                    converted_count = 0
                    for k, v in state_dict.items():
                        if isinstance(v, np.ndarray):
                            state_dict[k] = torch.from_numpy(v)
                            converted_count += 1
                    logger.debug(f"Converted {converted_count} numpy arrays to torch tensors")

                    # Get model keys
                    model_dict = self.torch_model.state_dict()
                    model_keys = set(model_dict.keys())
                    checkpoint_keys = set(state_dict.keys())
                    # Create comprehensive mapping
                    logger.info("Creating comprehensive key mapping...")

                    logger.info("Mapping keys...")
                    key_mapping = {}
                    for checkpoint_key in checkpoint_keys:  # pickle key
                        mapped_key = map_single_key(checkpoint_key)
                        if mapped_key in model_keys:  # torch keys
                            key_mapping[checkpoint_key] = mapped_key
                        else:
                            logger.debug(f"No mapping for mapped key: {mapped_key} (checkpoint key: {checkpoint_key})")

                    logger.info(f"Model_keys - {len(model_keys)} , checkpoint_keys - {len(checkpoint_keys)}")

                    # Apply mappings
                    mapped_state_dict = {}
                    for checkpoint_key, model_key in key_mapping.items():
                        mapped_state_dict[model_key] = state_dict[checkpoint_key]

                    # Try loading
                    try:
                        self.torch_model.load_state_dict(mapped_state_dict, strict=True)
                        logger.info(f"Successfully loaded all {len(mapped_state_dict)} mapped weights with strict=True")

                    except RuntimeError as e:
                        logger.warning(f"Strict loading failed:")
                        logger.info("Attempting partial loading...")

                        # Partial loading
                        loaded_keys = []
                        skipped_keys = []

                        for model_key, checkpoint_tensor in mapped_state_dict.items():
                            if model_key in model_dict:
                                if model_dict[model_key].shape == checkpoint_tensor.shape:
                                    model_dict[model_key] = checkpoint_tensor
                                    loaded_keys.append(model_key)
                                else:
                                    skipped_keys.append(f"{model_key}: shape mismatch")
                            else:
                                skipped_keys.append(f"{model_key}: not found in model")

                        # Load the updated model dict
                        self.torch_model.load_state_dict(model_dict)

                        total_model_params = len(model_dict)
                        load_ratio = len(loaded_keys) / total_model_params

                        logger.info(
                            f"Loaded {len(loaded_keys)}/{total_model_params} model parameters ({load_ratio:.1%})"
                        )

                        if skipped_keys:
                            logger.warning(f"Skipped {len(skipped_keys)} incompatible weights (showing first 10):")
                            for skip_msg in skipped_keys[:10]:
                                logger.warning(f"  - {skip_msg}")

                        if load_ratio >= 0.7:
                            logger.info(f"Successfully loaded {load_ratio:.1%} of model weights - excellent coverage!")
                        elif load_ratio >= 0.5:
                            logger.warning(f"Loaded {load_ratio:.1%} of model weights - decent coverage")
                        else:
                            logger.error(f"Only loaded {load_ratio:.1%} of model weights - poor coverage")

                    # Print sample weight values for a few loaded checkpoint keys and their mapped model keys
                    logger.info("Sample loaded weights (checkpoint key -> model key):")

                    # Verify sample parameters were updated
                    sample_params = list(self.torch_model.parameters())[:3]
                    if all(torch.any(p != 0) for p in sample_params):
                        logger.info("Weight verification passed - parameters contain non-zero values")
                    else:
                        logger.warning("Weight verification failed - found zero parameters")

                except Exception as e:
                    logger.error(f"Failed to load weights file: {str(e)}")
                    logger.warning("Falling back to random initialization")

            else:
                logger.warning("No weights provided - using random initialization")

        # Preprocess model parameters
        from ttnn.model_preprocessing import infer_ttnn_module_args, preprocess_model_parameters

        logger.info("Preprocessing model parameters for TTNN...")
        parameters = preprocess_model_parameters(
            initialize_model=lambda: reference_model,
            custom_preprocessor=create_custom_mesh_preprocessor(self.weights_mesh_mapper),
            device=None,
        )

        ########################
        parameters.conv_args = {}
        sample_x = torch.randn(1, 2048, 32, 64)
        sample_res3 = torch.randn(1, 512, 64, 128)
        sample_res2 = torch.randn(1, 256, 128, 256)

        # For semantic decoder
        if hasattr(parameters, "semantic_decoder"):
            # ASPP
            aspp_args = infer_ttnn_module_args(
                model=reference_model.semantic_decoder.aspp, run_model=lambda model: model(sample_x), device=None
            )
            if hasattr(parameters.semantic_decoder, "aspp"):
                parameters.semantic_decoder.aspp.conv_args = aspp_args

            # Res3
            aspp_out = reference_model.semantic_decoder.aspp(sample_x)
            res3_args = infer_ttnn_module_args(
                model=reference_model.semantic_decoder.res3,
                run_model=lambda model: model(aspp_out, sample_res3),
                device=None,
            )
            if hasattr(parameters.semantic_decoder, "res3"):
                parameters.semantic_decoder.res3.conv_args = res3_args

            # Res2
            res3_out = reference_model.semantic_decoder.res3(aspp_out, sample_res3)
            res2_args = infer_ttnn_module_args(
                model=reference_model.semantic_decoder.res2,
                run_model=lambda model: model(res3_out, sample_res2),
                device=None,
            )
            if hasattr(parameters.semantic_decoder, "res2"):
                parameters.semantic_decoder.res2.conv_args = res2_args

            # Head
            res2_out = reference_model.semantic_decoder.res2(res3_out, sample_res2)
            head_args = infer_ttnn_module_args(
                model=reference_model.semantic_decoder.head_1, run_model=lambda model: model(res2_out), device=None
            )
            if hasattr(parameters.semantic_decoder, "head_1"):
                parameters.semantic_decoder.head_1.conv_args = head_args

        # For instance decoder
        if hasattr(parameters, "instance_decoder"):
            # ASPP
            aspp_args = infer_ttnn_module_args(
                model=reference_model.instance_decoder.aspp, run_model=lambda model: model(sample_x), device=None
            )
            if hasattr(parameters.instance_decoder, "aspp"):
                parameters.instance_decoder.aspp.conv_args = aspp_args

            # Res3
            aspp_out = reference_model.instance_decoder.aspp(sample_x)
            res3_args = infer_ttnn_module_args(
                model=reference_model.instance_decoder.res3,
                run_model=lambda model: model(aspp_out, sample_res3),
                device=None,
            )
            if hasattr(parameters.instance_decoder, "res3"):
                parameters.instance_decoder.res3.conv_args = res3_args

            # Res2
            res3_out = reference_model.instance_decoder.res3(aspp_out, sample_res3)
            res2_args = infer_ttnn_module_args(
                model=reference_model.instance_decoder.res2,
                run_model=lambda model: model(res3_out, sample_res2),
                device=None,
            )
            if hasattr(parameters.instance_decoder, "res2"):
                parameters.instance_decoder.res2.conv_args = res2_args

            # Head
            res2_out = reference_model.instance_decoder.res2(res3_out, sample_res2)
            head_args_1 = infer_ttnn_module_args(
                model=reference_model.instance_decoder.head_1, run_model=lambda model: model(res2_out), device=None
            )
            head_args_2 = infer_ttnn_module_args(
                model=reference_model.instance_decoder.head_2, run_model=lambda model: model(res2_out), device=None
            )
            if hasattr(parameters.instance_decoder, "head_1"):
                parameters.instance_decoder.head_1.conv_args = head_args_1
            if hasattr(parameters.instance_decoder, "head_2"):
                parameters.instance_decoder.head_2.conv_args = head_args_2

        torch_conv = reference_model.backbone.layer3[5].conv3
        torch_bn = reference_model.backbone.layer3[5].bn3
        from ttnn.model_preprocessing import fold_batch_norm2d_into_conv2d

        # Fold BN for fair comparison
        folded_weight, _ = fold_batch_norm2d_into_conv2d(torch_conv, torch_bn)

        # Get TTNN weight
        ttnn_weight = ttnn.to_torch(parameters.backbone.layer3[5].conv3.weight)

        print(f"Folded PyTorch: mean={folded_weight.mean():.4f}, std={folded_weight.std():.4f}")
        print(f"TTNN processed: mean={ttnn_weight.mean():.4f}, std={ttnn_weight.std():.4f}")

        model_config = {
            "MATH_FIDELITY": ttnn.MathFidelity.LoFi,
            "WEIGHTS_DTYPE": ttnn.bfloat8_b,
            "ACTIVATIONS_DTYPE": ttnn.bfloat8_b,
        }

        # Create TTNN model
        self.ttnn_model = TTPanopticDeepLab(
            parameters=parameters,
            model_config=model_config,
        )

        logger.info("TTNN model initialized")

    def _setup_mesh_mappers(self):
        """Setup mesh mappers for multi-device support"""
        if self.ttnn_device.get_num_devices() != 1:
            self.inputs_mesh_mapper = ttnn.ShardTensorToMesh(self.ttnn_device, dim=0)
            self.weights_mesh_mapper = None
            self.output_mesh_composer = ttnn.ConcatMeshToTensor(self.ttnn_device, dim=0)
        else:
            self.inputs_mesh_mapper = None
            self.weights_mesh_mapper = None
            self.output_mesh_composer = None

    def save_preprocessed_inputs(self, torch_input: torch.Tensor, save_dir: str, filename: str):
        """Save preprocessed inputs for testing purposes"""

        # Create directory for test inputs
        test_inputs_dir = os.path.join(save_dir, "test_inputs")
        os.makedirs(test_inputs_dir, exist_ok=True)

        # Save torch input tensor
        torch_input_path = os.path.join(test_inputs_dir, f"{filename}_torch_input.pt")
        torch.save(
            {
                "tensor": torch_input,
                "shape": torch_input.shape,
                "dtype": torch_input.dtype,
                "mean": torch_input.mean().item(),
                "std": torch_input.std().item(),
                "min": torch_input.min().item(),
                "max": torch_input.max().item(),
            },
            torch_input_path,
        )

        logger.info(f"Saved preprocessed torch input to: {torch_input_path}")

        # Also save as numpy for compatibility
        numpy_input_path = os.path.join(test_inputs_dir, f"{filename}_input.npy")
        np.save(numpy_input_path, torch_input.cpu().numpy())

        # Save metadata about the preprocessing
        metadata = {
            "original_image_path": self.current_image_path if hasattr(self, "current_image_path") else "unknown",
            "input_shape": list(torch_input.shape),
            "preprocessing": {
                "normalized": self.config.normalize_enabled,
                "mean": self.config.mean,
                "std": self.config.std,
            },
            "input_size": {
                "height": self.config.input_height,
                "width": self.config.input_width,
            },
            "stats": {
                "mean": torch_input.mean().item(),
                "std": torch_input.std().item(),
                "min": torch_input.min().item(),
                "max": torch_input.max().item(),
            },
        }

        metadata_path = os.path.join(test_inputs_dir, f"{filename}_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        return torch_input_path, numpy_input_path, metadata_path

    def preprocess_image(self, image_path: str) -> Tuple[torch.Tensor, ttnn.Tensor, np.ndarray, Tuple[int, int]]:
        """Preprocess image for both PyTorch and TTNN"""
        # Load image
        image = Image.open(image_path).convert("RGB")
        original_size = image.size  # (width, height)
        original_array = np.array(image)

        # Resize to model input size
        target_size = (self.config.input_width, self.config.input_height)  # PIL expects (width, height)
        image_resized = image.resize(target_size)

        # PyTorch preprocessing
        torch_tensor = self.preprocess(image_resized).unsqueeze(0)  # Add batch dimension
        torch_tensor = torch_tensor.to(torch.float)

        # TTNN preprocessing
        ttnn_tensor = None
        if self.config.run_ttnn_pipeline:
            ttnn_tensor = ttnn.from_torch(
                torch_tensor.permute(0, 2, 3, 1),  # BCHW -> BHWC
                dtype=ttnn.bfloat16,
                device=self.ttnn_device,
                mesh_mapper=self.inputs_mesh_mapper,
            )

        print(f"PyTorch input stats: mean={torch_tensor.mean():.4f}, std={torch_tensor.std():.4f}")
        print(f"PyTorch input shape: {torch_tensor.shape}")
        print(f"PyTorch input range: [{torch_tensor.min():.4f}, {torch_tensor.max():.4f}]")

        if ttnn_tensor is not None:
            ttnn_as_torch = ttnn.to_torch(ttnn_tensor)
            print(f"TTNN input stats: mean={ttnn_as_torch.mean():.4f}, std={ttnn_as_torch.std():.4f}")
            print(f"TTNN input shape: {ttnn_as_torch.shape}")
            print(f"TTNN input range: [{ttnn_as_torch.min():.4f}, {ttnn_as_torch.max():.4f}]")

        return torch_tensor, ttnn_tensor, original_array, original_size

    def run_torch_inference(self, input_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Run PyTorch inference"""
        if not self.config.run_torch_pipeline:
            return {}

        logger.info("Running PyTorch inference...")
        start_time = time.time()

        with torch.no_grad():
            outputs, outputs_2, outputs_3 = self.torch_model(input_tensor)

        inference_time = time.time() - start_time
        logger.info(f"PyTorch inference completed in {inference_time:.4f}s")

        return outputs, outputs_2, outputs_3

    def run_ttnn_inference(self, input_tensor: ttnn.Tensor) -> Dict[str, ttnn.Tensor]:
        """Run TTNN inference"""
        if not self.config.run_ttnn_pipeline:
            return {}

        logger.info("Running TTNN inference...")
        start_time = time.time()

        ttnn_outputs, ttnn_outputs_2, ttnn_outputs_3 = self.ttnn_model(input_tensor, self.ttnn_device)

        inference_time = time.time() - start_time
        logger.info(f"TTNN inference completed in {inference_time:.4f}s")

        return ttnn_outputs, ttnn_outputs_2, ttnn_outputs_3

    #     return results
    def postprocess_outputs(
        self,
        torch_outputs: torch.Tensor,
        torch_outputs_2: torch.Tensor,
        torch_outputs_3: torch.Tensor,
        ttnn_outputs: ttnn.Tensor,
        ttnn_outputs_2: ttnn.Tensor,
        ttnn_outputs_3: ttnn.Tensor,
        original_size: Tuple[int, int],
    ) -> Dict[str, Dict]:
        """Postprocess outputs from both pipelines"""
        results = {"torch": {}, "ttnn": {}}

        # Process PyTorch outputs
        if torch_outputs is not None and torch_outputs_2 is not None and torch_outputs_3 is not None:
            logger.info("Processing PyTorch outputs...")
            try:
                semantic_logits = torch_outputs
                offset_map = torch_outputs_2
                center_heatmap = torch_outputs_3

                # semantic_logits
                if semantic_logits is not None and isinstance(semantic_logits, torch.Tensor):
                    np_array = semantic_logits.squeeze(0).cpu().numpy()
                    semantic_pred = np.argmax(np_array, axis=0)
                    results["torch"]["semantic_pred"] = cv2.resize(
                        semantic_pred.astype(np.uint8), original_size, interpolation=cv2.INTER_NEAREST
                    )
                    logger.debug(f"PyTorch semantic_pred shape: {results['torch']['semantic_pred'].shape}")

                # center_heatmap
                if center_heatmap is not None and isinstance(center_heatmap, torch.Tensor):
                    np_array = center_heatmap.squeeze(0).cpu().numpy()
                    if len(np_array.shape) == 3 and np_array.shape[0] == 1:
                        center_data = np_array[0]  # Remove channel dim
                    elif len(np_array.shape) == 2:
                        center_data = np_array
                    else:
                        center_data = np_array  # Use as is

                    results["torch"]["center_heatmap"] = cv2.resize(
                        center_data, original_size, interpolation=cv2.INTER_LINEAR
                    )
                    logger.debug(f"PyTorch center_heatmap shape: {results['torch']['center_heatmap'].shape}")

                # offset_map
                if offset_map is not None and isinstance(offset_map, torch.Tensor):
                    np_array = offset_map.squeeze(0).cpu().numpy()
                    if len(np_array.shape) == 3 and np_array.shape[0] == 2:
                        results["torch"]["offset_map"] = np.stack(
                            [
                                cv2.resize(np_array[0], original_size, interpolation=cv2.INTER_LINEAR),
                                cv2.resize(np_array[1], original_size, interpolation=cv2.INTER_LINEAR),
                            ]
                        )
                    else:
                        logger.warning(f"Unexpected offset_map shape: {np_array.shape}")
                # panoptic_pred
                panoptic_pred = PostProcessing(
                    thing_classes=self.config.thing_classes, stuff_classes=self.config.stuff_classes
                ).panoptic_fusion(semantic_logits=semantic_logits, center_heatmap=center_heatmap, offset_map=offset_map)
                if panoptic_pred is not None and isinstance(panoptic_pred, torch.Tensor):
                    np_array = panoptic_pred.squeeze(0).cpu().numpy()
                    results["torch"]["panoptic_pred"] = cv2.resize(
                        np_array.astype(np.int32), original_size, interpolation=cv2.INTER_NEAREST
                    )
                    logger.debug(f"PyTorch panoptic_pred shape: {results['torch']['panoptic_pred'].shape}")

            except Exception as e:
                logger.error(f"Error processing PyTorch outputs: {e}")

        # Process TTNN outputs
        if ttnn_outputs is not None and ttnn_outputs_2 is not None and ttnn_outputs_3 is not None:
            logger.info("Processing TTNN outputs...")
            try:
                # Convert TTNN to torch tensor
                torch_tensor = ttnn.to_torch(
                    ttnn_outputs, device=self.ttnn_device, mesh_composer=self.output_mesh_composer
                )
                torch_tensor_2 = ttnn.to_torch(
                    ttnn_outputs_2, device=self.ttnn_device, mesh_composer=self.output_mesh_composer
                )
                torch_tensor_3 = ttnn.to_torch(
                    ttnn_outputs_3, device=self.ttnn_device, mesh_composer=self.output_mesh_composer
                )

                # Debug: Log raw TTNN output stats
                logger.debug(
                    f"TTNN raw output 1 (semantic): shape={torch_tensor.shape}, "
                    f"min={torch_tensor.min():.4f}, max={torch_tensor.max():.4f}, "
                    f"mean={torch_tensor.mean():.4f}"
                )
                logger.debug(
                    f"TTNN raw output 2 (offset): shape={torch_tensor_2.shape}, "
                    f"min={torch_tensor_2.min():.4f}, max={torch_tensor_2.max():.4f}, "
                    f"mean={torch_tensor_2.mean():.4f}"
                )
                logger.debug(
                    f"TTNN raw output 3 (center): shape={torch_tensor_3.shape}, "
                    f"min={torch_tensor_3.min():.4f}, max={torch_tensor_3.max():.4f}, "
                    f"mean={torch_tensor_3.mean():.4f}"
                )

                # Reshape to proper format
                reshaped_tensor = self._reshape_ttnn_output(torch_tensor, "semantic_logits")
                reshaped_tensor_2 = self._reshape_ttnn_output(torch_tensor_2, "offset_map")
                reshaped_tensor_3 = self._reshape_ttnn_output(torch_tensor_3, "center_heatmap")

                # Process semantic segmentation
                np_array = reshaped_tensor.squeeze(0).cpu().float().numpy()
                semantic_pred = np.argmax(np_array, axis=0)
                results["ttnn"]["semantic_pred"] = cv2.resize(
                    semantic_pred.astype(np.uint8), original_size, interpolation=cv2.INTER_NEAREST
                )
                logger.debug(f"TTNN semantic_pred shape: {results['ttnn']['semantic_pred'].shape}")

                # Process center heatmap - FIXED: This was in elif block before
                np_array_3 = reshaped_tensor_3.squeeze(0).cpu().float().numpy()

                # Debug center heatmap values
                logger.debug(
                    f"TTNN center heatmap stats: shape={np_array_3.shape}, "
                    f"min={np_array_3.min():.4f}, max={np_array_3.max():.4f}, "
                    f"mean={np_array_3.mean():.4f}, std={np_array_3.std():.4f}"
                )

                if len(np_array_3.shape) == 3 and np_array_3.shape[0] == 1:
                    center_data = np_array_3[0]  # Remove channel dim
                elif len(np_array_3.shape) == 2:
                    center_data = np_array_3
                else:
                    center_data = np_array_3
                    logger.warning(f"Unexpected center heatmap shape: {np_array_3.shape}, using as is")

                # Check if center data has very small values and scale if needed
                if center_data.max() < 0.01:
                    logger.warning(f"Center heatmap has very small values (max={center_data.max():.6f}), scaling up")
                    center_data = center_data * 100  # Scale up for visualization

                results["ttnn"]["center_heatmap"] = cv2.resize(
                    center_data, original_size, interpolation=cv2.INTER_LINEAR
                )
                logger.debug(f"TTNN center_heatmap shape: {results['ttnn']['center_heatmap'].shape}")

                # Process offset map - FIXED: This was in elif block before
                np_array_2 = reshaped_tensor_2.squeeze(0).cpu().float().numpy()
                if len(np_array_2.shape) == 3 and np_array_2.shape[0] == 2:
                    results["ttnn"]["offset_map"] = np.stack(
                        [
                            cv2.resize(np_array_2[0], original_size, interpolation=cv2.INTER_LINEAR),
                            cv2.resize(np_array_2[1], original_size, interpolation=cv2.INTER_LINEAR),
                        ]
                    )
                    logger.debug(f"TTNN offset_map shape: {results['ttnn']['offset_map'].shape}")
                else:
                    logger.warning(f"Unexpected TTNN offset_map shape: {np_array_2.shape}")

                # panoptic_pred_ttnn
                # semantic_logits = ttnn.to_torch(semantic_logits) if hasattr(ttnn, "to_torch") else semantic_logits
                # center_heatmap = ttnn.to_torch(center_heatmap) if hasattr(ttnn, "to_torch") else center_heatmap
                # offset_map = ttnn.to_torch(offset_map) if hasattr(ttnn, "to_torch") else offset_map

                semantic_logits = reshaped_tensor
                center_heatmap = reshaped_tensor_3
                offset_map = reshaped_tensor_2

                panoptic_pred_ttnn = PostProcessing().panoptic_fusion(
                    semantic_logits=semantic_logits, center_heatmap=center_heatmap, offset_map=offset_map
                )
                panoptic_pred = ttnn.from_torch(panoptic_pred_ttnn, dtype=ttnn.int32)

                if panoptic_pred is not None and isinstance(panoptic_pred, ttnn.Tensor):
                    # Convert TTNN tensor to numpy properly
                    torch_tensor = ttnn.to_torch(
                        panoptic_pred, device=self.ttnn_device, mesh_composer=self.output_mesh_composer
                    )
                    np_array = torch_tensor.squeeze().cpu().numpy()

                    # Debug information
                    logger.debug(f"Panoptic array shape: {np_array.shape}")
                    logger.debug(f"Original size for resize: {original_size}")

                    # Validate original_size before resize
                    if not original_size or len(original_size) != 2 or original_size[0] <= 0 or original_size[1] <= 0:
                        logger.error(f"Invalid original_size: {original_size}")
                        original_size = (self.config.input_width, self.config.input_height)
                        logger.warning(f"Using fallback size: {original_size}")

                    # Ensure valid array dimensions
                    if np_array.size > 0 and len(np_array.shape) >= 2:
                        if len(np_array.shape) > 2:
                            np_array = np_array.squeeze()

                        results["ttnn"]["panoptic_pred"] = cv2.resize(
                            np_array.astype(np.int32), original_size, interpolation=cv2.INTER_NEAREST
                        )
                        logger.debug(f"TTNN panoptic_pred shape: {results['ttnn']['panoptic_pred'].shape}")
                    else:
                        logger.error("Invalid array for resize - skipping panoptic processing")

                # if panoptic_pred is not None and isinstance(panoptic_pred, ttnn.Tensor):
                #     if len(np_array.shape) > 2:
                #         np_array = np_array.squeeze()  # Remove single dimensions
                #     results["ttnn"]["panoptic_pred"] = cv2.resize(
                #         np_array.astype(np.uint8), original_size, interpolation=cv2.INTER_NEAREST
                #     )
                #     logger.debug(f"TTNN panoptic_pred shape: {results['ttnn']['panoptic_pred'].shape}")

            except Exception as e:
                logger.error(f"Error processing TTNN output: {e}")
                import traceback

                traceback.print_exc()

        return results

    def _reshape_ttnn_output(self, tensor: torch.Tensor, key: str) -> torch.Tensor:
        """Reshape TTNN output tensor to proper format - IMPROVED VERSION"""

        logger.debug(f"Reshaping TTNN output for {key}: input shape = {tensor.shape}")

        if len(tensor.shape) == 4:  # BHWC format from TTNN
            B, H, W, C = tensor.shape

            if key == "semantic_logits":
                # Should have num_classes channels
                expected_c = self.config.num_classes
                if C == expected_c:
                    result = tensor.permute(0, 3, 1, 2)  # BHWC -> BCHW
                else:
                    # Try to reshape if flattened
                    total_elements = B * H * W * C
                    expected_h = self.config.input_height // 4  # Typical output stride
                    expected_w = self.config.input_width // 4
                    if total_elements == B * expected_h * expected_w * expected_c:
                        tensor = tensor.reshape(B, expected_h, expected_w, expected_c)
                        result = tensor.permute(0, 3, 1, 2)
                    else:
                        logger.warning(f"Unexpected semantic_logits shape: {tensor.shape}")
                        result = tensor

            elif key == "center_heatmap":
                # Should have 1 channel
                if C == 1:
                    result = tensor.permute(0, 3, 1, 2)  # BHWC -> BCHW
                else:
                    # Take first channel or reshape
                    if C > 1:
                        tensor = tensor[:, :, :, :1]  # Take first channel
                    result = tensor.permute(0, 3, 1, 2)

            elif key == "offset_map":
                # Should have 2 channels (x, y offsets)
                if C == 2:
                    result = tensor.permute(0, 3, 1, 2)  # BHWC -> BCHW
                else:
                    logger.warning(f"Unexpected offset_map channels: {C}, expected 2")
                    result = tensor
            else:
                result = tensor

        elif len(tensor.shape) == 3:  # BHW format
            result = tensor

        else:
            logger.warning(f"Unexpected tensor shape for {key}: {tensor.shape}")
            result = tensor

        logger.debug(f"Reshaped TTNN output for {key}: output shape = {result.shape}")
        return result

    def compare_outputs(self, results: Dict[str, Dict]) -> Dict[str, float]:
        """Compare PyTorch and TTNN outputs"""
        if not (self.config.compare_outputs and "torch" in results and "ttnn" in results):
            return {}

        logger.info("Comparing PyTorch and TTNN outputs...")
        pcc_scores = {}

        for key in ["semantic_pred", "center_heatmap", "offset_map", "panoptic_pred"]:
            if key in results["torch"] and key in results["ttnn"]:
                torch_output = results["torch"][key]
                ttnn_output = results["ttnn"][key]

                # Debug shape information
                logger.debug(f"Comparing {key}:")
                logger.debug(f"  PyTorch shape: {torch_output.shape}")
                logger.debug(f"  TTNN shape: {ttnn_output.shape}")

                # Handle different shaped arrays
                if torch_output.shape != ttnn_output.shape:
                    logger.warning(f"  Shape mismatch for {key}: {torch_output.shape} vs {ttnn_output.shape}")
                    # Try to make compatible
                    if key == "offset_map":
                        if len(torch_output.shape) == 3 and len(ttnn_output.shape) == 2:
                            # Flatten both to same shape
                            torch_flat = torch_output.flatten()
                            ttnn_flat = ttnn_output.flatten()
                        else:
                            continue
                    else:
                        # Flatten both arrays
                        torch_flat = torch_output.flatten()
                        ttnn_flat = ttnn_output.flatten()

                        # Truncate to same length if needed
                        min_len = min(len(torch_flat), len(ttnn_flat))
                        torch_flat = torch_flat[:min_len]
                        ttnn_flat = ttnn_flat[:min_len]
                else:
                    torch_flat = torch_output.flatten()
                    ttnn_flat = ttnn_output.flatten()

                # Calculate statistics
                logger.debug(f"  PyTorch stats: mean={torch_flat.mean():.4f}, std={torch_flat.std():.4f}")
                logger.debug(f"  TTNN stats: mean={ttnn_flat.mean():.4f}, std={ttnn_flat.std():.4f}")

    #     logger.info(f"Visualization saved to: {save_path}")
    def visualize_results(self, original_image: np.ndarray, results: Dict, save_path: str):
        """Create comprehensive visualization with panoptic in separate row"""
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        logger.info("Creating visualization...")
        has_torch = "torch" in results and results["torch"]
        has_ttnn = "ttnn" in results and results["ttnn"]

        if has_torch and has_ttnn:
            # 4 rows for dual pipeline: original, torch outputs, ttnn outputs, panoptic comparison
            fig = plt.figure(figsize=(16, 20))
            gs = fig.add_gridspec(4, 4, height_ratios=[0.8, 1, 1, 1], hspace=0.3, wspace=0.2)
            pipelines = ["torch", "ttnn"]
        else:
            fig = plt.figure(figsize=(16, 12))
            gs = fig.add_gridspec(3, 4, height_ratios=[0.8, 1, 1], hspace=0.3, wspace=0.2)
            pipelines = ["torch"] if has_torch else ["ttnn"]

        # Row 0: Original image (centered)
        ax_orig = fig.add_subplot(gs[0, 1:3])
        ax_orig.imshow(original_image)
        ax_orig.set_title("Original Image", fontsize=14, fontweight="bold")
        ax_orig.axis("off")

        # Hide unused cells in first row
        for i in [0, 3]:
            ax = fig.add_subplot(gs[0, i])
            ax.axis("off")

        # Rows 1-2: Pipeline outputs (semantic, centers, offset)
        for i, pipeline in enumerate(pipelines):
            if pipeline not in results:
                continue
            pipeline_results = results[pipeline]
            row = i + 1

            # Semantic segmentation
            ax_sem = fig.add_subplot(gs[row, 0])
            if "semantic_pred" in pipeline_results:
                semantic_colored = self._colorize_segmentation(pipeline_results["semantic_pred"])
                ax_sem.imshow(semantic_colored)
                ax_sem.set_title(f"{pipeline.upper()} Semantic", fontsize=11)
            ax_sem.axis("off")

            # Centers
            ax_center = fig.add_subplot(gs[row, 1])
            if "center_heatmap" in pipeline_results:
                center_data = pipeline_results["center_heatmap"]
                if center_data.max() > center_data.min():
                    center_normalized = (center_data - center_data.min()) / (center_data.max() - center_data.min())
                else:
                    center_normalized = center_data
                ax_center.imshow(original_image, alpha=0.5)
                ax_center.imshow(center_normalized, cmap="hot", alpha=0.5, vmin=0, vmax=1)
                ax_center.set_title(f"{pipeline.upper()} Centers", fontsize=11)
            ax_center.axis("off")

            # Offset
            ax_offset = fig.add_subplot(gs[row, 2])
            if "offset_map" in pipeline_results:
                offset_data = pipeline_results["offset_map"]
                if len(offset_data.shape) == 3 and offset_data.shape[0] == 2:
                    offset_magnitude = np.sqrt(offset_data[0] ** 2 + offset_data[1] ** 2)
                else:
                    offset_magnitude = offset_data
                vmax = offset_magnitude.max() if offset_magnitude.max() > 0 else 1
                im = ax_offset.imshow(offset_magnitude, cmap="viridis", vmin=0, vmax=vmax)
                ax_offset.set_title(f"{pipeline.upper()} Offset", fontsize=11)

                # Add small colorbar
                divider = make_axes_locatable(ax_offset)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(im, cax=cax)
            ax_offset.axis("off")

            # Hide the 4th column in these rows
            ax_empty = fig.add_subplot(gs[row, 3])
            ax_empty.axis("off")

        # Row 3 (or 2 for single pipeline): Panoptic comparison
        if has_torch and has_ttnn:
            # Both panoptic side by side
            for i, pipeline in enumerate(pipelines):
                ax_pan = fig.add_subplot(gs[3, i * 2 : (i * 2) + 2])
                if pipeline in results and "panoptic_pred" in results[pipeline]:
                    panoptic_colored = self._colorize_panoptic(results[pipeline]["panoptic_pred"])
                    if self.config.show_instance_labels:
                        panoptic_with_labels = self._add_instance_labels(
                            panoptic_colored, results[pipeline]["panoptic_pred"]
                        )
                        ax_pan.imshow(panoptic_with_labels)
                    else:
                        ax_pan.imshow(panoptic_colored)
                    ax_pan.set_title(f"{pipeline.upper()} Panoptic Segmentation", fontsize=12, fontweight="bold")
                ax_pan.axis("off")
        else:
            # Single panoptic centered
            pipeline = pipelines[0]
            ax_pan = fig.add_subplot(gs[2, :])
            if "panoptic_pred" in results[pipeline]:
                panoptic_colored = self._colorize_panoptic(results[pipeline]["panoptic_pred"])
                if self.config.show_instance_labels:
                    panoptic_with_labels = self._add_instance_labels(
                        panoptic_colored, results[pipeline]["panoptic_pred"]
                    )
                    ax_pan.imshow(panoptic_with_labels)
                else:
                    ax_pan.imshow(panoptic_colored)
                ax_pan.set_title(f"{pipeline.upper()} Panoptic Segmentation", fontsize=12, fontweight="bold")
            ax_pan.axis("off")

        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Visualization saved to: {save_path}")

    def _add_instance_labels(self, image: np.ndarray, panoptic: np.ndarray) -> np.ndarray:
        """Add instance labels to the image"""
        import cv2

        labeled_image = image.copy()
        label_divisor = 1000

        # Group by semantic class to avoid duplicate labels
        labeled_classes = set()
        unique_ids = np.unique(panoptic)

        for pan_id in unique_ids:
            semantic_class = pan_id // label_divisor
            instance_id = pan_id % label_divisor

            if pan_id > 0 and semantic_class < len(self.config.class_names):
                # Only label once per semantic class for cleaner view
                class_key = semantic_class if instance_id == 0 else f"{semantic_class}_{instance_id}"

                if semantic_class not in labeled_classes or instance_id > 0:
                    mask = (panoptic == pan_id).astype(np.uint8)
                    M = cv2.moments(mask)

                    if M["m00"] > 500:  # Larger threshold for cleaner labels
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])

                        # Simple class name only
                        label = self.config.class_names[semantic_class]

                        # Add text with semi-transparent background
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 0.5
                        thickness = 2
                        text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]

                        # Semi-transparent white background
                        overlay = labeled_image.copy()
                        cv2.rectangle(
                            overlay,
                            (cx - 3, cy - text_size[1] - 3),
                            (cx + text_size[0] + 3, cy + 3),
                            (255, 255, 255),
                            -1,
                        )
                        cv2.addWeighted(overlay, 0.7, labeled_image, 0.3, 0, labeled_image)

                        # Black text
                        cv2.putText(labeled_image, label, (cx, cy), font, font_scale, (0, 0, 0), thickness)

                        if instance_id == 0:
                            labeled_classes.add(semantic_class)

        return labeled_image

    def _colorize_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        """Convert segmentation map to colored image"""
        colored = np.zeros((*segmentation.shape, 3), dtype=np.uint8)
        for class_id in range(self.config.num_classes):
            mask = segmentation == class_id
            if class_id < len(self.colors):
                colored[mask] = self.colors[class_id]
        return colored

    def _colorize_panoptic(self, panoptic: np.ndarray) -> np.ndarray:
        """Convert panoptic prediction to colored image with proper instance coloring"""
        colored = np.zeros((*panoptic.shape, 3), dtype=np.uint8)
        label_divisor = 1000

        unique_ids = np.unique(panoptic)

        # Use more vibrant colors for instances
        instance_colors = {}

        for pan_id in unique_ids:
            mask = panoptic == pan_id
            semantic_class = pan_id // label_divisor
            instance_id = pan_id % label_divisor

            if semantic_class < len(self.colors):
                base_color = self.colors[semantic_class]

                if instance_id == 0:
                    # Stuff classes - use original semantic color
                    colored[mask] = base_color
                else:
                    # Thing instances - create distinct colors
                    if pan_id not in instance_colors:
                        # Generate distinct color for each instance
                        np.random.seed(int(pan_id))

                        # Use golden angle for hue distribution
                        golden_angle = 137.5
                        hue = ((instance_id - 1) * golden_angle) % 360

                        # High saturation and value for vibrancy
                        from colorsys import hsv_to_rgb

                        r, g, b = hsv_to_rgb(hue / 360, 0.8, 0.9)
                        instance_colors[pan_id] = np.array([r * 255, g * 255, b * 255], dtype=np.uint8)

                    colored[mask] = instance_colors[pan_id]
            else:
                colored[mask] = [128, 128, 128]

        return colored

    def save_results(self, results: Dict, original_image: np.ndarray, output_dir: str, filename: str):
        """Save all results to output directory"""
        os.makedirs(output_dir, exist_ok=True)

        # Save original image
        original_path = os.path.join(output_dir, f"{filename}_original.png")
        Image.fromarray(original_image).save(original_path)

        # Save results for each pipeline
        for pipeline, pipeline_results in results.items():
            pipeline_dir = os.path.join(output_dir, pipeline)
            os.makedirs(pipeline_dir, exist_ok=True)

            # Save semantic segmentation
            if "semantic_pred" in pipeline_results:
                semantic_colored = self._colorize_segmentation(pipeline_results["semantic_pred"])
                semantic_path = os.path.join(pipeline_dir, f"{filename}_semantic.png")
                Image.fromarray(semantic_colored).save(semantic_path)

                # Save raw semantic prediction
                raw_semantic_path = os.path.join(pipeline_dir, f"{filename}_semantic_raw.npy")
                np.save(raw_semantic_path, pipeline_results["semantic_pred"])

            # Save center heatmap
            if "center_heatmap" in pipeline_results:
                center_path = os.path.join(pipeline_dir, f"{filename}_centers.png")
                center_normalized = (pipeline_results["center_heatmap"] * 255).astype(np.uint8)
                Image.fromarray(center_normalized, mode="L").save(center_path)

                # Save raw center heatmap
                raw_center_path = os.path.join(pipeline_dir, f"{filename}_centers_raw.npy")
                np.save(raw_center_path, pipeline_results["center_heatmap"])

            # Save offset map
            if "offset_map" in pipeline_results:
                offset_path = os.path.join(pipeline_dir, f"{filename}_offset_raw.npy")
                np.save(offset_path, pipeline_results["offset_map"])

            # Save panoptic segmentation
            if "panoptic_pred" in pipeline_results:
                panoptic_colored = self._colorize_panoptic(pipeline_results["panoptic_pred"])
                panoptic_path = os.path.join(pipeline_dir, f"{filename}_panoptic.png")
                Image.fromarray(panoptic_colored).save(panoptic_path)

                # Save raw panoptic prediction
                raw_panoptic_path = os.path.join(pipeline_dir, f"{filename}_panoptic_raw.npy")
                np.save(raw_panoptic_path, pipeline_results["panoptic_pred"])

        logger.info(f"Results saved to: {output_dir}")

    def run_demo(self, image_path: str, output_dir: str):
        """Run complete demo pipeline"""
        logger.info(f"Starting demo for image: {image_path}")

        self.current_image_path = image_path
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Initialize models
        if self.config.run_torch_pipeline:
            self.initialize_torch_model()

        if self.config.run_ttnn_pipeline:
            self.initialize_ttnn_model()

        # Preprocess image
        torch_input, ttnn_input, original_image, original_size = self.preprocess_image(image_path)

        #############################
        #############################
        base_name = Path(image_path).stem
        torch_input_path, numpy_input_path, metadata_path = self.save_preprocessed_inputs(
            torch_input, output_dir, base_name
        )
        logger.info(f"Preprocessed inputs saved for testing: {torch_input_path}")
        #############################
        # import onnx
        # # onnx.save_model(self.torch_model, "/home/ubuntu/ign-tt-sm/tt-metal/models/experimental/panoptic_deeplab/panoptic_deeplab_torch_model.onnx")
        # torch.onnx.export(
        #     self.torch_model,
        #     torch_input,
        #     "/home/ubuntu/ign-tt-sm/tt-metal/models/experimental/panoptic_deeplab/panoptic_deeplab_torch_model.onnx",
        #     export_params=True,       # store trained weights
        # )
        #############################
        # Run inference
        torch_outputs, torch_outputs_2, torch_outputs_3 = (
            self.run_torch_inference(torch_input) if self.config.run_torch_pipeline else {}
        )
        ttnn_outputs, ttnn_outputs_2, ttnn_outputs_3 = (
            self.run_ttnn_inference(ttnn_input) if self.config.run_ttnn_pipeline else {}
        )

        # print(f"torch_outputs: {torch_outputs}")
        # print(f"ttnn_outputs: {ttnn_outputs}")

        # Postprocess results
        results = self.postprocess_outputs(
            torch_outputs, torch_outputs_2, torch_outputs_3, ttnn_outputs, ttnn_outputs_2, ttnn_outputs_3, original_size
        )
        # # Validate outputs
        # if torch_outputs is not None and ttnn_outputs is not None:
        #     self.validate(torch_outputs, ttnn_outputs)

        # Compare outputs if both pipelines ran
        self.compare_outputs(results)
        # pcc_scores = self.compare_outputs(results)

        # Generate filename
        base_name = Path(image_path).stem

        # Save individual results
        self.save_results(results, original_image, output_dir, base_name)

        # Create visualization
        viz_path = os.path.join(output_dir, f"{base_name}_comparison.png")
        self.visualize_results(original_image, results, viz_path)

        # Save metadata and results summary
        self._save_metadata(image_path, results, output_dir, base_name)
        # self._save_metadata(image_path, results, pcc_scores, output_dir, base_name)

        logger.info(f"Demo completed! Results saved to: {output_dir}")

        # Cleanup
        if ttnn_input is not None:
            ttnn.deallocate(ttnn_input)

        # for tensor in ttnn_outputs and tensor in ttnn_outputs_2 and tensor in ttnn_outputs_3:
        #     if hasattr(tensor, "deallocate"):
        #         ttnn.deallocate(tensor)

    def _save_metadata(self, image_path: str, results: Dict, output_dir: str, filename: str):
        # def _save_metadata(self, image_path: str, results: Dict, pcc_scores: Dict, output_dir: str, filename: str):
        """Save metadata and comparison results"""
        metadata = {
            "image_path": image_path,
            "config": asdict(self.config),
            "results": {
                "pipelines_run": list(results.keys()),
                # "pcc_scores": pcc_scores,
            },
            "output_files": {
                "visualization": f"{filename}_comparison.png",
                "original": f"{filename}_original.png",
            },
        }

        # Add pipeline-specific metadata
        for pipeline in results.keys():
            metadata["output_files"][pipeline] = {
                "semantic": f"{pipeline}/{filename}_semantic.png",
                "centers": f"{pipeline}/{filename}_centers.png",
                "panoptic": f"{pipeline}/{filename}_panoptic.png",
            }

        metadata_path = os.path.join(output_dir, f"{filename}_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        # Create summary report
        summary_path = os.path.join(output_dir, f"{filename}_summary.txt")
        with open(summary_path, "w") as f:
            f.write(f"Panoptic DeepLab Demo Results\n")
            f.write(f"============================\n\n")
            f.write(f"Image: {image_path}\n")
            f.write(f"Input Size: {self.config.input_height}x{self.config.input_width}\n")
            f.write(f"Pipelines Run: {', '.join(results.keys())}\n\n")

            # if pcc_scores:
            #     f.write(f"PCC Comparison Results:\n")
            #     for key, score in pcc_scores.items():
            #         status = "PASS" if score >= self.config.pcc_threshold else "FAIL"
            #         f.write(f"  {key}: {score:.4f} ({status})\n")

            #     avg_pcc = np.mean(list(pcc_scores.values()))
            #     overall_status = "PASS" if avg_pcc >= self.config.pcc_threshold else "FAIL"
            #     f.write(f"\nOverall Average PCC: {avg_pcc:.4f} ({overall_status})\n")

    def cleanup(self):
        """Cleanup resources"""
        if self.ttnn_device is not None:
            ttnn.close_device(self.ttnn_device)
            logger.info("TTNN device closed")


def create_sample_configs():
    """Create sample configuration files"""

    configs_dir = Path("configs")
    configs_dir.mkdir(exist_ok=True)

    # Basic Cityscapes config
    basic_config = DemoConfig(
        weights_path=None,  # No weights for basic demo
        input_height=512,
        input_width=1024,
        run_torch_pipeline=True,
        run_ttnn_pipeline=True,
        compare_outputs=True,
    )
    basic_config.to_yaml("configs/demo_basic.yaml")

    # High resolution config
    hr_config = DemoConfig(
        weights_path="models/panoptic_deeplab_r52_cityscapes.pkl",
        input_height=1024,
        input_width=2048,
        center_threshold=0.15,
        nms_kernel=9,
        math_fidelity="HiFi4",
    )
    hr_config.to_yaml("configs/demo_high_res.yaml")

    # Fast inference config
    fast_config = DemoConfig(
        input_height=256,
        input_width=512,
        center_threshold=0.2,
        top_k_instances=100,
        math_fidelity="HiFi4",
        run_torch_pipeline=False,  # Only TTNN for speed
        compare_outputs=False,
    )
    fast_config.to_yaml("configs/demo_fast.yaml")

    # Comparison config (for validation)
    comparison_config = DemoConfig(
        weights_path="models/panoptic_deeplab_r52_cityscapes.pkl",
        input_height=512,
        input_width=1024,
        run_torch_pipeline=True,
        run_ttnn_pipeline=True,
        compare_outputs=True,
        pcc_threshold=0.95,
        save_semantic=True,
        save_instance=True,
        save_panoptic=True,
        save_visualization=True,
        save_comparison=True,
    )
    comparison_config.to_yaml("configs/demo_comparison.yaml")

    logger.info(f"Sample configurations created in: {configs_dir}")


def main():
    """Main demo function"""
    parser = argparse.ArgumentParser(description="Enhanced TT Panoptic DeepLab Demo with YAML Config")
    parser.add_argument(
        "--config", "-c", type=str, default="configs/demo_basic.yaml", help="Path to YAML configuration file"
    )
    parser.add_argument("--input", "-i", type=str, required=False, help="Path to input image")
    parser.add_argument("--output", "-o", type=str, required=False, help="Output directory for results")
    parser.add_argument("--create-configs", action="store_true", help="Create sample configuration files and exit")

    # Override options
    parser.add_argument("--weights", type=str, required=False, help="Override model weights path")
    parser.add_argument(
        "--input-size", nargs=2, type=int, metavar=("H", "W"), help="Override input size (height width)"
    )
    parser.add_argument("--torch-only", action="store_true", help="Run only PyTorch pipeline")
    parser.add_argument("--ttnn-only", action="store_true", help="Run only TTNN pipeline")
    parser.add_argument("--device-id", type=int, help="Override TT device ID")

    args = parser.parse_args()

    # Create sample configs if requested
    if args.create_configs:
        create_sample_configs()
        return 0

    # Validate input file
    if not os.path.exists(args.input):
        logger.error(f"Input image not found: {args.input}")
        return 1

    # Load configuration
    if os.path.exists(args.config):
        config = DemoConfig.from_yaml(args.config)
        logger.info(f"Loaded configuration from: {args.config}")
    else:
        logger.warning(f"Config file not found: {args.config}, using default configuration")
        config = DemoConfig()

    # Apply command line overrides
    if args.weights:
        config.weights_path = args.weights
    if args.input_size:
        config.input_height, config.input_width = args.input_size
    if args.torch_only:
        config.run_torch_pipeline = True
        config.run_ttnn_pipeline = False
        config.compare_outputs = False
    if args.ttnn_only:
        config.run_torch_pipeline = False
        config.run_ttnn_pipeline = True
        config.compare_outputs = False
    if args.device_id is not None:
        config.device_id = args.device_id

    # Validate configuration
    if config.weights_path and not os.path.exists(config.weights_path):
        logger.warning(f"Weights file not found: {config.weights_path}")
        logger.info("Proceeding with random initialization")

    # Initialize demo
    logger.info("=== Enhanced Panoptic DeepLab Demo ===")
    logger.info(f"Config: {args.config}")
    logger.info(f"Input: {args.input}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Input Size: {config.input_height}x{config.input_width}")
    logger.info(f"PyTorch: {'ON' if config.run_torch_pipeline else 'OFF'}")
    logger.info(f"TTNN: {'ON' if config.run_ttnn_pipeline else 'OFF'}")

    try:
        demo = DualPipelineDemo(config)
        demo.run_demo(args.input, args.output)
        logger.info("Demo completed successfully!")
        return 0

    except Exception as e:
        logger.error(f"Demo failed with error: {e}")
        import traceback

        traceback.print_exc()
        return 1

    finally:
        # Cleanup
        try:
            demo.cleanup()
        except:
            pass


if __name__ == "__main__":
    exit(main())
