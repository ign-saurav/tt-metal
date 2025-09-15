# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from loguru import logger

import pickle
import numpy as np
import os
from PIL import Image
from typing import Tuple
import torchvision.transforms as transforms
from typing import Optional, Any
import ttnn
from models.experimental.panoptic_deeplab.reference.resnet52_backbone import ResNet52BackBone as TorchBackbone
from models.experimental.panoptic_deeplab.reference.resnet52_stem import DeepLabStem
from torchvision.models.resnet import Bottleneck
from models.experimental.panoptic_deeplab.reference.aspp import ASPPModel
from models.experimental.panoptic_deeplab.reference.decoder import DecoderModel
from models.experimental.panoptic_deeplab.reference.res_block import ResModel
from models.experimental.panoptic_deeplab.reference.head import HeadModel
from models.experimental.panoptic_deeplab.reference.panoptic_deeplab import TorchPanopticDeepLab


def map_single_key(checkpoint_key):
    """
    Map checkpoint keys to model keys.
    """

    if not checkpoint_key:
        return ""

    key = checkpoint_key

    # BACKBONE MAPPINGS
    if key.startswith("backbone."):
        # Batch norm mapping: conv1/2/3.norm -> bn1/2/3
        key = key.replace("conv1.norm", "bn1")
        key = key.replace("conv2.norm", "bn2")
        key = key.replace("conv3.norm", "bn3")

        # Downsample mapping: shortcut -> downsample
        key = key.replace(".shortcut.norm.", ".downsample.1.")
        # Handle shortcut.weight
        if ".shortcut." in key and ".shortcut.norm." not in checkpoint_key:
            key = key.replace(".shortcut.", ".downsample.0.")

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
    if ".decoder.res5.project_conv." in key:
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


def load_partial_state(torch_model: torch.nn.Module, state_dict, layer_name: str = ""):
    partial_state_dict = {}
    layer_prefix = layer_name + "."
    for k, v in state_dict.items():
        if k.startswith(layer_prefix):
            partial_state_dict[k[len(layer_prefix) :]] = v
    torch_model.load_state_dict(partial_state_dict, strict=True)
    logger.info(f"Successfully loaded all mapped weights with strict=True")
    return torch_model


def load_torch_model_state(torch_model: torch.nn.Module = None, layer_name: str = "", model_location_generator=None):
    if model_location_generator == None or "TT_GH_CI_INFRA" not in os.environ:
        model_path = "models"
    else:
        model_path = model_location_generator("vision-models/panoptic_deeplab", model_subdir="", download_if_ci_v2=True)
    if model_path == "models":
        if not os.path.exists(
            "models/experimental/panoptic_deeplab/resources/Panoptic_Deeplab_R52.pkl"
        ):  # check if Panoptic_Deeplab_R52.pkl is available
            os.system(
                "models/experimental/panoptic_deeplab/resources/panoptic_deeplab_weights_download.sh"
            )  # execute the panoptic_deeplab_weights_download.sh file
        weights_path = "models/experimental/panoptic_deeplab/resources/Panoptic_Deeplab_R52.pkl"
    else:
        weights_path = os.path.join(model_path, "Panoptic_Deeplab_R52.pkl")

    # Load checkpoint
    with open(weights_path, "rb") as f:
        checkpoint = pickle.load(f, encoding="latin1")
    state_dict = checkpoint["model"]

    converted_count = 0
    for k, v in state_dict.items():
        if isinstance(v, np.ndarray) or isinstance(v, np.array):
            state_dict[k] = torch.from_numpy(v)
            converted_count += 1

    # Get keys
    checkpoint_keys = set(state_dict.keys())

    # Get key mappings
    logger.info("Mapping keys...")
    key_mapping = {}
    for checkpoint_key in checkpoint_keys:  # pickle key
        mapped_key = map_single_key(checkpoint_key)
        key_mapping[checkpoint_key] = mapped_key

    # Apply mappings
    mapped_state_dict = {}
    for checkpoint_key, model_key in key_mapping.items():
        mapped_state_dict[model_key] = state_dict[checkpoint_key]
    del mapped_state_dict["pixel_mean"]
    del mapped_state_dict["pixel_std"]
    logger.debug(f"Mapped {len(mapped_state_dict)} weights")

    if isinstance(
        torch_model,
        (
            DeepLabStem,
            Bottleneck,
            TorchBackbone,
            ASPPModel,
            ResModel,
            HeadModel,
            DecoderModel,
        ),
    ):
        torch_model = load_partial_state(torch_model, mapped_state_dict, layer_name)
    elif isinstance(torch_model, TorchPanopticDeepLab):
        torch_model.load_state_dict(mapped_state_dict, strict=True)
    else:
        raise NotImplementedError("Unknown torch model. Weight loading not implemented")

    return torch_model.eval()


def parameter_conv_args(torch_model: torch.nn.Module = None, parameters: dict = None):
    from ttnn.model_preprocessing import infer_ttnn_module_args

    if isinstance(torch_model, TorchPanopticDeepLab):
        parameters.conv_args = {}
        sample_x = torch.randn(1, 2048, 32, 64)
        sample_res3 = torch.randn(1, 512, 64, 128)
        sample_res2 = torch.randn(1, 256, 128, 256)

        # For semantic decoder
        if hasattr(parameters, "semantic_decoder"):
            # ASPP
            aspp_args = infer_ttnn_module_args(
                model=torch_model.semantic_decoder.aspp, run_model=lambda model: model(sample_x), device=None
            )
            if hasattr(parameters.semantic_decoder, "aspp"):
                parameters.semantic_decoder.aspp.conv_args = aspp_args

            # Res3
            aspp_out = torch_model.semantic_decoder.aspp(sample_x)
            res3_args = infer_ttnn_module_args(
                model=torch_model.semantic_decoder.res3,
                run_model=lambda model: model(aspp_out, sample_res3),
                device=None,
            )
            if hasattr(parameters.semantic_decoder, "res3"):
                parameters.semantic_decoder.res3.conv_args = res3_args

            # Res2
            res3_out = torch_model.semantic_decoder.res3(aspp_out, sample_res3)
            res2_args = infer_ttnn_module_args(
                model=torch_model.semantic_decoder.res2,
                run_model=lambda model: model(res3_out, sample_res2),
                device=None,
            )
            if hasattr(parameters.semantic_decoder, "res2"):
                parameters.semantic_decoder.res2.conv_args = res2_args

            # Head
            res2_out = torch_model.semantic_decoder.res2(res3_out, sample_res2)
            head_args = infer_ttnn_module_args(
                model=torch_model.semantic_decoder.head_1, run_model=lambda model: model(res2_out), device=None
            )
            if hasattr(parameters.semantic_decoder, "head_1"):
                parameters.semantic_decoder.head_1.conv_args = head_args

        # For instance decoder
        if hasattr(parameters, "instance_decoder"):
            # ASPP
            aspp_args = infer_ttnn_module_args(
                model=torch_model.instance_decoder.aspp, run_model=lambda model: model(sample_x), device=None
            )
            if hasattr(parameters.instance_decoder, "aspp"):
                parameters.instance_decoder.aspp.conv_args = aspp_args

            # Res3
            aspp_out = torch_model.instance_decoder.aspp(sample_x)
            res3_args = infer_ttnn_module_args(
                model=torch_model.instance_decoder.res3,
                run_model=lambda model: model(aspp_out, sample_res3),
                device=None,
            )
            if hasattr(parameters.instance_decoder, "res3"):
                parameters.instance_decoder.res3.conv_args = res3_args

            # Res2
            res3_out = torch_model.instance_decoder.res3(aspp_out, sample_res3)
            res2_args = infer_ttnn_module_args(
                model=torch_model.instance_decoder.res2,
                run_model=lambda model: model(res3_out, sample_res2),
                device=None,
            )
            if hasattr(parameters.instance_decoder, "res2"):
                parameters.instance_decoder.res2.conv_args = res2_args

            # Head
            res2_out = torch_model.instance_decoder.res2(res3_out, sample_res2)
            head_args_1 = infer_ttnn_module_args(
                model=torch_model.instance_decoder.head_1, run_model=lambda model: model(res2_out), device=None
            )
            head_args_2 = infer_ttnn_module_args(
                model=torch_model.instance_decoder.head_2, run_model=lambda model: model(res2_out), device=None
            )
            if hasattr(parameters.instance_decoder, "head_1"):
                parameters.instance_decoder.head_1.conv_args = head_args_1
            if hasattr(parameters.instance_decoder, "head_2"):
                parameters.instance_decoder.head_2.conv_args = head_args_2
    else:
        raise NotImplementedError("Unknown torch model. Parameter conv args not implemented")
    return parameters


def preprocess_image(
    image_path: str, input_width: int, input_height: int, ttnn_device: ttnn.Device, inputs_mesh_mapper: Optional[Any]
) -> Tuple[torch.Tensor, ttnn.Tensor, np.ndarray, Tuple[int, int]]:
    """Preprocess image for both PyTorch and TTNN"""
    # Load image
    image = Image.open(image_path).convert("RGB")
    original_size = image.size  # (width, height)
    original_array = np.array(image)
    preprocess = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    )

    # Resize to model input size
    target_size = (input_width, input_height)  # PIL expects (width, height)
    image_resized = image.resize(target_size)

    # PyTorch preprocessing
    torch_tensor = preprocess(image_resized).unsqueeze(0)  # Add batch dimension
    torch_tensor = torch_tensor.to(torch.float)

    # TTNN preprocessing
    ttnn_tensor = None
    ttnn_tensor = ttnn.from_torch(
        torch_tensor.permute(0, 2, 3, 1),  # BCHW -> BHWC
        dtype=ttnn.bfloat16,
        device=ttnn_device,
        mesh_mapper=inputs_mesh_mapper,
    )

    if ttnn_tensor is not None:
        ttnn_as_torch = ttnn.to_torch(ttnn_tensor)

    return torch_tensor, ttnn_tensor, original_array, original_size


def save_preprocessed_inputs(torch_input: torch.Tensor, save_dir: str, filename: str):
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

    return torch_input_path
