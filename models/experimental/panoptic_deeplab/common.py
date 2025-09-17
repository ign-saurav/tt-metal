# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import os
import pickle
import ttnn
import torch
import numpy as np
import torchvision.transforms as transforms

from PIL import Image
from loguru import logger
from typing import Tuple, Optional, Any
from ttnn.model_preprocessing import infer_ttnn_module_args

from models.experimental.panoptic_deeplab.reference.resnet52_backbone import ResNet52BackBone as TorchBackbone
from models.experimental.panoptic_deeplab.reference.resnet52_stem import DeepLabStem
from models.experimental.panoptic_deeplab.reference.aspp import ASPPModel
from models.experimental.panoptic_deeplab.reference.decoder import DecoderModel
from models.experimental.panoptic_deeplab.reference.res_block import ResModel
from models.experimental.panoptic_deeplab.reference.head import HeadModel
from models.experimental.panoptic_deeplab.reference.panoptic_deeplab import TorchPanopticDeepLab
from models.experimental.panoptic_deeplab.reference.resnet52_bottleneck import Bottleneck


# ---------------------------
# Key mapping & model loading
# ---------------------------

key_mappings = {
    # Semantic head mappings
    "sem_seg_head.": "semantic_decoder.",
    ".predictor.": ".head_1.predictor.",
    ".head.pointwise.": ".head_1.conv2.",
    ".head.depthwise.": ".head_1.conv1.",
    # Instance head mappings
    "ins_embed_head.": "instance_decoder.",
    ".center_head.0.": ".head_2.conv1.",
    ".center_head.1.": ".head_2.conv2.",
    ".center_predictor.": ".head_2.predictor.",
    ".offset_head.depthwise.": ".head_1.conv1.",
    ".offset_head.pointwise.": ".head_1.conv2.",
    ".offset_predictor.": ".head_1.predictor.",
    # ASPP mappings (res5 -> aspp)
    "decoder.res5.project_conv": "aspp",
    # Decoder res3 mappings
    ".decoder.res3.": ".res3.",
    # Decoder res2 mappings
    ".decoder.res2.": ".res2.",
}


def map_single_key(checkpoint_key):
    for key, value in key_mappings.items():
        checkpoint_key = checkpoint_key.replace(key, value)
    return checkpoint_key


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


def _infer_and_set(module, params_holder, attr_name, run_fn):
    """Infer conv args for a TTNN module and set them if present in parameters."""
    if hasattr(params_holder, attr_name):
        args = infer_ttnn_module_args(model=module, run_model=run_fn, device=None)
        getattr(params_holder, attr_name).conv_args = args


def _populate_decoder(torch_dec: torch.nn.Module = None, params_dec: dict = None):
    """Warm up a single decoder (semantic or instance) to populate conv_args."""
    if not (torch_dec and params_dec):
        return

    # Synthetic tensors that match typical Panoptic-DeepLab strides
    input_tensor = torch.randn(1, 2048, 32, 64)
    res3_tensor = torch.randn(1, 512, 64, 128)
    res2_tensor = torch.randn(1, 256, 128, 256)

    # ASPP
    _infer_and_set(torch_dec.aspp, params_dec, "aspp", lambda m: m(input_tensor))
    aspp_out = torch_dec.aspp(input_tensor)

    # res3
    _infer_and_set(torch_dec.res3, params_dec, "res3", lambda m: m(aspp_out, res3_tensor))
    res3_out = torch_dec.res3(aspp_out, res3_tensor)

    # res2
    _infer_and_set(torch_dec.res2, params_dec, "res2", lambda m: m(res3_out, res2_tensor))
    res2_out = torch_dec.res2(res3_out, res2_tensor)

    # heads (one or two, if present)
    if hasattr(torch_dec, "head_1"):
        _infer_and_set(torch_dec.head_1, params_dec, "head_1", lambda m: m(res2_out))
    if hasattr(torch_dec, "head_2"):
        _infer_and_set(torch_dec.head_2, params_dec, "head_2", lambda m: m(res2_out))


def _populate_all_decoders(torch_model: torch.nn.Module = None, parameters: dict = None):
    if hasattr(parameters, "semantic_decoder"):
        _populate_decoder(torch_model.semantic_decoder, parameters.semantic_decoder)
    if hasattr(parameters, "instance_decoder"):
        _populate_decoder(torch_model.instance_decoder, parameters.instance_decoder)


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
        _ = ttnn.to_torch(ttnn_tensor)

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
