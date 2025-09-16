# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import os
import pickle
from typing import Tuple, Optional, Any

import numpy as np
from PIL import Image

import torch
import torchvision.transforms as transforms
from loguru import logger

import ttnn
from models.experimental.panoptic_deeplab.reference.resnet52_backbone import ResNet52BackBone as TorchBackbone
from models.experimental.panoptic_deeplab.reference.resnet52_stem import DeepLabStem
from torchvision.models.resnet import Bottleneck
from models.experimental.panoptic_deeplab.reference.aspp import ASPPModel
from models.experimental.panoptic_deeplab.reference.decoder import DecoderModel
from models.experimental.panoptic_deeplab.reference.res_block import ResModel
from models.experimental.panoptic_deeplab.reference.head import HeadModel
from models.experimental.panoptic_deeplab.reference.panoptic_deeplab import TorchPanopticDeepLab
from tests.ttnn.ttnn_utility_fuction import get_shard_grid_from_num_cores
from ttnn.model_preprocessing import infer_ttnn_module_args


# ---------------------------
# Key mapping & model loading
# ---------------------------


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


def _infer_and_set(module, params_holder, attr_name, run_fn):
    """Infer conv args for a TTNN module and set them if present in parameters."""
    if hasattr(params_holder, attr_name):
        args = infer_ttnn_module_args(model=module, run_model=run_fn, device=None)
        getattr(params_holder, attr_name).conv_args = args


def _populate_decoder(torch_dec, params_dec):
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


# ---------------------------
# TTNN utility modules
# ---------------------------


class TTConv2D:
    def __init__(
        self,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        parameters: dict | None = None,
        kernel_fidelity: dict | None = None,
        *,
        memory_config=None,
        act_block_h=None,
        act_block_w=None,
        deallocate_activation=False,
        reallocate_halo_output=False,
        shard_layout=None,
        activation=None,
        groups=1,
        num_cores_nhw=None,
        is_reshape=False,
        enable_act_double_buffer=False,
        enable_weights_double_buffer=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
        math_approx_mode=False,
        input_channels_alignment=32,
        reshard_if_not_optimal=False,
        slice_config=None,
        dtype=None,
        weights_dtype=None,
        math_fidelity=None,
    ) -> None:
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        elif isinstance(kernel_size, tuple):
            self.kernel_size = kernel_size
        else:
            ValueError("Invalid config")
        if isinstance(stride, int):
            self.stride = (stride, stride)
        elif isinstance(stride, tuple):
            self.stride = stride
        else:
            ValueError("Invalid config")
        if isinstance(padding, int):
            self.padding = (padding, padding, padding, padding)
        elif isinstance(padding, tuple):
            self.padding = padding
        else:
            ValueError("Invalid config")
        if isinstance(dilation, int):
            self.dilation = (dilation, dilation)
        elif isinstance(dilation, tuple):
            self.dilation = dilation
        else:
            ValueError("Invalid config")

        self.kernel_fidelity = kernel_fidelity
        self.weights = parameters["weight"]
        self.bias = parameters["bias"]
        self.deallocate_activation = deallocate_activation
        self.reallocate_halo_output = reallocate_halo_output
        self.fp32_dest_acc_en = fp32_dest_acc_en
        self.packer_l1_acc = packer_l1_acc
        self.math_approx_mode = math_approx_mode
        self.input_channels_alignment = input_channels_alignment
        self.reshard_if_not_optimal = reshard_if_not_optimal
        self.out_channels = self.weights.shape[0]
        self.act_block_h = act_block_h
        self.act_block_w = act_block_w
        self.groups = groups
        self.activation = activation
        self.memory_config = memory_config
        self.shard_layout = shard_layout
        self.slice_config = slice_config
        self.num_cores_nhw = num_cores_nhw
        self.is_reshape = is_reshape
        self.enable_act_double_buffer = enable_act_double_buffer
        self.enable_weights_double_buffer = enable_weights_double_buffer
        if dtype is not None:
            self.dtype = dtype
        else:
            self.dtype = self.kernel_fidelity["ACTIVATIONS_DTYPE"]
        if weights_dtype is not None:
            self.weights_dtype = weights_dtype
        else:
            self.weights_dtype = self.kernel_fidelity["WEIGHTS_DTYPE"]
        if math_fidelity is not None:
            self.math_fidelity = math_fidelity
        else:
            self.math_fidelity = self.kernel_fidelity["MATH_FIDELITY"]

    def __call__(self, device, input_tensor, input_shape):
        conv_config = ttnn.Conv2dConfig(
            weights_dtype=self.weights_dtype,
            activation=self.activation,
            deallocate_activation=self.deallocate_activation,
            reallocate_halo_output=self.reallocate_halo_output,
            reshard_if_not_optimal=self.reshard_if_not_optimal,
            shard_layout=self.shard_layout,
            enable_act_double_buffer=self.enable_act_double_buffer,
            enable_weights_double_buffer=self.enable_weights_double_buffer,
            in_place=True,
        )
        compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=self.kernel_fidelity["MATH_FIDELITY"],
            fp32_dest_acc_en=self.fp32_dest_acc_en,
            packer_l1_acc=self.packer_l1_acc,
            math_approx_mode=self.math_approx_mode,
        )
        if self.num_cores_nhw is not None:
            shard_grid = get_shard_grid_from_num_cores(self.num_cores_nhw, device)
            conv_config.core_grid = shard_grid
            conv_config.override_sharding_config = True

        if self.act_block_h is not None:
            conv_config.act_block_h_override = self.act_block_h
        if self.act_block_w is not None:
            conv_config.act_block_w_div = self.act_block_w

        [output_tensor, [_out_height, _out_width], [self.weights, self.bias]] = ttnn.conv2d(
            input_tensor=input_tensor,
            weight_tensor=self.weights,
            bias_tensor=self.bias,
            in_channels=input_shape[-1],
            out_channels=self.out_channels,
            device=device,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            batch_size=input_shape[-4],
            input_height=input_shape[-3],
            input_width=input_shape[-2],
            conv_config=conv_config,
            compute_config=compute_config,
            slice_config=self.slice_config,
            groups=self.groups,
            return_weights_and_bias=True,
            return_output_dim=True,
            dtype=self.dtype,
            memory_config=self.memory_config,
        )

        if self.is_reshape:
            output_tensor = ttnn.sharded_to_interleaved(output_tensor, ttnn.L1_MEMORY_CONFIG)
            output_tensor = ttnn.to_layout(output_tensor, ttnn.TILE_LAYOUT)
            output_tensor = ttnn.reshape(
                output_tensor, (input_tensor.shape[0], _out_height, _out_width, output_tensor.shape[-1])
            )
            output_tensor = ttnn.permute(output_tensor, (0, 3, 1, 2))
        return output_tensor, (input_tensor.shape[0], _out_height, _out_width, output_tensor.shape[-1])


class TTUpsample:
    def __init__(
        self,
        scale_factor: int = 1,
        mode: str = "bilinear",
        memory_config=ttnn.L1_MEMORY_CONFIG,
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
    ) -> None:
        self.scale_factor = scale_factor
        self.mode = mode
        self.memory_config = memory_config

        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=math_fidelity,
            math_approx_mode=math_approx_mode,
            fp32_dest_acc_en=fp32_dest_acc_en,
        )

    def __call__(
        self,
        device,
        input_tensor,
        input_shape=None,
        reshape_output=False,
        pad_ch_to_32=False,
        sent_to_dram=False,
        dtype=ttnn.bfloat8_b,
    ):
        if sent_to_dram:
            input_tensor = ttnn.sharded_to_interleaved(input_tensor, ttnn.DRAM_MEMORY_CONFIG)
        else:
            input_tensor = ttnn.sharded_to_interleaved(input_tensor, ttnn.L1_MEMORY_CONFIG)

        input_tensor = ttnn.to_layout(input_tensor, ttnn.ROW_MAJOR_LAYOUT)
        input_tensor = ttnn.reshape(input_tensor, input_shape)

        if pad_ch_to_32:
            input_tensor = ttnn.pad(input_tensor, [(0, 0), (0, 0), (0, 0), (0, 32 - input_tensor.shape[-1] % 32)], 0)

        output_tensor = ttnn.upsample(
            input_tensor,
            scale_factor=self.scale_factor,
            mode=self.mode,
            memory_config=self.memory_config,
            compute_kernel_config=self.compute_kernel_config,
        )

        if pad_ch_to_32:
            output_tensor = ttnn.slice(
                output_tensor,
                [0, 0, 0, 0],
                [output_tensor.shape[0], output_tensor.shape[1], output_tensor.shape[2], input_shape[-1]],
            )

        if reshape_output:
            output_tensor = ttnn.from_device(output_tensor)
            output_tensor = ttnn.to_dtype(output_tensor, dtype)
            output_tensor = ttnn.to_device(output_tensor, device)

            output_tensor = ttnn.reshape(
                output_tensor,
                [
                    1,
                    1,
                    output_tensor.shape[0] * output_tensor.shape[1] * output_tensor.shape[2],
                    output_tensor.shape[3],
                ],
            )

        return output_tensor
