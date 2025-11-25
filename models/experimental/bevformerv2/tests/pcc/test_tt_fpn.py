# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn

from types import SimpleNamespace

from models.experimental.bevformerv2.reference.fpn import FPN as RefFPN
from models.experimental.bevformerv2.reference.resnet import resnet50_mmdet, ResNet
from models.experimental.bevformerv2.tt.tt_fpn import TtFPN
from models.experimental.bevformerv2.tt.tt_resnet import TtResNet50_MMD_C345
from models.experimental.bevformerv2.utils import load_resnet50_backbone_weights
from models.experimental.bevformerv2.tt.model_configs import BevFormerV2ModelConfig
from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.model_preprocessing import (
    fold_batch_norm2d_into_conv2d,
    infer_ttnn_module_args,
    preprocess_model_parameters,
)


def _pack_conv_tensors(weight, bias):
    return {
        "weight": ttnn.from_torch(weight, dtype=ttnn.bfloat16),
        "bias": ttnn.from_torch(bias.reshape((1, 1, 1, -1)), dtype=ttnn.bfloat16),
    }


def _custom_resnet_preprocessor(model, name):
    parameters = {}
    if isinstance(model, ResNet):
        resnet_params = {}

        weight, bias = fold_batch_norm2d_into_conv2d(model.conv1, model.bn1)
        resnet_params["conv1"] = _pack_conv_tensors(weight, bias)

        for layer_idx in range(1, 5):
            layer = getattr(model, f"layer{layer_idx}")
            for block_idx, block in enumerate(layer):
                block_params = {}
                for conv_name in ["conv1", "conv2", "conv3"]:
                    conv = getattr(block, conv_name)
                    bn = getattr(block, f"bn{conv_name[-1]}")
                    weight, bias = fold_batch_norm2d_into_conv2d(conv, bn)
                    block_params[conv_name] = _pack_conv_tensors(weight, bias)

                if hasattr(block, "downsample") and block.downsample is not None:
                    conv = block.downsample[0]
                    bn = block.downsample[1]
                    weight, bias = fold_batch_norm2d_into_conv2d(conv, bn)
                    block_params["downsample"] = _pack_conv_tensors(weight, bias)

                resnet_params[f"layer{layer_idx}_{block_idx}"] = block_params

        parameters["res_model"] = resnet_params
    return parameters


def _prepare_resnet_parameters(model: ResNet, example_input: torch.Tensor, device):
    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        custom_preprocessor=_custom_resnet_preprocessor,
        device=device,
    )

    conv_args = infer_ttnn_module_args(model=model, run_model=lambda mod: mod(example_input), device=None)
    for key in conv_args.keys():
        if hasattr(model, key):
            conv_args[key].module = getattr(model, key)

    parameters.conv_args = conv_args
    return parameters


def _custom_fpn_preprocessor(model, name):
    parameters = {}
    if isinstance(model, RefFPN):
        parameters["fpn"] = {"lateral_convs": {}, "fpn_convs": {}}

        for i, l_conv in enumerate(model.lateral_convs):
            conv = l_conv.conv
            parameters["fpn"]["lateral_convs"][str(i)] = {
                "conv": {
                    "weight": ttnn.from_torch(conv.weight, dtype=ttnn.float32),
                    "bias": ttnn.from_torch(conv.bias.reshape((1, 1, 1, -1)), dtype=ttnn.float32),
                }
            }

        for i, fpn_conv in enumerate(model.fpn_convs):
            conv = fpn_conv.conv
            parameters["fpn"]["fpn_convs"][str(i)] = {
                "conv": {
                    "weight": ttnn.from_torch(conv.weight, dtype=ttnn.float32),
                    "bias": ttnn.from_torch(conv.bias.reshape((1, 1, 1, -1)), dtype=ttnn.float32),
                }
            }
    return parameters


def _make_conv_arg(conv_module, input_shape):
    conv = conv_module.conv
    batch, channels, height, width = input_shape
    return SimpleNamespace(
        conv=SimpleNamespace(
            in_channels=conv.in_channels,
            out_channels=conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
            input_height=height,
            input_width=width,
            batch_size=batch,
        )
    )


def _prepare_fpn_parameters(model: RefFPN, example_feats, example_outputs, device):
    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        custom_preprocessor=_custom_fpn_preprocessor,
        device=device,
    )
    conv_args = SimpleNamespace(lateral_convs=[], fpn_convs=[])
    weight_laterals = []
    weight_fpn = []

    # Build lateral conv args and stash activation shapes on the weight blobs for reshape.
    for idx, feat in enumerate(example_feats):
        conv_args.lateral_convs.append(_make_conv_arg(model.lateral_convs[idx], feat.shape))
        if str(idx) in parameters.fpn["lateral_convs"]:
            conv_blob = parameters.fpn["lateral_convs"][str(idx)]["conv"]
            conv_blob["height"] = feat.shape[2]
            conv_blob["width"] = feat.shape[3]
            conv_blob["batch"] = feat.shape[0]
            weight_laterals.append(SimpleNamespace(conv=SimpleNamespace(**conv_blob)))

    # Build FPN conv args using the reference outputs to derive input shapes.
    prev_shape = None
    for idx, fpn_conv in enumerate(model.fpn_convs):
        if idx < len(example_feats):
            shape = example_outputs[idx].shape
            prev_shape = shape
        else:
            shape = prev_shape
        conv_args.fpn_convs.append(_make_conv_arg(fpn_conv, shape))
        if str(idx) in parameters.fpn["fpn_convs"]:
            conv_blob = parameters.fpn["fpn_convs"][str(idx)]["conv"]
            conv_blob["height"] = shape[2]
            conv_blob["width"] = shape[3]
            conv_blob["batch"] = shape[0]
            weight_fpn.append(SimpleNamespace(conv=SimpleNamespace(**conv_blob)))

    # Convert the nested dicts into simple namespace lists so the TT FPN
    # implementation can index them directly (matching conv_args layout).
    parameters.fpn = SimpleNamespace(
        lateral_convs=weight_laterals,
        fpn_convs=weight_fpn,
    )
    parameters.conv_args = conv_args
    return parameters


@pytest.mark.parametrize("device_params", [{"l1_small_size": 10 * 1024}], indirect=True)
def test_bevformerv2_fpn_matches_reference(device, reset_seeds):
    # Reference backbone + FPN configuration (C3, C4, C5 -> P3..P6).
    backbone = resnet50_mmdet(out_indices=(1, 2, 3))
    load_resnet50_backbone_weights(backbone)
    backbone.eval()

    fpn = RefFPN(
        in_channels=[512, 1024, 2048],
        out_channels=256,
        num_outs=4,
        add_extra_convs="on_output",
    )

    torch_input = torch.randn(2, 3, 256, 256)
    with torch.no_grad():
        c_feats = backbone(torch_input)
        torch_outputs = fpn(list(c_feats))

    backbone_params = _prepare_resnet_parameters(backbone, torch_input, device)
    fpn_params = _prepare_fpn_parameters(fpn, list(c_feats), torch_outputs, device)

    # Build TTNN backbone to produce C3, C4, C5.
    nhwc = torch_input.permute(0, 2, 3, 1).contiguous()
    nhwc = nhwc.reshape(1, 1, nhwc.shape[0] * nhwc.shape[1] * nhwc.shape[2], nhwc.shape[3])
    ttnn_input = ttnn.from_torch(nhwc, dtype=ttnn.bfloat16, device=device)

    tt_backbone = TtResNet50_MMD_C345(backbone_params.conv_args, backbone_params.res_model, device)
    tt_c_feats = tt_backbone(ttnn_input, batch_size=torch_input.shape[0])

    # Prepare TTNN FPN conv args / params and run it.
    model_cfg = BevFormerV2ModelConfig()
    tt_fpn = TtFPN(fpn_params.conv_args, fpn_params.fpn, device, model_configs=model_cfg)

    ttnn_outputs = tt_fpn(list(tt_c_feats))

    # Compare each FPN level with PCC.
    for level_idx, (torch_level, tt_level) in enumerate(zip(torch_outputs, ttnn_outputs)):
        n, c, h, w = torch_level.shape
        converted = ttnn.to_torch(tt_level)
        converted = converted.reshape(n, h, w, c)
        converted = converted.permute(0, 3, 1, 2).contiguous().to(dtype=torch.float32)
        _, pcc_value = assert_with_pcc(converted, torch_level, 0.95)
        print(f"PCC(P{level_idx + 3}) = {pcc_value:.5f}")
