# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from loguru import logger
from types import SimpleNamespace

from models.experimental.bevformerv2.reference.fpn import FPN as RefFPN
from models.experimental.bevformerv2.reference.resnet import resnet50_mmdet, ResNet
from models.experimental.bevformerv2.tt.tt_fpn import TtFPN
from models.experimental.bevformerv2.tt.tt_resnet import TtResNet50_MMD_C345
from models.experimental.bevformerv2.common import load_resnet50_backbone_weights, load_fpn_weights
from models.experimental.bevformerv2.tt.model_configs import BevFormerV2ModelConfig
from tests.ttnn.utils_for_testing import check_with_pcc
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
    if not example_outputs:
        raise ValueError("example_outputs must contain at least one tensor to infer FPN shapes.")

    num_lateral = len(example_feats)
    for idx, fpn_conv in enumerate(model.fpn_convs):
        if idx < num_lateral:
            shape = example_outputs[idx].shape
        else:
            prev_idx = max(idx - 1, 0)
            shape = example_outputs[prev_idx].shape
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


class FPNTestInfra:
    def __init__(
        self,
        device,
        batch_size,
        in_channels,
        height,
        width,
        use_pretrained_weights=True,
        use_backbone=True,
    ):
        super().__init__()
        if not hasattr(self, "_model_initialized"):
            torch.manual_seed(42)  # Seed once for determinism
            self._model_initialized = True
            torch.cuda.manual_seed_all(42)
            torch.backends.cudnn.deterministic = True

        self.pcc_passed_all = []
        self.pcc_message_all = []
        self.device = device
        self.batch_size = batch_size
        self.use_pretrained_weights = use_pretrained_weights
        self.use_backbone = use_backbone

        # Reference FPN
        fpn = RefFPN(
            in_channels=[512, 1024, 2048],
            out_channels=256,
            num_outs=5,
            add_extra_convs="on_output",
        )
        if use_pretrained_weights:
            load_fpn_weights(fpn)
        fpn.eval()

        # Generate input features
        if use_backbone:
            # Use backbone to generate real features
            backbone = resnet50_mmdet(out_indices=(1, 2, 3))
            load_resnet50_backbone_weights(backbone)
            backbone.eval()

            torch_input = torch.randn(batch_size, in_channels, height, width)
            with torch.no_grad():
                c_feats = backbone(torch_input)
                self.torch_outputs = fpn(list(c_feats))

            # Prepare backbone parameters
            backbone_params = _prepare_resnet_parameters(backbone, torch_input, device)

            # Build TTNN backbone
            nhwc = torch_input.permute(0, 2, 3, 1).contiguous()
            nhwc = nhwc.reshape(1, 1, nhwc.shape[0] * nhwc.shape[1] * nhwc.shape[2], nhwc.shape[3])
            ttnn_input = ttnn.from_torch(nhwc, dtype=ttnn.bfloat16, device=device)

            tt_backbone = TtResNet50_MMD_C345(backbone_params.conv_args, backbone_params.res_model, device)
            self.tt_c_feats = tt_backbone(ttnn_input, batch_size=torch_input.shape[0])
            c_feats_list = list(c_feats)  # Convert tuple to list for FPN parameter preparation
        else:
            # Use synthetic features
            c_feats = [
                torch.randn(batch_size, 512, 32, 32),  # C3
                torch.randn(batch_size, 1024, 16, 16),  # C4
                torch.randn(batch_size, 2048, 8, 8),  # C5
            ]
            with torch.no_grad():
                self.torch_outputs = fpn(list(c_feats))

            # Convert input features to TTNN format
            self.tt_c_feats = []
            for feat in c_feats:
                nhwc = feat.permute(0, 2, 3, 1).contiguous()
                nhwc = nhwc.reshape(1, 1, nhwc.shape[0] * nhwc.shape[1] * nhwc.shape[2], nhwc.shape[3])
                ttnn_feat = ttnn.from_torch(nhwc, dtype=ttnn.bfloat16, device=device)
                self.tt_c_feats.append(ttnn_feat)
            c_feats_list = c_feats

        # Prepare FPN parameters
        fpn_params = _prepare_fpn_parameters(fpn, c_feats_list, self.torch_outputs, device)

        # Build TTNN FPN
        model_cfg = BevFormerV2ModelConfig()
        self.ttnn_model = TtFPN(fpn_params.conv_args, fpn_params.fpn, device, model_configs=model_cfg)

        # Run + validate
        self.run()
        self.validate()

    def run(self):
        self.tt_outputs = self.ttnn_model(list(self.tt_c_feats))
        return self.tt_outputs

    def validate(self, tt_outputs=None):
        tt_outputs = self.tt_outputs if tt_outputs is None else tt_outputs

        assert len(self.torch_outputs) == len(
            tt_outputs
        ), f"Mismatch between reference ({len(self.torch_outputs)}) and TTNN ({len(tt_outputs)}) FPN levels"

        valid_pcc = 0.99
        for level_idx, (torch_level, tt_level) in enumerate(zip(self.torch_outputs, tt_outputs)):
            n, c, h, w = torch_level.shape
            converted = ttnn.to_torch(tt_level)
            converted = converted.reshape(n, h, w, c)
            converted = converted.permute(0, 3, 1, 2).contiguous().to(dtype=torch.float32)

            # Free device memory
            ttnn.deallocate(tt_level)

            pcc_passed, pcc_message = check_with_pcc(converted, torch_level, pcc=valid_pcc)
            self.pcc_passed_all.append(pcc_passed)
            self.pcc_message_all.append(pcc_message)

            assert pcc_passed, logger.error(f"PCC check failed for P{level_idx + 3}: {pcc_message}")
            logger.info(f"PCC(P{level_idx + 3}) = {pcc_message}")

        assert all(self.pcc_passed_all), logger.error(f"PCC check failed: {self.pcc_message_all}")
        logger.info(
            f"FPN passed: "
            f"batch_size={self.batch_size}, "
            f"use_pretrained_weights={self.use_pretrained_weights}, "
            f"use_backbone={self.use_backbone}, "
            f"PCC={self.pcc_message_all}"
        )

        return self.pcc_passed_all, self.pcc_message_all


@pytest.mark.parametrize("device_params", [{"l1_small_size": 10 * 1024}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, in_channels, height, width",
    [
        (2, 3, 256, 256),
    ],
)
def test_bevformerv2_fpn_matches_reference(device, reset_seeds, batch_size, in_channels, height, width):
    """Test FPN with backbone and pretrained weights."""
    FPNTestInfra(
        device,
        batch_size,
        in_channels,
        height,
        width,
        use_pretrained_weights=True,
        use_backbone=True,
    )


@pytest.mark.parametrize("device_params", [{"l1_small_size": 10 * 1024}], indirect=True)
@pytest.mark.parametrize(
    "batch_size",
    [
        2,
    ],
)
def test_bevformerv2_fpn_pretrained_weights(device, reset_seeds, batch_size):
    """
    Test FPN only with pretrained weights (no backbone).
    Uses random input features to simulate backbone outputs.
    """
    FPNTestInfra(
        device,
        batch_size,
        in_channels=3,  # Not used when use_backbone=False
        height=256,  # Not used when use_backbone=False
        width=256,  # Not used when use_backbone=False
        use_pretrained_weights=True,
        use_backbone=False,
    )


@pytest.mark.parametrize("device_params", [{"l1_small_size": 10 * 1024}], indirect=True)
@pytest.mark.parametrize(
    "batch_size",
    [
        2,
    ],
)
def test_bevformerv2_fpn_random_weights(device, reset_seeds, batch_size):
    """
    Test FPN only with random weights (no pretrained weights).
    Uses random input features to simulate backbone outputs.
    """
    FPNTestInfra(
        device,
        batch_size,
        in_channels=3,  # Not used when use_backbone=False
        height=256,  # Not used when use_backbone=False
        width=256,  # Not used when use_backbone=False
        use_pretrained_weights=False,
        use_backbone=False,
    )
