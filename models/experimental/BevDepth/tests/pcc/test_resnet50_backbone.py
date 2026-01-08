# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger
import ttnn

from models.common.utility_functions import comp_pcc
from models.experimental.BevDepth.tt.ttnn_resnet50_backbone import TtResNet50Backbone
from models.experimental.BevDepth.tt.custom_preprocessing import (
    extract_backbone_state_dict,
    fuse_batchnorm_into_conv,
    prepare_resnet_parameters,
)
from models.experimental.BevDepth.common import download_bevdepth_weights


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("height, width", [(256, 704)])
def test_resnet50_bevdepth_pcc(device, batch_size, height, width):
    from torchvision.models import resnet50

    torch.manual_seed(42)

    weights_path = download_bevdepth_weights()
    backbone_state = extract_backbone_state_dict(weights_path)
    backbone_state = fuse_batchnorm_into_conv(backbone_state)

    reference_model = resnet50(pretrained=False)

    modules_dict = dict(reference_model.named_modules())
    for name, module in list(modules_dict.items()):
        if name and isinstance(module, torch.nn.Conv2d) and module.bias is None:
            new_conv = torch.nn.Conv2d(
                module.in_channels,
                module.out_channels,
                module.kernel_size,
                module.stride,
                module.padding,
                module.dilation,
                module.groups,
                bias=True,
            )
            new_conv.weight.data = module.weight.data.clone()
            if "." in name:
                parent_name, child_name = name.rsplit(".", 1)
                parent = modules_dict[parent_name]
                setattr(parent, child_name, new_conv)
            else:
                setattr(reference_model, name, new_conv)

    reference_model.load_state_dict(backbone_state, strict=False)

    modules_dict = dict(reference_model.named_modules())
    for name, module in list(modules_dict.items()):
        if name and isinstance(module, torch.nn.BatchNorm2d):
            if "." in name:
                parent_name, child_name = name.rsplit(".", 1)
                parent = modules_dict[parent_name]
                setattr(parent, child_name, torch.nn.Identity())
            else:
                setattr(reference_model, name, torch.nn.Identity())

    reference_model.eval()

    model_config = {
        "WEIGHTS_DTYPE": ttnn.bfloat16,
        "ACTIVATIONS_DTYPE": ttnn.bfloat16,
        "MATH_FIDELITY": ttnn.MathFidelity.HiFi4,
    }

    ttnn_params = prepare_resnet_parameters(backbone_state)

    ttnn_model = TtResNet50Backbone(
        device=device,
        parameters=ttnn_params,
        batch_size=batch_size,
        model_config=model_config,
        return_intermediate=True,
        return_block_outputs=True,
    )

    torch_input = torch.randn(batch_size, 3, height, width)
    torch_input_reshaped = torch_input.permute(0, 2, 3, 1).contiguous()

    ttnn_input = ttnn.from_torch(
        torch_input_reshaped,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_input = ttnn.to_layout(ttnn_input, ttnn.TILE_LAYOUT)

    with torch.no_grad():
        x = reference_model.conv1(torch_input)
        x = reference_model.bn1(x)
        x = reference_model.relu(x)
        x = reference_model.maxpool(x)
        ref_layer1 = reference_model.layer1(x)
        ref_layer2 = reference_model.layer2(ref_layer1)
        ref_layer3 = reference_model.layer3(ref_layer2)
        ref_layer4 = reference_model.layer4(ref_layer3)

    ttnn_features = ttnn_model(ttnn_input, input_height=height, input_width=width)

    layers = {
        "layer1": ref_layer1,
        "layer2": ref_layer2,
        "layer3": ref_layer3,
        "layer4": ref_layer4,
    }

    for layer_name, ref_output in layers.items():
        ttnn_output = ttnn.to_torch(ttnn_features[layer_name])
        ttnn_output = ttnn_output.permute(0, 3, 1, 2).contiguous()

        pcc_result = comp_pcc(ref_output, ttnn_output)
        pcc_value = pcc_result[1] if isinstance(pcc_result, tuple) else pcc_result

        logger.info(f"{layer_name}: PCC = {pcc_value:.6f}")
        assert pcc_value > 0.99, f"{layer_name} PCC {pcc_value:.6f} is below threshold 0.99"
