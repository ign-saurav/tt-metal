# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from ttnn.model_preprocessing import infer_ttnn_module_args as infer_ttnn_module_args_torch
from models.experimental.transfuser.reference.bottleneck import Bottleneck as PyTorchBottleneck
from models.experimental.transfuser.tt.custom_preprocessing import create_custom_mesh_preprocessor
from models.experimental.transfuser.tt.stages import optimization_dict
from models.experimental.transfuser.tt.bottleneck import TTRegNetBottleneck
from ttnn.model_preprocessing import (
    preprocess_model_parameters,
)
from tests.ttnn.utils_for_testing import check_with_pcc
from loguru import logger

activations = {}


def hook_fn(module, inp, out):
    # inp is a tuple
    activations["se_fc2_in"] = inp[0].detach().cpu()
    activations["se_fc2_out"] = out.detach().cpu()


def fix_regnet_downsample_keys(state_dict):
    """
    Remap RegNet downsample keys:
      downsample.conv.* -> downsample.0.*
      downsample.bn.*   -> downsample.1.*

    Args:
        state_dict (dict): input state_dict

    Returns:
        dict: new state_dict with corrected keys
    """
    new_sd = {}

    for k, v in state_dict.items():
        new_k = k

        if k.startswith("downsample.conv."):
            new_k = k.replace("downsample.conv.", "downsample.0.", 1)
        elif k.startswith("downsample.bn."):
            new_k = k.replace("downsample.bn.", "downsample.1.", 1)

        new_sd[new_k] = v

    return new_sd


class TransfuserBottleneckInfra:
    def __init__(
        self,
        device,
        block_name,
        use_fallback,
        save_fc2_data,
        in_chs,
        out_chs,
        stride,
        input_size,
        stage_name,
        model_config,
    ):
        super().__init__()
        self.device = device
        self.in_chs = in_chs
        self.out_chs = out_chs
        self.stride = stride
        self.input_size = input_size
        self.stage_name = stage_name
        self.use_fallback = use_fallback
        self.model_config = model_config
        self.num_devices = device.get_num_devices()
        self.inputs_mesh_mapper, self.weights_mesh_mapper, self.output_mesh_composer = self.get_mesh_mappers(device)

        # Build reference torch model
        torch_model = PyTorchBottleneck(in_chs=in_chs, out_chs=out_chs, stride=stride, group_size=24)
        torch_model.eval()
        model_args = infer_ttnn_module_args_torch(
            model=torch_model, run_model=lambda model: model(torch.randn(self.input_size)), device=None
        )

        state_dict = torch.load("image_enc_s2_weights.pt", map_location="cpu")
        # import pdb; pdb.set_trace()
        state_dict = {
            k.replace(f"{block_name}.", "", 1): v for k, v in state_dict.items() if k.startswith(f"{block_name}.")
        }
        state_dict = fix_regnet_downsample_keys(state_dict)
        torch_model.load_state_dict(state_dict, strict=True)

        fc2_hook = torch_model.se.fc2.register_forward_hook(hook_fn)

        self.tt_input = ttnn.load_tensor(f"image_layer2_input_{block_name}.tensorbin")
        act = torch.load("captured_inputs.pt")
        self.torch_input = act[f"image_encoder.features.s2.{block_name}.conv1"]
        with torch.no_grad():
            self.torch_output = torch_model(self.torch_input)

        if save_fc2_data:
            print("dumping torch fc2 input and state_dict...")
            torch.save(activations["se_fc2_in"], "se_fc2_torch_input.pt")
            torch.save(torch_model.se.fc2.state_dict(), "se_fc2_state_dict.pt")
        fc2_hook.remove()
        parameters = preprocess_model_parameters(
            initialize_model=lambda: torch_model,
            custom_preprocessor=create_custom_mesh_preprocessor(self.weights_mesh_mapper),
            device=None,
        )
        downsample = True
        if in_chs == out_chs and stride == 1:
            downsample = False
        bottle_ratio = 1.0
        group_size = 24
        bottleneck_chs = int(round(out_chs * bottle_ratio))
        groups = bottleneck_chs // group_size

        layer_config = optimization_dict[stage_name]

        self.ttnn_model = TTRegNetBottleneck(
            device=self.device,
            parameters=parameters,
            model_args=model_args,
            model_config=self.model_config,
            stride=self.stride,
            downsample=downsample,
            groups=groups,
            layer_config=layer_config,
            use_fallback=self.use_fallback,
            torch_model=torch_model if self.use_fallback else None,
            stage_name=stage_name,
            save_fc2_data=save_fc2_data,
        )
        # Run + validate
        self.run()
        self.validate(self.model_config)

    def get_mesh_mappers(self, device):
        if device.get_num_devices() != 1:
            return (
                ttnn.ShardTensorToMesh(device, dim=0),
                None,
                ttnn.ConcatMeshToTensor(device, dim=0),
            )
        return None, None, None

    def run(self):
        self.tt_output = self.ttnn_model(self.tt_input, self.device)
        return self.tt_output

    def validate(self, model_config):
        self.tt_torch_output = ttnn.to_torch(
            self.tt_output,
            device=self.device,
            mesh_composer=self.output_mesh_composer,
        )
        expected_image_shape = self.torch_output.shape
        self.tt_torch_output = torch.reshape(
            self.tt_torch_output,
            (expected_image_shape[0], expected_image_shape[2], expected_image_shape[3], expected_image_shape[1]),
        )
        self.tt_torch_output = torch.permute(self.tt_torch_output, (0, 3, 1, 2))
        pcc_passed, pcc_message = check_with_pcc(self.torch_output, self.tt_torch_output, pcc=0.99)

        logger.info(f"Image Output PCC: {pcc_message}")
        assert pcc_passed, logger.error(f"PCC check failed - pcc_message: {pcc_message}")

        print("RegNet bottleneck TTNN implementation matches PyTorch with PCC > 0.99")

        return pcc_passed, f"Bottleneck: {pcc_message}"


# High accuracy model config
model_config = {
    "MATH_FIDELITY": ttnn.MathFidelity.HiFi4,
    "WEIGHTS_DTYPE": ttnn.bfloat16,
    "ACTIVATIONS_DTYPE": ttnn.bfloat16,
    "fp32_dest_acc_en": True,
    "packer_l1_acc": True,
    "math_approx_mode": False,
}


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize(
    "in_chs, out_chs, stride, input_size, stage_name, block_name",
    [
        # (32, 72, 2, (1, 32, 80, 352), "layer1"),  # stage 1 DS
        # (72, 72, 1, (1, 72, 80//2, 352//2), "layer1"),  # stage 1 NDS
        # (72, 216, 2, (1, 72, 80//2, 352//2), "layer2", "b1"),  # stage 2 DS b1
        (216, 216, 1, (1, 216, 80 // 4, 352 // 4), "layer2", "b2"),  # stage 2 NDS b1
        # (216, 216, 1, (1, 216, 80 // 4, 352 // 4), "layer2", "b3"),  # stage 2 NDS b1
        # (216, 216, 1, (1, 216, 80 // 4, 352 // 4), "layer2", "b4"),  # stage 2 NDS b1
        # (216, 216, 1, (1, 216, 80 // 4, 352 // 4), "layer2", "b5"),  # stage 2 NDS b1
    ],
)
@pytest.mark.parametrize("use_fallback", [False])
@pytest.mark.parametrize("save_fc2_data", [True])
def test_transfuser_bottleneck(
    device,
    block_name,
    use_fallback,
    save_fc2_data,
    in_chs,
    out_chs,
    stride,
    input_size,
    stage_name,
):
    TransfuserBottleneckInfra(
        device,
        block_name,
        use_fallback,
        save_fc2_data,
        in_chs,
        out_chs,
        stride,
        input_size,
        stage_name,
        model_config,
    )
