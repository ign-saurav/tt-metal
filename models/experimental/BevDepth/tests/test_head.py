# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger
import ttnn

from ttnn.model_preprocessing import preprocess_model_parameters, infer_ttnn_module_args
from tests.ttnn.utils_for_testing import check_with_pcc

from models.experimental.BevDepth.reference.bevdepth.exps.nuscenes.mv.bev_depth_lss_r50_256x704_128x128_24e_2key import (
    BEVDepthLightningModel,
)
from models.experimental.BevDepth.tt.bev_depth_head import TtBEVDepthHead, head_optimisations
from models.experimental.BevDepth.tt.custom_preprocessing import create_custom_mesh_preprocessor


class HeadTestInfra:
    def __init__(
        self,
        device,
        model_config,
    ):
        super().__init__()
        if not hasattr(self, "_model_initialized"):
            torch.manual_seed(42)
            torch.cuda.manual_seed_all(42)
            torch.backends.cudnn.deterministic = True
            self._model_initialized = True

        # Core config
        self.device = device
        self.num_devices = device.get_num_devices()
        self.model_config = model_config

        # Mesh mappers
        self.inputs_mesh_mapper, self.weights_mesh_mapper, self.output_mesh_composer = self.get_mesh_mappers(device)

        # Torch model
        torch_model = BEVDepthLightningModel()
        torch_model.load_checkpoint("../reference/checkpoints/bev_depth_lss_r50_256x704_128x128_24e_2key.pth")
        torch_model = torch_model.model.head
        torch_model.eval()

        # Synthetic input
        self.torch_input_tensor = self._create_input_tensor()

        # Torch output
        self.torch_output = torch_model(self.torch_input_tensor)
        print(f"Torch output type: {type(self.torch_output)}")
        print(f"length of torch output: {len(self.torch_output)}")
        print(f"type of torch output[0]: {type(self.torch_output[0])}")
        print(f"length of torch output[0]: {len(self.torch_output[0])}")
        print(f"type of torch output[0][0]: {type(self.torch_output[0][0])}")
        print(f"length of torch output[0][0]: {len(self.torch_output[0][0])}")

        # Preprocess model parameters
        parameters = preprocess_model_parameters(
            initialize_model=lambda: torch_model,
            custom_preprocessor=create_custom_mesh_preprocessor(self.weights_mesh_mapper),
            device=None,
        )
        parameters.conv_args = infer_ttnn_module_args(
            model=torch_model,
            run_model=lambda m: m(self.torch_input_tensor),
            device=None,
        )
        # print(parameters)

        # Initialize TTNN model
        self.ttnn_model = TtBEVDepthHead(parameters, model_config, layer_optimisations=head_optimisations)

        # Run model in phases and validate
        logger.info(f"Running TTNN Head model")

        # Rebuild TTNN input (since buffers may be freed across passes)
        tt_host_tensor = ttnn.from_torch(
            self.torch_input_tensor.permute(0, 2, 3, 1),
            dtype=ttnn.bfloat8_b,
            device=self.device,
            mesh_mapper=self.inputs_mesh_mapper,
        )
        self.input_tensor = ttnn.to_device(tt_host_tensor, self.device, memory_config=ttnn.L1_MEMORY_CONFIG)

        # Optional: reinstantiate TTNN model
        self.ttnn_model = TtBEVDepthHead(parameters, model_config, layer_optimisations=head_optimisations)

        self.run()
        self.validate()

    def _create_input_tensor(self):
        shape = (2, 160, 128, 128)
        logger.info(f"Generating synthetic input tensor of shape {shape}")
        return torch.randn(shape, dtype=torch.float32)

    @classmethod
    def get_mesh_mappers(cls, device):
        if device.get_num_devices() != 1:
            return (
                ttnn.ShardTensorToMesh(device, dim=0),
                None,
                ttnn.ConcatMeshToTensor(device, dim=0),
            )
        return None, None, None

    def run(self):
        logger.info("Running TTNN Head model...")
        self.ttnn_output = self.ttnn_model(self.input_tensor, self.device)
        return self.ttnn_output

    def _tt_to_torch_nchw(self, tt_tensor, expected_shape):
        torch_tensor = ttnn.to_torch(tt_tensor, device=self.device, mesh_composer=self.output_mesh_composer)
        torch_tensor = torch.reshape(
            torch_tensor,
            (expected_shape[0], expected_shape[2], expected_shape[3], expected_shape[1]),
        )
        return torch.permute(torch_tensor, (0, 3, 1, 2))

    def validate(self):
        logger.info("Validating TTNN output against PyTorch...")

        # Torch output structure: tuple of 6 lists, each list contains one dict
        # torch_output = (
        #     [{"reg": ..., "height": ..., "dim": ..., "rot": ..., "vel": ..., "heatmap": ...}],  # task head 0
        #     [...],  # task head 1
        #     ...
        # )
        # TTNN output structure: list of 6 dicts
        # ttnn_output = [
        #     {"reg": ..., "height": ..., ...},  # task head 0
        #     {...},  # task head 1
        #     ...
        # ]
        assert isinstance(
            self.torch_output, (tuple, list)
        ), f"Torch output should be a tuple or list, got {type(self.torch_output)}"
        assert isinstance(self.ttnn_output, list), "TTNN output should be a list"
        assert len(self.torch_output) == len(
            self.ttnn_output
        ), f"Output length mismatch: {len(self.torch_output)} vs {len(self.ttnn_output)}"

        pcc_threshold = 0.99
        all_passed = True
        all_messages = []

        # Iterate through each task head
        for head_idx, (torch_head_list, tt_head_output) in enumerate(zip(self.torch_output, self.ttnn_output)):
            # Torch output: each element is a list containing one dict
            assert isinstance(
                torch_head_list, list
            ), f"Torch head {head_idx} output should be a list, got {type(torch_head_list)}"
            assert (
                len(torch_head_list) == 1
            ), f"Torch head {head_idx} list should contain exactly 1 dict, got {len(torch_head_list)}"
            torch_head_output = torch_head_list[0]

            assert isinstance(
                torch_head_output, dict
            ), f"Torch head {head_idx} output[0] should be a dict, got {type(torch_head_output)}"
            assert isinstance(tt_head_output, dict), f"TTNN head {head_idx} output should be a dict"

            # Ensure both dicts have the same keys
            torch_keys = set(torch_head_output.keys())
            tt_keys = set(tt_head_output.keys())
            assert torch_keys == tt_keys, f"Head {head_idx} key mismatch: torch keys {torch_keys} vs tt keys {tt_keys}"

            # Iterate through each key in the dict
            for key in torch_head_output.keys():
                torch_tensor = torch_head_output[key]
                tt_output_tuple = tt_head_output[key]

                # TTNN output is a tuple (tensor, output_shape)
                assert (
                    isinstance(tt_output_tuple, tuple) and len(tt_output_tuple) == 2
                ), f"TTNN output for head {head_idx}, key {key} should be a tuple (tensor, shape)"

                tt_tensor, tt_output_shape = tt_output_tuple

                # Convert TTNN tensor to torch format
                tt_tensor_torch = self._tt_to_torch_nchw(tt_tensor, torch_tensor.shape)

                # Deallocate to save memory
                ttnn.deallocate(tt_tensor)

                # Evaluate PCC
                passed, msg = check_with_pcc(torch_tensor, tt_tensor_torch, pcc=pcc_threshold)

                if not passed:
                    all_passed = False
                    error_msg = f"Head {head_idx}, key '{key}' PCC check failed: {msg}"
                    logger.error(error_msg)
                    all_messages.append(error_msg)
                else:
                    logger.info(f"Head {head_idx}, key '{key}': PCC check passed : {msg}")

        assert all_passed, f"Some PCC checks failed:\n" + "\n".join(all_messages)

        logger.info(
            f"BEVDepth Head Tested: "
            f"act_dtype={self.model_config['ACTIVATIONS_DTYPE']}, "
            f"weight_dtype={self.model_config['WEIGHTS_DTYPE']}, "
            f"math_fidelity={self.model_config['MATH_FIDELITY']}, "
        )

        return True, "All PCC checks passed"


model_config = {
    "MATH_FIDELITY": ttnn.MathFidelity.LoFi,
    "WEIGHTS_DTYPE": ttnn.bfloat8_b,
    "ACTIVATIONS_DTYPE": ttnn.bfloat8_b,
}


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_head(device):
    HeadTestInfra(
        device=device,
        model_config=model_config,
    )
