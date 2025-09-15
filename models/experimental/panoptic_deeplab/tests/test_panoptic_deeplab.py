# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger
import ttnn
from ttnn.model_preprocessing import preprocess_model_parameters
from tests.ttnn.utils_for_testing import check_with_pcc
import os
import numpy as np
from models.experimental.panoptic_deeplab.reference.panoptic_deeplab import TorchPanopticDeepLab
from models.experimental.panoptic_deeplab.tt.panoptic_deeplab import TTPanopticDeepLab
from models.experimental.panoptic_deeplab.tt.custom_preprocessing import create_custom_mesh_preprocessor
from ttnn.model_preprocessing import preprocess_model_parameters
from models.experimental.panoptic_deeplab.common import load_torch_model_state
from models.experimental.panoptic_deeplab.common import parameter_conv_args


class PanopticDeepLabTestInfra:
    def __init__(
        self,
        device,
        batch_size,
        in_channels,
        height,
        width,
        model_config,
        real_input_path=None,
    ):
        super().__init__()
        if not hasattr(self, "_model_initialized"):
            torch.manual_seed(42)
            self._model_initialized = True
            torch.cuda.manual_seed_all(42)
            torch.backends.cudnn.deterministic = True

        self.pcc_passed = False
        self.pcc_message = "call validate()?"
        self.device = device
        self.num_devices = device.get_num_devices()
        self.batch_size = batch_size
        self.in_channels = in_channels
        self.height = height
        self.width = width
        self.inputs_mesh_mapper, self.weights_mesh_mapper, self.output_mesh_composer = self.get_mesh_mappers(device)
        self.real_input_path = real_input_path
        # Initialize torch model
        torch_model = TorchPanopticDeepLab()
        torch_model = load_torch_model_state(torch_model, "panoptic_deeplab")

        # Create input tensor
        if self.real_input_path and os.path.exists(self.real_input_path):
            logger.info(f"Loading real input from: {self.real_input_path}")
            self.torch_input_tensor = self.load_real_input(self.real_input_path)

            # Verify shape matches expected dimensions
            expected_shape = (batch_size * self.num_devices, in_channels, height, width)
            if self.torch_input_tensor.shape != expected_shape:
                logger.warning(
                    f"Input shape mismatch. Expected: {expected_shape}, Got: {self.torch_input_tensor.shape}"
                )
        else:
            logger.info("Using random input tensor (no real input provided)")
            input_shape = (batch_size * self.num_devices, in_channels, height, width)
            self.torch_input_tensor = torch.rand(input_shape, dtype=torch.float32)

        # Preprocess model parameters
        parameters = preprocess_model_parameters(
            initialize_model=lambda: torch_model,
            custom_preprocessor=create_custom_mesh_preprocessor(self.weights_mesh_mapper),
            device=None,
        )

        parameters = parameter_conv_args(torch_model, parameters)

        # Run torch model with bfloat16
        logger.info("Running PyTorch model...")
        self.torch_output_tensor, self.torch_output_tensor_2, self.torch_output_tensor_3 = torch_model(
            self.torch_input_tensor
        )

        # Convert input to TTNN format (NHWC)
        logger.info("Converting input to TTNN format...")
        tt_host_tensor = ttnn.from_torch(
            self.torch_input_tensor.permute(0, 2, 3, 1),
            dtype=ttnn.bfloat16,
            mesh_mapper=self.inputs_mesh_mapper,
        )

        # Initialize TTNN model
        logger.info("Initializing TTNN model...")
        print("Initializing TTNN model...")
        self.ttnn_model = TTPanopticDeepLab(
            parameters=parameters,
            model_config=model_config,
        )

        logger.info("Running first TTNN model pass (JIT configuration)...")
        # first run configures convs JIT
        self.input_tensor = ttnn.to_device(tt_host_tensor, device)
        self.run()
        self.validate()

        logger.info("Running optimized TTNN model pass...")
        # Optimized run
        self.input_tensor = ttnn.to_device(tt_host_tensor, device)
        self.run()
        self.validate()

    def load_real_input(self, input_path: str) -> torch.Tensor:
        """Load real input from saved file"""

        if input_path.endswith(".pt"):
            # Load PyTorch tensor
            data = torch.load(input_path, map_location="cpu")
            if isinstance(data, dict):
                tensor = data["tensor"]
                logger.info(f"Loaded input metadata: {data.keys()}")
                if "stats" in data:
                    logger.info(f"Original input stats: {data['stats']}")
            else:
                tensor = data
        elif input_path.endswith(".npy"):
            # Load numpy array
            np_array = np.load(input_path)
            tensor = torch.from_numpy(np_array)
        else:
            raise ValueError(f"Unsupported input file format: {input_path}")

        # Ensure tensor is float32
        if tensor.dtype != torch.float32:
            tensor = tensor.to(torch.float32)

        return tensor

    def get_mesh_mappers(self, device):
        if device.get_num_devices() != 1:
            inputs_mesh_mapper = ttnn.ShardTensorToMesh(device, dim=0)
            weights_mesh_mapper = None
            output_mesh_composer = ttnn.ConcatMeshToTensor(device, dim=0)
        else:
            inputs_mesh_mapper = None
            weights_mesh_mapper = None
            output_mesh_composer = None
        return inputs_mesh_mapper, weights_mesh_mapper, output_mesh_composer

    def run(self):
        self.output_tensor, self.output_tensor_2, self.output_tensor_3 = self.ttnn_model(self.input_tensor, self.device)
        return self.output_tensor, self.output_tensor_2, self.output_tensor_3

    def validate(self, output_tensor=None):
        output_tensor = self.output_tensor if output_tensor is None else output_tensor
        output_tensor = ttnn.to_torch(output_tensor, device=self.device, mesh_composer=self.output_mesh_composer)
        expected_shape = self.torch_output_tensor.shape
        output_tensor = torch.reshape(
            output_tensor, (expected_shape[0], expected_shape[2], expected_shape[3], expected_shape[1])
        )
        output_tensor = torch.permute(output_tensor, (0, 3, 1, 2))

        batch_size = output_tensor.shape[0]

        valid_pcc = 0.97
        self.pcc_passed, self.pcc_message = check_with_pcc(self.torch_output_tensor, output_tensor, pcc=valid_pcc)
        assert self.pcc_passed, logger.error(f"Semantic Segmentation Head PCC check failed: {self.pcc_message}")
        logger.info(
            f"Panoptic DeepLab - Semantic Segmentation Head: batch_size={self.batch_size}, "
            f"act_dtype={model_config['ACTIVATIONS_DTYPE']}, weight_dtype={model_config['WEIGHTS_DTYPE']}, "
            f"math_fidelity={model_config['MATH_FIDELITY']}, PCC={self.pcc_message}, shape={self.output_tensor.shape}"
        )

        # Validate instance segmentation head outputs
        output_tensor = self.output_tensor_2
        output_tensor = ttnn.to_torch(output_tensor, device=self.device, mesh_composer=self.output_mesh_composer)
        expected_shape = self.torch_output_tensor_2.shape
        output_tensor = torch.reshape(
            output_tensor, (expected_shape[0], expected_shape[2], expected_shape[3], expected_shape[1])
        )
        output_tensor = torch.permute(output_tensor, (0, 3, 1, 2))

        batch_size = output_tensor.shape[0]

        valid_pcc = 0.97
        self.pcc_passed, self.pcc_message = check_with_pcc(self.torch_output_tensor_2, output_tensor, pcc=valid_pcc)
        assert self.pcc_passed, logger.error(f"Instance Segmentation Head PCC check failed: {self.pcc_message}")
        logger.info(
            f"Panoptic DeepLab - Instance Segmentation Offset Head: batch_size={self.batch_size}, "
            f"act_dtype={model_config['ACTIVATIONS_DTYPE']}, weight_dtype={model_config['WEIGHTS_DTYPE']}, "
            f"math_fidelity={model_config['MATH_FIDELITY']}, PCC={self.pcc_message}, shape={self.output_tensor_2.shape}"
        )

        output_tensor = self.output_tensor_3
        output_tensor = ttnn.to_torch(output_tensor, device=self.device, mesh_composer=self.output_mesh_composer)
        expected_shape = self.torch_output_tensor_3.shape
        output_tensor = torch.reshape(
            output_tensor, (expected_shape[0], expected_shape[2], expected_shape[3], expected_shape[1])
        )
        output_tensor = torch.permute(output_tensor, (0, 3, 1, 2))

        batch_size = output_tensor.shape[0]

        valid_pcc = 0.97
        self.pcc_passed, self.pcc_message = check_with_pcc(self.torch_output_tensor_3, output_tensor, pcc=valid_pcc)
        assert self.pcc_passed, logger.error(f"Instance Segmentation Head 2 PCC check failed: {self.pcc_message}")
        logger.info(
            f"Panoptic DeepLab - Instance Segmentation Center Head: batch_size={self.batch_size}, "
            f"act_dtype={model_config['ACTIVATIONS_DTYPE']}, weight_dtype={model_config['WEIGHTS_DTYPE']}, "
            f"math_fidelity={model_config['MATH_FIDELITY']}, PCC={self.pcc_message}, shape={self.output_tensor_3.shape}"
        )

        return self.pcc_passed, self.pcc_message


model_config = {
    "MATH_FIDELITY": ttnn.MathFidelity.LoFi,
    "WEIGHTS_DTYPE": ttnn.bfloat8_b,
    "ACTIVATIONS_DTYPE": ttnn.bfloat8_b,
}


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, in_channels, height, width, real_input_path",
    [
        # (1, 3, 512, 1024),
        (
            1,
            3,
            512,
            1024,
            "/home/ubuntu/ign-tt-sm/tt-metal/models/experimental/panoptic_deeplab/result/fullnet/test_inputs/frankfurt_000000_005543_leftImg8bit_torch_input.pt",
        ),
    ],
)
def test_panoptic_deeplab(
    device,
    batch_size,
    in_channels,
    height,
    width,
    real_input_path,
):
    PanopticDeepLabTestInfra(
        device,
        batch_size,
        in_channels,
        height,
        width,
        model_config,
        real_input_path,
    )
