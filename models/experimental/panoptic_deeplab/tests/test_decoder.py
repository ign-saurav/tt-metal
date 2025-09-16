# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch
from loguru import logger

import ttnn
from tests.ttnn.utils_for_testing import check_with_pcc
from ttnn.model_preprocessing import (
    preprocess_model_parameters,
    infer_ttnn_module_args,
)

from models.experimental.panoptic_deeplab.tt.decoder import (
    TTDecoder,
    decoder_layer_optimisations,
)
from models.experimental.panoptic_deeplab.tt.custom_preprocessing import (
    create_custom_mesh_preprocessor,
)
from models.experimental.panoptic_deeplab.reference.decoder import DecoderModel
from models.experimental.panoptic_deeplab.tt.common import load_torch_model_state


# -------------------------
# Deterministic seeding once
# -------------------------
class _SeedOnce:
    _done = False

    @classmethod
    def ensure(cls) -> None:
        if cls._done:
            return
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        cls._done = True


# -------------------------
# Test Infra
# -------------------------
class DecoderTestInfra:
    """Builds torch + TTNN decoder graphs, runs forward, and validates PCC.

    The flow mirrors the reference decoder with ASPP → res3 → res2 → heads.
    """

    def __init__(
        self,
        device: ttnn.Device,
        batch_size: int,
        model_config: dict,
        in_channels: int,
        res3_intermediate_channels: int,
        res2_intermediate_channels: int,
        out_channels: tuple[int, ...] | tuple[int] | int,
        upsample_channels: int,
        height: int,
        width: int,
        name: str,
    ) -> None:
        _SeedOnce.ensure()

        # --- Public-ish test fields ---
        self.device = device
        self.batch_size = batch_size
        self.in_channels = in_channels
        self.res3_intermediate_channels = res3_intermediate_channels
        self.res2_intermediate_channels = res2_intermediate_channels
        self.out_channels = out_channels
        self.upsample_channels = upsample_channels
        self.height = height
        self.width = width
        self.name = name
        self.model_config = model_config

        # PCC state
        self.pcc_passed: bool = False
        self.pcc_message: str = "call validate()?"

        # Mesh-mapping policy
        (
            self.inputs_mesh_mapper,
            self.weights_mesh_mapper,
            self.output_mesh_composer,
        ) = self._select_mesh_mappers(device)

        # ------------------------
        # Build reference (Torch)
        # ------------------------
        self.torch_input_tensor = torch.randn((batch_size, in_channels, height, width), dtype=torch.float)
        self.torch_res3_tensor = torch.randn((batch_size, 512, height * 2, width * 2), dtype=torch.float)
        self.torch_res2_tensor = torch.randn((batch_size, upsample_channels, height * 4, width * 4), dtype=torch.float)

        torch_model = DecoderModel(name)
        torch_model = load_torch_model_state(torch_model, name)

        # Preprocess weights w/ mesh-aware custom preprocessor
        parameters = preprocess_model_parameters(
            initialize_model=lambda: torch_model,
            custom_preprocessor=create_custom_mesh_preprocessor(self.weights_mesh_mapper),
            device=None,
        )

        # Infer per-submodule conv args by briefly running torch subgraphs
        # (only when the corresponding params exist)
        self._maybe_infer_and_attach(torch_model.aspp, "aspp", parameters, run=lambda m: m(self.torch_input_tensor))

        res3_input = torch_model.aspp(self.torch_input_tensor)
        self._maybe_infer_and_attach(
            torch_model.res3,
            "res3",
            parameters,
            run=lambda m: m(res3_input, self.torch_res3_tensor),
        )

        res2_input = torch_model.res3(res3_input, self.torch_res3_tensor)
        self._maybe_infer_and_attach(
            torch_model.res2,
            "res2",
            parameters,
            run=lambda m: m(res2_input, self.torch_res2_tensor),
        )

        head_input = torch_model.res2(res2_input, self.torch_res2_tensor)
        self._maybe_infer_and_attach(
            torch_model.head_1,
            "head_1",
            parameters,
            run=lambda m: m(head_input),
        )

        if "instance" in name:
            self._maybe_infer_and_attach(
                torch_model.head_2,
                "head_2",
                parameters,
                run=lambda m: m(head_input),
            )

        # Golden outputs
        (
            self.torch_output_tensor,
            self.torch_output_tensor_2,
        ) = torch_model(self.torch_input_tensor, self.torch_res3_tensor, self.torch_res2_tensor)

        # ------------------------
        # Build device (TTNN)
        # ------------------------
        tt_in = self._torch_to_ttnn_host(self.torch_input_tensor)
        tt_res3 = self._torch_to_ttnn_host(self.torch_res3_tensor)
        tt_res2 = self._torch_to_ttnn_host(self.torch_res2_tensor)

        self.input_tensor = ttnn.to_device(tt_in, device)
        self.res3_tensor = ttnn.to_device(tt_res3, device)
        self.res2_tensor = ttnn.to_device(tt_res2, device)

        logger.info("Initializing TTNN model…")
        self.ttnn_model = TTDecoder(
            parameters,
            model_config,
            layer_optimisations=decoder_layer_optimisations[name],
            name=name,
        )

        # Eager run + validate (keeps test signature identical)
        self.run()
        self.validate()

    # -------------------------
    # Helpers
    # -------------------------
    def _select_mesh_mappers(self, device: ttnn.Device):
        if device.get_num_devices() != 1:
            inputs_mesh_mapper = ttnn.ShardTensorToMesh(device, dim=0)
            weights_mesh_mapper = None
            output_mesh_composer = ttnn.ConcatMeshToTensor(device, dim=0)
        else:
            inputs_mesh_mapper = None
            weights_mesh_mapper = None
            output_mesh_composer = None
        return inputs_mesh_mapper, weights_mesh_mapper, output_mesh_composer

    def _torch_to_ttnn_host(self, tensor: torch.Tensor) -> ttnn.Tensor:
        # NHWC + dtype conversion for TTNN host tensor
        return ttnn.from_torch(
            tensor.permute(0, 2, 3, 1),
            dtype=ttnn.bfloat8_b,
            device=self.device,
            mesh_mapper=self.inputs_mesh_mapper,
        )

    def _ttnn_to_torch(self, tensor: ttnn.Tensor, expected: torch.Size) -> torch.Tensor:
        # Compose, reshape to N H W C, then back to N C H W
        x = ttnn.to_torch(tensor, device=self.device, mesh_composer=self.output_mesh_composer)
        x = torch.reshape(x, (expected[0], expected[2], expected[3], expected[1]))
        return torch.permute(x, (0, 3, 1, 2))

    def _maybe_infer_and_attach(self, module, attr_name: str, parameters, run):
        if hasattr(parameters, attr_name):
            args = infer_ttnn_module_args(model=module, run_model=run, device=None)
            getattr(parameters, attr_name).conv_args = args

    # -------------------------
    # Execution + Validation
    # -------------------------
    def run(self):
        self.output_tensor, self.output_tensor_2 = self.ttnn_model(
            self.input_tensor,
            self.res3_tensor,
            self.res2_tensor,
            self.upsample_channels,
            self.device,
        )
        return self.output_tensor, self.output_tensor_2

    def validate(self, output_tensor: ttnn.Tensor | None = None):
        # --- Head 1 ---
        out = self.output_tensor if output_tensor is None else output_tensor
        out_torch = self._ttnn_to_torch(out, expected=self.torch_output_tensor.shape)

        valid_pcc = 0.99
        self.pcc_passed, self.pcc_message = check_with_pcc(self.torch_output_tensor, out_torch, pcc=valid_pcc)
        assert self.pcc_passed, logger.error(f"PCC check failed: {self.pcc_message}")

        if "instance" in self.name:
            head_name = "Instance Offset Head"
        else:
            head_name = "Semantic Head"

        logger.info(
            f"{head_name}, batch_size={out_torch.shape[0]}, "
            f"act_dtype={self.model_config['ACTIVATIONS_DTYPE']}, "
            f"weight_dtype={self.model_config['WEIGHTS_DTYPE']}, "
            f"math_fidelity={self.model_config['MATH_FIDELITY']}, PCC={self.pcc_message}, Shape={self.output_tensor.shape}"
        )

        # --- Head 2 (instance head only) ---
        if "instance" in self.name:
            out2_torch = self._ttnn_to_torch(self.output_tensor_2, expected=self.torch_output_tensor_2.shape)

            self.pcc_passed, self.pcc_message = check_with_pcc(self.torch_output_tensor_2, out2_torch, pcc=valid_pcc)
            assert self.pcc_passed, logger.error(f"PCC check failed: {self.pcc_message}")

            logger.info(
                f"Instance Center Head, batch_size={out2_torch.shape[0]}, "
                f"act_dtype={self.model_config['ACTIVATIONS_DTYPE']}, "
                f"weight_dtype={self.model_config['WEIGHTS_DTYPE']}, "
                f"math_fidelity={self.model_config['MATH_FIDELITY']}, PCC={self.pcc_message}, Shape={self.output_tensor_2.shape}"
            )

        return self.pcc_passed, self.pcc_message


# -------------------------
# Test parameters & entry
# -------------------------
model_config = {
    "MATH_FIDELITY": ttnn.MathFidelity.LoFi,
    "WEIGHTS_DTYPE": ttnn.bfloat8_b,
    "ACTIVATIONS_DTYPE": ttnn.bfloat8_b,
}


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, in_channels, res3_intermediate_channels, res2_intermediate_channels, out_channels, "
    "upsample_channels, height, width, name",
    [
        # semantic head
        (1, 2048, 320, 288, (19,), 256, 32, 64, "semantic_decoder"),
        # instance offset head
        (1, 2048, 320, 160, (2, 1), 256, 32, 64, "instance_decoder"),
    ],
)
def test_decoder(
    device,
    batch_size,
    in_channels,
    res3_intermediate_channels,
    res2_intermediate_channels,
    out_channels,
    upsample_channels,
    height,
    width,
    name,
):
    DecoderTestInfra(
        device=device,
        batch_size=batch_size,
        model_config=model_config,
        in_channels=in_channels,
        res3_intermediate_channels=res3_intermediate_channels,
        res2_intermediate_channels=res2_intermediate_channels,
        out_channels=out_channels,
        upsample_channels=upsample_channels,
        height=height,
        width=width,
        name=name,
    )
