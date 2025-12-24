# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from loguru import logger
import ttnn
from ttnn.model_preprocessing import preprocess_model_parameters, infer_ttnn_module_args
from models.common.utility_functions import comp_pcc, comp_allclose
from models.experimental.detr3d.common import load_torch_model_state
from models.experimental.detr3d.ttnn.model_3detr import build_ttnn_3detr
from models.experimental.detr3d.reference.model_3detr import build_3detr
from models.experimental.detr3d.reference.model_config import Detr3dArgs
from models.experimental.detr3d.reference.utils.dataset import SunrgbdDatasetConfig
from models.experimental.detr3d.ttnn.custom_preprocessing import create_custom_mesh_preprocessor
from models.experimental.detr3d.ttnn.constant import ON_DEVICE


class Tt3DetrArgs(Detr3dArgs):
    def __init__(self):
        self.parameters = None
        self.device = None


class Detr3dPerformanceRunnerInfra:
    def __init__(
        self,
        device,
        batch_size=1,
        model_location_generator=None,
        input_shape=(1, 20000, 3),
        encoder_only=False,
        torch_input_dict=None,
        inputs_mesh_mapper=None,
        weights_mesh_mapper=None,
        outputs_mesh_composer=None,
        load_real_input=True,
    ):
        torch.manual_seed(0)
        self.device = device
        self.batch_size = batch_size
        self.num_devices = device.get_num_devices()
        self.input_shape = input_shape
        self.encoder_only = encoder_only
        self.pcc_passed = False
        self.pcc_message = "Did you forget to call validate()?"
        self.model_location_generator = model_location_generator

        # Mesh mappers for multi-device support
        self.inputs_mesh_mapper = inputs_mesh_mapper
        self.weights_mesh_mapper = weights_mesh_mapper
        self.outputs_mesh_composer = outputs_mesh_composer

        # PCC threshold
        self.PCC_THRESHOLD = 0.92

        # Initialize models and inputs
        self._setup_models_and_inputs(torch_input_dict, load_real_input)

    def _setup_models_and_inputs(self, torch_input_dict, load_real_input):
        """Initialize reference and TTNN models with inputs"""
        args = Detr3dArgs()
        dataset_config = SunrgbdDatasetConfig()

        # Setup input dictionary
        if load_real_input and torch_input_dict is None:
            self.torch_input_dict = torch.load("models/experimental/detr3d/resources/inputs.pt", map_location="cpu")
            logger.info("REAL INPUTS LOADED")
        elif torch_input_dict is not None:
            self.torch_input_dict = torch_input_dict
        else:
            # Generate synthetic input
            min_val = -1.8827
            max_val = 8.3542
            pc = (max_val - min_val) * torch.rand(self.input_shape) + min_val
            self.torch_input_dict = {
                "point_clouds": pc,
                "point_cloud_dims_min": torch.min(pc, 1)[0],
                "point_cloud_dims_max": torch.max(pc, 1)[0],
            }

        # Build and load reference model
        self.ref_module, _ = build_3detr(args, dataset_config)
        load_torch_model_state(self.ref_module)

        # Get reference outputs
        self._get_reference_outputs()

        # Setup TTNN model parameters
        self.ref_module_parameters = preprocess_model_parameters(
            initialize_model=lambda: self.ref_module,
            custom_preprocessor=create_custom_mesh_preprocessor(self.weights_mesh_mapper),
            device=self.device,
        )
        self.ref_module_parameters.layer_args = {}
        self.ref_module_parameters.layer_args = infer_ttnn_module_args(
            model=self.ref_module,
            run_model=lambda model: self.ref_module(inputs=self.torch_input_dict, encoder_only=self.encoder_only),
            device=self.device,
        )

        # Build TTNN model
        ttnn_args = Tt3DetrArgs()
        ttnn_args.parameters = self.ref_module_parameters
        ttnn_args.device = self.device
        self.ttnn_module, _ = build_ttnn_3detr(ttnn_args, dataset_config)

    def _get_reference_outputs(self):
        """Get outputs from reference model"""
        (
            self.torch_cls_logits,
            self.torch_center_offset,
            self.torch_size_normalized,
            self.torch_angle_logits,
            self.torch_angle_residual_normalized,
            self.torch_angle_residual,
            self.torch_num_layers,
            self.torch_query_xyz,
            self.torch_point_cloud_dims,
        ) = self.ref_module(inputs=self.torch_input_dict, encoder_only=self.encoder_only)

    def setup_device_input(self, input_dict=None):
        """Convert input dict to TTNN tensors on device"""
        input_dict = self.torch_input_dict if input_dict is None else input_dict

        if ON_DEVICE:
            ttnn_dict = {}
            for key, value in input_dict.items():
                if isinstance(value, torch.Tensor):
                    ttnn_dict[key] = ttnn.from_torch(
                        value,
                        dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT,
                        # device=self.device,
                        mesh_mapper=self.inputs_mesh_mapper,
                    )
                else:
                    ttnn_dict[key] = value
            return ttnn_dict
        else:
            return input_dict

    def run(self, input_dict=None):
        """Run inference on TTNN model"""
        ttnn_dict = self.setup_device_input(input_dict)

        (
            self.cls_logits,
            self.center_offset,
            self.size_normalized,
            self.angle_logits,
            self.angle_residual_normalized,
            self.angle_residual,
            self.num_layers,
            self.ttnn_query_xyz,
            self.ttnn_point_cloud_dims,
        ) = self.ttnn_module(inputs=ttnn_dict, encoder_only=self.encoder_only)

    def validate(self, tt_outputs=None):
        """Validate TTNN outputs against reference"""
        if tt_outputs is None:
            # Convert outputs to torch for comparison
            cls_logits = ttnn.to_torch(self.cls_logits, mesh_composer=self.outputs_mesh_composer)
            center_offset = ttnn.to_torch(self.center_offset, mesh_composer=self.outputs_mesh_composer)
            size_normalized = ttnn.to_torch(self.size_normalized, mesh_composer=self.outputs_mesh_composer)
            angle_logits = ttnn.to_torch(self.angle_logits, mesh_composer=self.outputs_mesh_composer)
            angle_residual_normalized = ttnn.to_torch(
                self.angle_residual_normalized, mesh_composer=self.outputs_mesh_composer
            )
            angle_residual = ttnn.to_torch(self.angle_residual, mesh_composer=self.outputs_mesh_composer)

            if ON_DEVICE:
                ttnn_query_xyz = ttnn.to_torch(self.ttnn_query_xyz, mesh_composer=self.outputs_mesh_composer)
                ttnn_point_cloud_dims = []
                for i in range(len(self.ttnn_point_cloud_dims)):
                    ttnn_point_cloud_dims.append(
                        ttnn.to_torch(self.ttnn_point_cloud_dims[i], mesh_composer=self.outputs_mesh_composer)
                    )
        else:
            # Use provided outputs
            (
                cls_logits,
                center_offset,
                size_normalized,
                angle_logits,
                angle_residual_normalized,
                angle_residual,
                ttnn_query_xyz,
                ttnn_point_cloud_dims,
            ) = tt_outputs

        all_passing = True

        # Compare raw outputs
        raw_outputs_pairs = [
            ("cls_logits", self.torch_cls_logits, cls_logits),
            ("center_offset", self.torch_center_offset, center_offset),
            ("size_normalized", self.torch_size_normalized, size_normalized),
            ("angle_logits", self.torch_angle_logits, angle_logits),
            ("angle_residual_normalized", self.torch_angle_residual_normalized, angle_residual_normalized),
            ("ttnn_query_xyz", self.torch_query_xyz, ttnn_query_xyz),
            ("angle_residual", self.torch_angle_residual, angle_residual),
        ]

        for name, ref_output, tt_output in raw_outputs_pairs:
            passing, pcc_message = comp_pcc(ref_output, tt_output, self.PCC_THRESHOLD)
            logger.info(f"Raw Output '{name}' PCC: {pcc_message}")
            logger.info(comp_allclose(ref_output, tt_output))

            if passing:
                logger.info(f"Raw Output '{name}' Test Passed!")
            else:
                logger.warning(f"Raw Output '{name}' Test Failed!")
                all_passing = False

        # Compare point cloud dims
        for i in range(len(self.torch_point_cloud_dims)):
            passing, pcc_message = comp_pcc(
                self.torch_point_cloud_dims[i], ttnn_point_cloud_dims[i], self.PCC_THRESHOLD
            )
            if passing:
                logger.info(f"point_cloud_dims[{i}] Test Passed!")
            else:
                logger.warning(f"point_cloud_dims[{i}] Test Failed!")
                all_passing = False

        self.pcc_passed = all_passing
        self.pcc_message = f"PCC check with threshold {self.PCC_THRESHOLD}"

        logger.info(f"DETR3D - batch_size={self.batch_size}, PCC={self.pcc_message}")

        assert all_passing, f"One or more raw outputs failed PCC check with threshold {self.PCC_THRESHOLD}"

    def dealloc_output(self):
        """Deallocate output tensors"""
        if hasattr(self, "cls_logits"):
            ttnn.deallocate(self.cls_logits)
        if hasattr(self, "center_offset"):
            ttnn.deallocate(self.center_offset)
        if hasattr(self, "size_normalized"):
            ttnn.deallocate(self.size_normalized)
        if hasattr(self, "angle_logits"):
            ttnn.deallocate(self.angle_logits)
        if hasattr(self, "angle_residual_normalized"):
            ttnn.deallocate(self.angle_residual_normalized)
        if hasattr(self, "angle_residual"):
            ttnn.deallocate(self.angle_residual)
        if hasattr(self, "ttnn_query_xyz"):
            ttnn.deallocate(self.ttnn_query_xyz)
        if hasattr(self, "ttnn_point_cloud_dims"):
            for tensor in self.ttnn_point_cloud_dims:
                ttnn.deallocate(tensor)


def create_detr3d_test_infra(
    device,
    batch_size=1,
    model_location_generator=None,
    input_shape=(1, 20000, 3),
    encoder_only=False,
    torch_input_dict=None,
    inputs_mesh_mapper=None,
    weights_mesh_mapper=None,
    outputs_mesh_composer=None,
    load_real_input=True,
):
    """Factory function to create DETR3D test infrastructure"""
    return Detr3dPerformanceRunnerInfra(
        device=device,
        batch_size=batch_size,
        model_location_generator=model_location_generator,
        input_shape=input_shape,
        encoder_only=encoder_only,
        torch_input_dict=torch_input_dict,
        inputs_mesh_mapper=inputs_mesh_mapper,
        weights_mesh_mapper=weights_mesh_mapper,
        outputs_mesh_composer=outputs_mesh_composer,
        load_real_input=load_real_input,
    )
