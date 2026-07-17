# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

import torch
from loguru import logger

import ttnn
from ttnn.model_preprocessing import preprocess_model_parameters

from models.common.utility_functions import comp_pcc, tt2torch_tensor
from models.experimental.pointpillars.tt.pointpillars import TtPointPillars, PointPillarsPreprocessor
from models.experimental.pointpillars.reference.pointpillars import PointPillars
from models.experimental.pointpillars.tt.custom_preprocessor import create_custom_mesh_preprocessor
from models.tt_cnn.tt.pipeline import get_memory_config_for_persistent_dram_tensor


class PointPillarsPerformanceRunnerInfra:
    def __init__(
        self,
        device,
        batch_size,
        model_location_generator=None,
        inputs_mesh_mapper=None,
        weights_mesh_mapper=None,
        outputs_mesh_composer=None,
        checkpoint_path=None,
    ):
        if not hasattr(self, "_model_initialized"):
            torch.manual_seed(0)
            self._model_initialized = True

        self.pcc_passed = False
        self.pcc_message = "Did you forget to call validate()?"
        self.device = device
        self.batch_size = batch_size
        self.model_location_generator = model_location_generator
        self.num_devices = device.get_num_devices()
        self.inputs_mesh_mapper = inputs_mesh_mapper
        self.weights_mesh_mapper = weights_mesh_mapper
        self.outputs_mesh_composer = outputs_mesh_composer
        self.checkpoint_path = checkpoint_path

        self.voxel_size = [0.16, 0.16, 4]
        self.point_cloud_range = [0, -39.68, -3, 69.12, 39.68, 1]
        self.max_num_points = 32
        self.max_voxels = (16000, 40000)
        self.nclasses = 3

        self.torch_model = PointPillars(
            nclasses=self.nclasses,
            voxel_size=self.voxel_size,
            point_cloud_range=self.point_cloud_range,
            max_num_points=self.max_num_points,
            max_voxels=self.max_voxels,
        )

        self._load_checkpoint()

        self.torch_model = self.torch_model.to(dtype=torch.bfloat16)
        self.torch_model.eval()

        self.parameters = preprocess_model_parameters(
            initialize_model=lambda: self.torch_model,
            custom_preprocessor=create_custom_mesh_preprocessor(mesh_mapper=self.weights_mesh_mapper),
            device=device,
        )

        self.preprocessor = PointPillarsPreprocessor(
            voxel_size=self.voxel_size,
            point_cloud_range=self.point_cloud_range,
            max_num_points=self.max_num_points,
            max_voxels=self.max_voxels,
            parameters=self.parameters,
            device=device,
        )

        self.ttnn_model = TtPointPillars(
            nclasses=self.nclasses,
            parameters=self.parameters,
            device=device,
        )

        self.tt_output = None
        self.torch_output = None

    def _load_checkpoint(self):
        if self.checkpoint_path is None:
            logger.warning("No checkpoint path provided, using random weights")
            return

        try:
            checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
            if "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            elif "model" in checkpoint:
                state_dict = checkpoint["model"]
            else:
                state_dict = checkpoint
            self.torch_model.load_state_dict(state_dict)
            logger.info(f"Successfully loaded pretrained weights from {self.checkpoint_path}")
        except FileNotFoundError:
            logger.warning(f"Checkpoint file '{self.checkpoint_path}' not found, using random weights")

    def preprocess_point_cloud(self, batched_pts):
        pillar_features = self.preprocessor.forward(batched_pts)
        pillar_features = ttnn.permute(pillar_features, (0, 2, 3, 1))
        pillar_features = ttnn.reshape(
            pillar_features,
            (
                pillar_features.shape[0],
                1,
                pillar_features.shape[1] * pillar_features.shape[2],
                pillar_features.shape[3],
            ),
        )
        return pillar_features

    def get_torch_reference(self, batched_pts):
        with torch.no_grad():
            torch_cls, torch_reg, torch_dir = self.torch_model(batched_pts)
        self.torch_output = (torch_cls, torch_reg, torch_dir)
        return self.torch_output

    def setup_dram_interleaved_input(self, batched_pts=None):
        if batched_pts is not None:
            pillar_features = self.preprocess_point_cloud(batched_pts)
        else:
            pillar_features = self.preprocess_point_cloud([torch.randn(18221, 4, dtype=torch.bfloat16)])

        tt_inputs_host = ttnn.from_torch(
            ttnn.to_torch(pillar_features),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=self.inputs_mesh_mapper,
        )
        return tt_inputs_host, ttnn.DRAM_MEMORY_CONFIG

    def setup_sharded_input(self, device, pillar_features_host):
        tt_inputs_host = pillar_features_host

        dram_input_mem_config = get_memory_config_for_persistent_dram_tensor(
            tt_inputs_host.shape, ttnn.TensorMemoryLayout.HEIGHT_SHARDED, device.dram_grid_size()
        )

        input_l1_core_grid = ttnn.CoreGrid(x=8, y=8)
        height_dim = tt_inputs_host.shape[-2]

        if height_dim % input_l1_core_grid.num_cores != 0:
            num_cores = input_l1_core_grid.num_cores
            while height_dim % num_cores != 0 and num_cores > 1:
                num_cores -= 1
            y = min(8, num_cores)
            while num_cores % y != 0:
                y -= 1
            x = num_cores // y
            input_l1_core_grid = ttnn.CoreGrid(x=x, y=y)

        l1_input_mem_config = ttnn.create_sharded_memory_config(
            shape=(height_dim // input_l1_core_grid.num_cores, tt_inputs_host.shape[-1]),
            core_grid=input_l1_core_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

        return tt_inputs_host, dram_input_mem_config, l1_input_mem_config

    def run(self):
        self.tt_output = self.ttnn_model.forward(self.input_tensor)

    def validate(self, tt_output=None):
        tt_output = self.tt_output if tt_output is None else tt_output

        if self.torch_output is None:
            logger.warning("No torch reference output. Call get_torch_reference() first.")
            return

        torch_cls, torch_reg, torch_dir = self.torch_output
        tt_cls, tt_reg, tt_dir = tt_output

        self.pcc_passed = []
        self.pcc_message = []

        tt_cls_torch = tt2torch_tensor(tt_cls).permute(0, 3, 1, 2)
        passing_cls, pcc_cls = comp_pcc(torch_cls, tt_cls_torch, 0.97)
        self.pcc_passed.append(passing_cls)
        self.pcc_message.append(f"Classification: {pcc_cls}")
        logger.info(f"Classification PCC: {pcc_cls}")

        tt_reg_torch = tt2torch_tensor(tt_reg).permute(0, 3, 1, 2)
        passing_reg, pcc_reg = comp_pcc(torch_reg, tt_reg_torch, 0.99)
        self.pcc_passed.append(passing_reg)
        self.pcc_message.append(f"Regression: {pcc_reg}")
        logger.info(f"Regression PCC: {pcc_reg}")

        tt_dir_torch = tt2torch_tensor(tt_dir).permute(0, 3, 1, 2)
        passing_dir, pcc_dir = comp_pcc(torch_dir, tt_dir_torch, 0.99)
        self.pcc_passed.append(passing_dir)
        self.pcc_message.append(f"Direction: {pcc_dir}")
        logger.info(f"Direction PCC: {pcc_dir}")

        assert all(self.pcc_passed), f"PointPillars PCC check failed: {self.pcc_message}"

    def dealloc_output(self):
        if self.tt_output is not None:
            tt_cls, tt_reg, tt_dir = self.tt_output
            ttnn.deallocate(tt_cls)
            ttnn.deallocate(tt_reg)
            ttnn.deallocate(tt_dir)
            self.tt_output = None
