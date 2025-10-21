# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
import pytest

import ttnn
from loguru import logger

from models.experimental.transfuser.reference.config import GlobalConfig
from models.experimental.transfuser.reference.transfuser_backbone import TransfuserBackbone
from models.experimental.transfuser.tt.custom_preprocessing import create_custom_mesh_preprocessor

from models.experimental.transfuser.tests.test_gpt import create_gpt_preprocessor
from models.experimental.transfuser.tt.transfuser_backbone import TtTransfuserBackbone
from ttnn.model_preprocessing import (
    preprocess_model_parameters,
)
from tests.ttnn.utils_for_testing import check_with_pcc


# import torch
from typing import Dict, Tuple
import torchvision.transforms.functional as TF


def _to_nchw_3(t: torch.Tensor) -> torch.Tensor:
    """
    Convert tensor to [1,3,H,W] (batch=1). Accepts:
      [H,W], [C,H,W], [H,W,C], [1,C,H,W], [1,H,W,C]
    If channels==1, replicate to 3. If channels>3, take first 3.
    """
    t = t.detach().cpu().float()
    if t.ndim == 4:  # [N,*,*,*]
        if t.shape[0] != 1:
            raise ValueError(f"Expected batch=1, got shape {tuple(t.shape)}")
        t = t[0]  # drop batch

    if t.ndim == 2:  # [H,W] -> [1,H,W]
        t = t.unsqueeze(0)

    if t.ndim == 3:
        # could be [C,H,W] or [H,W,C]
        C_first = t.shape[0] <= 4  # heuristic: small first dim means channels
        if not C_first:  # assume [H,W,C]
            t = t.permute(2, 0, 1)
        C, H, W = t.shape
        if C == 1:
            t = t.repeat(3, 1, 1)
        elif C > 3:
            t = t[:3]
        t = t.unsqueeze(0)  # [1,3,H,W]
        return t

    raise ValueError(f"Unsupported tensor shape {tuple(t.shape)}")


def _safe_resize(img_1x3hw: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
    """Resize to [H,W] with bilinear (keeps [1,3,H,W])."""
    _, C, H, W = img_1x3hw.shape
    assert C == 3
    return TF.resize(img_1x3hw, size=target_hw, antialias=True)


def _normalize_01(x: torch.Tensor) -> torch.Tensor:
    """Per-sample min-max to [0,1] (robust default for raw tensors)."""
    x = x.clone()
    x -= x.amin(dim=(1, 2, 3), keepdim=True)
    denom = x.amax(dim=(1, 2, 3), keepdim=True)
    x = x / (denom + 1e-12)
    return x


def load_inputs_from_pth(
    path: str,
    rgb_target_hw=(160, 704),
    lidar_target_hw=(256, 256),
) -> Dict[str, torch.Tensor]:
    """
    Returns a dict with:
      - 'rgb_input'        -> [1,3,160,704] float32 in [0,1]
      - 'lidar_bev_input'  -> [1,3,256,256] float32 in [0,1]
      - 'velocity'         -> [1,1] (optional if found)
    """
    obj = torch.load(path, map_location="cpu")

    # Case 1: direct dict with expected keys
    if isinstance(obj, dict) and ("rgb_input" in obj or "lidar_bev_input" in obj):
        out = {}
        if "rgb_input" in obj:
            rgb = _to_nchw_3(obj["rgb_input"])
            rgb = _safe_resize(rgb, rgb_target_hw)
            rgb = _normalize_01(rgb)
            out["rgb_input"] = rgb
        if "lidar_bev_input" in obj:
            bev = _to_nchw_3(obj["lidar_bev_input"])
            bev = _safe_resize(bev, lidar_target_hw)
            bev = _normalize_01(bev)
            out["lidar_bev_input"] = bev
        if "velocity" in obj:
            vel = torch.as_tensor(obj["velocity"], dtype=torch.float32).view(1, 1)
            out["velocity"] = vel
        return out

    # Case 2: nested dict under a known key
    for k in ("inputs", "data", "sample", "batch"):
        if isinstance(obj, dict) and k in obj and isinstance(obj[k], dict):
            return load_inputs_from_pth({k: obj[k]}[k], rgb_target_hw, lidar_target_hw)

    # Case 3: list/tuple of tensors (guess order)
    if isinstance(obj, (list, tuple)):
        cand = [x for x in obj if isinstance(x, torch.Tensor)]
        if len(cand) >= 1:
            # Heuristic: largest spatial tensor -> rgb, second -> lidar
            cand_sorted = sorted(cand, key=lambda t: (t.ndim, *(t.shape[-2:] if t.ndim >= 2 else (0, 0))), reverse=True)
            out = {}
            if len(cand_sorted) >= 1:
                rgb = _to_nchw_3(cand_sorted[0])
                rgb = _safe_resize(rgb, rgb_target_hw)
                out["rgb_input"] = _normalize_01(rgb)
            if len(cand_sorted) >= 2:
                bev = _to_nchw_3(cand_sorted[1])
                bev = _safe_resize(bev, lidar_target_hw)
                out["lidar_bev_input"] = _normalize_01(bev)
            return out

    # Case 4: likely a state_dict (model weights)
    if isinstance(obj, dict) and all(isinstance(v, torch.Tensor) for v in obj.values()):
        raise ValueError(
            f"‘{path}’ looks like a state_dict (model weights), not input data. "
            "Please supply a .pth/.pt saved with input tensors, e.g. "
            "{'rgb_input': ..., 'lidar_bev_input': ..., 'velocity': ...}."
        )

    # Unknown format → help the user by showing what’s inside
    raise ValueError(
        f"Unrecognized .pth format at {path}. Top-level type: {type(obj)}. "
        "If it’s a dict, keys were: "
        f"{list(obj.keys()) if isinstance(obj, dict) else 'N/A'}."
    )


class TransfuserBackboneInfra:
    def __init__(
        self,
        device,
        image_architecture,
        lidar_architecture,
        n_layer,
        use_velocity,
        use_target_point_image,
        img_input_shape,
        lidar_input_shape,
        model_config,
    ):
        super().__init__()
        # self._init_seeds()
        self.device = device
        self.n_layer = n_layer
        self.image_arch = image_architecture
        self.lidar_arch = lidar_architecture
        self.use_velocity = use_velocity
        self.img_input_shape = img_input_shape
        self.lidar_input_shape = lidar_input_shape
        self.num_devices = device.get_num_devices()
        # self.batch_size = batch_size * self.num_devices
        self.inputs_mesh_mapper, self.weights_mesh_mapper, self.output_mesh_composer = self.get_mesh_mappers(device)
        # self.name = name

        inputs = torch.load("transfuser_inputs_final.pt")
        self.rgb = inputs["image"]  # RGB camera image tensor
        # save_tensor_as_image(rgb, 'rgb.png')
        self.lidar_bev = inputs["lidar"]  # LiDAR BEV tensor
        # save_tensor_as_image(lidar_bev, 'lidar_bev.png')

        self.ego_vel = inputs["velocity"]  # Ego velocity tensor
        # save_tensor_as_image(ego_vel, 'ego_vel.png')
        # Check shapes
        print("RGB shape:", self.rgb.shape)
        print("LiDAR BEV shape:", self.lidar_bev.shape)
        print("Ego velocity shape:", self.ego_vel.shape)

        # setting machine to avoid loading files
        self.config = GlobalConfig(setting="eval")
        self.config.n_layer = self.n_layer
        if use_target_point_image:
            self.config.use_target_point_image = use_target_point_image

        # Build reference torch model
        torch_model = TransfuserBackbone(
            self.config,
            image_architecture=self.image_arch,
            lidar_architecture=self.lidar_arch,
            use_velocity=self.use_velocity,
        )
        torch_model.eval()

        # Preprocess parameters for TTNN
        parameters = preprocess_model_parameters(
            initialize_model=lambda: torch_model,
            custom_preprocessor=create_custom_mesh_preprocessor(self.weights_mesh_mapper),
            device=None,
        )
        gpt1_parameters = preprocess_model_parameters(
            initialize_model=lambda: torch_model.transformer1,
            custom_preprocessor=create_gpt_preprocessor(device, n_layer, ttnn.bfloat16),
            device=device,
        )
        parameters["transformer1"] = gpt1_parameters
        gpt2_parameters = preprocess_model_parameters(
            initialize_model=lambda: torch_model.transformer2,
            custom_preprocessor=create_gpt_preprocessor(device, n_layer, ttnn.bfloat16),
            device=device,
        )
        parameters["transformer2"] = gpt2_parameters
        gpt3_parameters = preprocess_model_parameters(
            initialize_model=lambda: torch_model.transformer3,
            custom_preprocessor=create_gpt_preprocessor(device, n_layer, ttnn.bfloat16),
            device=device,
        )
        parameters["transformer3"] = gpt3_parameters
        gpt4_parameters = preprocess_model_parameters(
            initialize_model=lambda: torch_model.transformer4,
            custom_preprocessor=create_gpt_preprocessor(device, n_layer, ttnn.bfloat16),
            device=device,
        )
        parameters["transformer4"] = gpt4_parameters
        # inputs = load_inputs_from_pth("/path/to/your_inputs.pth")
        # Prepare golden inputs/outputs
        self.torch_image_input = torch.randn(self.img_input_shape)
        self.torch_lidar_input = torch.randn(self.lidar_input_shape)
        self.torch_velocity_input = torch.randn(1, 1)
        # self.torch_image_input   = inputs.get("rgb_input",   torch.randn(1, 3, 160, 704))
        # self.torch_lidar_input   = inputs.get("lidar_bev_input", torch.randn(1, 3, 256, 256))
        # self.torch_velocity_input= inputs.get("velocity", torch.randn(1, 1))
        with torch.no_grad():
            self.torch_features, self.torch_image_grid, self.torch_fused = torch_model(
                self.torch_image_input,
                self.torch_lidar_input,
                self.torch_velocity_input,
            )

        # Convert input to TTNN format
        # tt_image_input = ttnn.from_torch(
        #     self.torch_image_input.permute(0, 2, 3, 1),
        #     dtype=ttnn.bfloat16,
        #     mesh_mapper=self.inputs_mesh_mapper,
        # )
        # tt_lidar_input = ttnn.from_torch(
        #     self.torch_lidar_input.permute(0, 2, 3, 1),
        #     dtype=ttnn.bfloat16,
        #     layout=ttnn.TILE_LAYOUT,
        #     device=device,
        #     mesh_mapper=self.inputs_mesh_mapper,
        # )
        # tt_velocity_input = ttnn.from_torch(
        #     self.torch_velocity_input,
        #     device=device,
        #     dtype=ttnn.bfloat16,
        #     layout=ttnn.ROW_MAJOR_LAYOUT,
        # )
        tt_image_input = ttnn.from_torch(
            # self.torch_image_input.permute(0, 2, 3, 1),
            self.rgb.permute(0, 2, 3, 1),
            dtype=ttnn.bfloat16,
            mesh_mapper=self.inputs_mesh_mapper,
        )
        tt_lidar_input = ttnn.from_torch(
            self.lidar_bev.permute(0, 2, 3, 1),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            mesh_mapper=self.inputs_mesh_mapper,
        )
        tt_velocity_input = ttnn.from_torch(
            self.ego_vel,
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )

        self.input_image_tensor = ttnn.to_device(tt_image_input, device)
        self.input_lidar_tensor = ttnn.to_device(tt_lidar_input, device)
        self.input_velocity_tensor = ttnn.to_device(tt_velocity_input, device)

        # Build TTNN model
        self.ttnn_model = TtTransfuserBackbone(
            device,
            parameters=parameters,
            stride=2,
            model_config=model_config,
            config=self.config,
        )

        # Run + validate

        self.run()
        self.validate(model_config)

    def _init_seeds(self):
        if not hasattr(self, "_model_initialized"):
            torch.manual_seed(42)
            torch.cuda.manual_seed_all(42)
            torch.backends.cudnn.deterministic = True
            self._model_initialized = True

    def get_mesh_mappers(self, device):
        if device.get_num_devices() != 1:
            return (
                ttnn.ShardTensorToMesh(device, dim=0),
                None,
                ttnn.ConcatMeshToTensor(device, dim=0),
            )
        return None, None, None

    def run(self):
        self.output_features, self.output_image_grid, self.output_fused = self.ttnn_model(
            self.input_image_tensor, self.input_lidar_tensor, self.input_velocity_tensor, self.device
        )
        return self.output_features, self.output_image_grid, self.output_fused

    def validate(self, model_config, output_tensor=None):
        # Validate image output
        tt_features_torch = []
        fpn_names = ["p2", "p3", "p4", "p5"]
        for i, (feature, name) in enumerate(zip(self.output_features, fpn_names)):
            tt_feat = ttnn.to_torch(
                feature,
                device=self.device,
                mesh_composer=self.output_mesh_composer,
            )

            # Permute NHWC -> NCHW
            tt_feat = tt_feat.permute(0, 3, 1, 2)
            tt_features_torch.append(tt_feat)

        # Validate output_image_grid
        tt_image_grid_torch = ttnn.to_torch(
            self.output_image_grid,
            device=self.device,
            mesh_composer=self.output_mesh_composer,
        )
        tt_image_grid_torch = tt_image_grid_torch.permute(0, 3, 1, 2)

        # Validate output_fused_tensor
        tt_fused_torch = ttnn.to_torch(
            self.output_fused,
            device=self.device,
            mesh_composer=self.output_mesh_composer,
        )

        # Deallocate output tensors
        for feature in self.output_features:
            ttnn.deallocate(feature)
        ttnn.deallocate(self.output_image_grid)
        ttnn.deallocate(self.output_fused)

        # Validate FPN features
        fpn_pcc_results = []
        for torch_feat, tt_feat, name in zip(self.torch_features, tt_features_torch, fpn_names):
            pcc_passed, pcc_msg = check_with_pcc(torch_feat, tt_feat, pcc=0.95)
            fpn_pcc_results.append((pcc_passed, pcc_msg))
            logger.info(f"{name} PCC: {pcc_msg}")

        # Validate image grid
        grid_pcc_passed, grid_pcc_msg = check_with_pcc(self.torch_image_grid, tt_image_grid_torch, pcc=0.95)
        logger.info(f"Image Grid PCC: {grid_pcc_msg}")

        # Validate fused features
        fused_pcc_passed, fused_pcc_msg = check_with_pcc(self.torch_fused, tt_fused_torch, pcc=0.95)
        logger.info(f"Fused Features PCC: {fused_pcc_msg}")

        # All outputs must pass
        all_fpn_passed = all(result[0] for result in fpn_pcc_results)
        overall_passed = all_fpn_passed and grid_pcc_passed and fused_pcc_passed

        assert overall_passed, logger.error(
            f"PCC check failed - FPN: {fpn_pcc_results}, Grid: {grid_pcc_msg}, Fused: {fused_pcc_msg}"
        )

        return overall_passed, f"FPN: {fpn_pcc_results}, Grid: {grid_pcc_msg}, Fused: {fused_pcc_msg}"


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
    "image_architecture, lidar_architecture, n_layer, use_velocity, use_target_point_image, img_input_shape, lidar_input_shape",
    [("regnety_032", "regnety_032", 4, False, True, (1, 3, 160, 704), (1, 3, 256, 256))],
)
def test_stem(
    device,
    image_architecture,
    lidar_architecture,
    n_layer,
    use_velocity,
    use_target_point_image,
    img_input_shape,
    lidar_input_shape,
):
    TransfuserBackboneInfra(
        device,
        image_architecture,
        lidar_architecture,
        n_layer,
        use_velocity,
        use_target_point_image,
        img_input_shape,
        lidar_input_shape,
        model_config,
    )
