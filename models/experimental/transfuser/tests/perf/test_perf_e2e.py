# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import time
import torch
from collections import OrderedDict
from typing import Dict, Any, List
from loguru import logger

import ttnn
from ttnn.model_preprocessing import preprocess_model_parameters
from ttnn.model_preprocessing import infer_ttnn_module_args as infer_ttnn_module_args_torch

from models.experimental.transfuser.reference.config import GlobalConfig
from models.experimental.transfuser.reference.lidar_center_net import LidarCenterNet, process_input
from models.experimental.transfuser.tt.lidar_center_net import LidarCenterNet as TtLidarCenterNet
from models.experimental.transfuser.tests.pcc.test_gpt import create_gpt_preprocessor
from models.experimental.transfuser.tt.custom_preprocessing import create_custom_mesh_preprocessor
from models.experimental.transfuser.tests.pcc.test_transfuser_backbone import regroup_model_args
from models.experimental.transfuser.resources.transfuser_dataset import ensure_scenario3_town01_curved_route0
from models.experimental.transfuser.resources.transfuser_checkpoint import ensure_transfuser_checkpoint_2022
from models.perf.perf_utils import prep_perf_report
from models.tt_cnn.tt.pipeline import PipelineConfig, create_pipeline_from_config


def create_lidar_center_net_head_preprocessor(device, weight_dtype=ttnn.bfloat16):
    def custom_preprocessor(torch_model, name, ttnn_module_args):
        parameters = {}

        for head_name in [
            "heatmap_head",
            "wh_head",
            "offset_head",
            "yaw_class_head",
            "yaw_res_head",
            "velocity_head",
            "brake_head",
        ]:
            local_dtype = weight_dtype
            if head_name == "heatmap_head":
                local_dtype = ttnn.float32

            if hasattr(torch_model, head_name):
                head = getattr(torch_model, head_name)
                parameters[head_name] = {}
                parameters[head_name]["conv1_weight"] = ttnn.from_torch(
                    head[0].weight, dtype=local_dtype, layout=ttnn.ROW_MAJOR_LAYOUT
                )
                parameters[head_name]["conv1_bias"] = ttnn.from_torch(
                    head[0].bias.reshape(1, 1, 1, -1), dtype=local_dtype, layout=ttnn.ROW_MAJOR_LAYOUT
                )
                parameters[head_name]["conv2_weight"] = ttnn.from_torch(
                    head[2].weight, dtype=local_dtype, layout=ttnn.ROW_MAJOR_LAYOUT
                )
                parameters[head_name]["conv2_bias"] = ttnn.from_torch(
                    head[2].bias.reshape(1, 1, 1, -1), dtype=local_dtype, layout=ttnn.ROW_MAJOR_LAYOUT
                )

        return parameters

    return custom_preprocessor


def load_trained_weights(weight_path: str) -> Dict[str, Any]:
    checkpoint = torch.load(weight_path, map_location="cpu")
    state_dict = {}
    for key, value in checkpoint.items():
        if key.startswith("module._model."):
            clean_key = key[len("module._model.") :]
            state_dict[clean_key] = value
        else:
            state_dict[key] = value

    backbone_keys = [
        "image_encoder",
        "lidar_encoder",
        "transformer1",
        "transformer2",
        "transformer3",
        "transformer4",
        "change_channel_conv_image",
        "change_channel_conv_lidar",
        "up_conv5",
        "up_conv4",
        "up_conv3",
        "c5_conv",
    ]
    for key in list(state_dict.keys()):
        for backbone in backbone_keys:
            if key.startswith(f"{backbone}."):
                new_key = f"_model.{backbone}.{key[len(backbone)+1:]}"
                state_dict[new_key] = state_dict.pop(key)
                break

    detection_components = ["head", "pred_bev", "join", "decoder", "output"]
    for key in list(state_dict.keys()):
        for component in detection_components:
            if key.startswith(f"module.{component}."):
                new_key = key[len("module.") :]
                state_dict[new_key] = state_dict.pop(key)
                break

    return state_dict


def delete_incompatible_keys(state_dict: Dict[str, Any], keys_to_delete: List[str]) -> Dict[str, Any]:
    new_state = OrderedDict(state_dict)
    for k in keys_to_delete:
        if k in new_state:
            del new_state[k]
    return new_state


def create_transfuser_pipeline_model(ttnn_model, tt_image, tt_lidar_bev, tt_velocity, target_point):
    """Wrapper to adapt transfuser model inputsfor pipeline interface."""

    def run(pipeline_input):
        logger.info(f"RUN")
        ttnn.deallocate(pipeline_input)
        tt_features, tt_fused_features = ttnn_model.forward_ego(tt_image, tt_lidar_bev, tt_velocity, target_point)
        return [tt_features, tt_fused_features]

    return run


"""
        PYTEST E2E transfuser
"""
SEED = 42


@pytest.mark.parametrize(
    "device_params",
    [
        {
            "l1_small_size": 16384,
            "trace_region_size": 400,
            "num_command_queues": 2,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize("num_iterations", [32])
@pytest.mark.parametrize("image_architecture", ["regnety_032"])
@pytest.mark.parametrize("lidar_architecture", ["regnety_032"])
@pytest.mark.parametrize("n_layer", [4])
@pytest.mark.parametrize("frame", ["0120"])
@pytest.mark.parametrize("use_optimized_self_attn", [False])
@pytest.mark.parametrize(
    "batch_size, expected_compile_time, expected_throughput_fps",
    [(1, 2, 3)],
)
def test_perf_transfuser_ttnn(
    device,
    num_iterations,
    image_architecture,
    lidar_architecture,
    n_layer,
    frame,
    use_optimized_self_attn,
    batch_size,
    expected_compile_time,
    expected_throughput_fps,
):
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)

    data_root = ensure_scenario3_town01_curved_route0()
    weights_path = ensure_transfuser_checkpoint_2022()

    config = GlobalConfig(setting="eval")
    config.n_layer = n_layer
    config.use_target_point_image = True

    inputs = process_input(data_root, frame, config=config, normalize_image=False)

    image = inputs["image"]
    lidar_bev = inputs["lidar"]
    velocity = inputs["velocity"]
    target_point = inputs["target_point"]

    ref_layer = LidarCenterNet(
        config,
        backbone="transFuser",
        image_architecture=image_architecture,
        lidar_architecture=lidar_architecture,
        use_velocity=False,
    ).eval()

    modified_state_dict = load_trained_weights(weights_path)
    modified_state_dict = delete_incompatible_keys(
        modified_state_dict,
        [
            "_model.lidar_encoder._model.stem.conv.weight",
            "module.seg_decoder.deconv1.0.weight",
            "module.seg_decoder.deconv1.0.bias",
            "module.seg_decoder.deconv1.2.weight",
            "module.seg_decoder.deconv1.2.bias",
            "module.seg_decoder.deconv2.0.weight",
            "module.seg_decoder.deconv2.0.bias",
            "module.seg_decoder.deconv2.2.weight",
            "module.seg_decoder.deconv2.2.bias",
            "module.seg_decoder.deconv3.0.weight",
            "module.seg_decoder.deconv3.0.bias",
            "module.seg_decoder.deconv3.2.weight",
            "module.seg_decoder.deconv3.2.bias",
            "module.depth_decoder.deconv1.0.weight",
            "module.depth_decoder.deconv1.0.bias",
            "module.depth_decoder.deconv1.2.weight",
            "module.depth_decoder.deconv1.2.bias",
            "module.depth_decoder.deconv2.0.weight",
            "module.depth_decoder.deconv2.0.bias",
            "module.depth_decoder.deconv2.2.weight",
            "module.depth_decoder.deconv2.2.bias",
            "module.depth_decoder.deconv3.0.weight",
            "module.depth_decoder.deconv3.0.bias",
            "module.depth_decoder.deconv3.2.weight",
            "module.depth_decoder.deconv3.2.bias",
        ],
    )
    ref_layer.load_state_dict(modified_state_dict, strict=True)

    # with torch.no_grad():
    #     (
    #         ref_fused_features,
    #         ref_feature,
    #         pred_wp,
    #         ref_head_results,
    #         ref_boxes,
    #         ref_rotated_bboxes,
    #     ) = ref_layer.forward_ego(image, lidar_bev, target_point, velocity)

    torch_model = ref_layer._model

    # Preprocess parameters for TTNN
    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model,
        custom_preprocessor=create_custom_mesh_preprocessor(None),
        device=None,
    )

    # GPT submodules
    for name in ["transformer1", "transformer2", "transformer3", "transformer4"]:
        parameters[name] = preprocess_model_parameters(
            initialize_model=lambda n=name: getattr(torch_model, n),
            custom_preprocessor=create_gpt_preprocessor(device, n_layer, ttnn.bfloat16, use_optimized_self_attn),
            device=device,
        )

    # Head
    parameters["head"] = preprocess_model_parameters(
        initialize_model=lambda: ref_layer.head,
        custom_preprocessor=create_lidar_center_net_head_preprocessor(device, ttnn.bfloat16),
        device=device,
    )

    model_args = infer_ttnn_module_args_torch(
        model=torch_model,
        run_model=lambda model: model(image, lidar_bev, velocity),
        device=None,
        absolute_name=True,
    )
    model_args = regroup_model_args(model_args)

    transfuser_model = ref_layer._model
    ttnn_model = TtLidarCenterNet(
        device,
        parameters,
        config,
        backbone="transFuser",
        torch_model=transfuser_model,
        model_args=model_args,
    )

    # Convert inputs to TTNN
    tt_image_input = ttnn.from_torch(
        image.permute(0, 2, 3, 1),
        dtype=ttnn.bfloat16,
        device=device,
    )
    tt_lidar_input = ttnn.from_torch(
        lidar_bev.permute(0, 2, 3, 1),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    tt_velocity_input = ttnn.from_torch(velocity, device=device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

    image_shape = tt_image_input.shape
    dram_grid_size = device.dram_grid_size()

    # Calculate physical 2D dimensions for WIDTH_SHARDED layout
    width = image_shape[-1]
    volume = image_shape[0] * image_shape[1] * image_shape[2] * image_shape[3]
    physical_height = volume // width
    max_cores = dram_grid_size.x

    # Find optimal cores ensuring tile-aligned shards (multiple of 32)
    dram_cores = 1
    for cores in range(max_cores, 0, -1):
        if width % cores == 0 and (width // cores) % 32 == 0:
            dram_cores = cores
            break

    # Create sharded memory configs for DRAM and L1
    shard_width = width // dram_cores
    dram_shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram_cores - 1, 0))}),
        [physical_height, shard_width],
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    dram_input_memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, dram_shard_spec
    )

    l1_input_memory_config = ttnn.create_sharded_memory_config(
        shape=(physical_height, shard_width),
        core_grid=ttnn.CoreGrid(y=1, x=dram_cores),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    run_model = create_transfuser_pipeline_model(
        ttnn_model, tt_image_input, tt_lidar_input, tt_velocity_input, target_point
    )
    config_pipeline = PipelineConfig(
        use_trace=True, num_command_queues=2, all_transfers_on_separate_command_queue=False
    )
    pipeline = create_pipeline_from_config(
        config_pipeline,
        run_model,
        device,
        dram_input_memory_config=dram_input_memory_config,
        l1_input_memory_config=l1_input_memory_config,
    )

    image_host = tt_image_input.cpu()
    host_inputs = [image_host] * num_iterations

    start = time.time()
    pipeline.compile(image_host)
    end = time.time()
    compile_time = end - start

    pipeline.preallocate_output_tensors_on_host(num_iterations)

    start = time.time()
    outputs = pipeline.enqueue(host_inputs).pop_all()
    end = time.time()

    pipeline.cleanup()

    inference_time = (end - start) / num_iterations
    logger.info(f"Average model time={1000.0 * inference_time:.2f} ms")
    logger.info(f"Average model performance={num_iterations * batch_size / (end - start):.2f} fps")

    prep_perf_report(
        model_name="transfuser-2cq",
        batch_size=batch_size,
        inference_and_compile_time=compile_time,
        inference_time=inference_time,
        expected_compile_time=expected_compile_time,
        expected_inference_time=batch_size / expected_throughput_fps,
        comments=f"batch_{batch_size}",
    )
