# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
#
# SPDX-License-Identifier: Apache-2.0

import time
import pytest
import torch
import ttnn
from loguru import logger

from models.perf.perf_utils import prep_perf_report
from models.tt_cnn.tt.pipeline import (
    PipelineConfig,
    create_pipeline_from_config,
    get_memory_config_for_persistent_dram_tensor,
)
from models.common.utility_functions import run_for_wormhole_b0
from models.experimental.BEVFormerV2.reference import bevformer_v2
from models.experimental.BEVFormerV2.tt.ttnn_bevformer_v2 import TtBevFormerV2
from models.experimental.BEVFormerV2.tt.model_preprocessing import (
    create_bevformerv2_model_parameters,
)
from models.experimental.BEVFormerV2.common import load_torch_model
from models.experimental.BEVFormerV2.demo.processing import prepare_demo_sample


def create_bevformerv2_pipeline_model(ttnn_model, torch_input, img_metas, dtype=ttnn.bfloat16, original_shape=None):
    def run(l1_input_tensor):
        l1_input_tensor = ttnn.to_memory_config(l1_input_tensor, ttnn.DRAM_MEMORY_CONFIG)
        target_shape = tuple(original_shape)
        ttnn_input = ttnn.reshape(l1_input_tensor, target_shape)
        ttnn_input = ttnn.to_memory_config(ttnn_input, ttnn.DRAM_MEMORY_CONFIG)

        unwrapped_img_metas = img_metas
        while (
            isinstance(unwrapped_img_metas, list)
            and len(unwrapped_img_metas) > 0
            and isinstance(unwrapped_img_metas[0], list)
        ):
            unwrapped_img_metas = unwrapped_img_metas[0]

        if "can_bus" not in unwrapped_img_metas[0]:
            unwrapped_img_metas[0]["can_bus"] = [0.0] * 18

        img_feats = ttnn_model.extract_feat(img=[ttnn_input], img_metas=unwrapped_img_metas)

        x = img_feats
        x[0] = ttnn.to_layout(x[0], layout=ttnn.TILE_LAYOUT)

        outs = ttnn_model.pts_bbox_head(x, unwrapped_img_metas, prev_bev=None)

        bev_embed = outs["bev_embed"]
        all_cls_scores = outs["all_cls_scores"]
        all_bbox_preds = outs["all_bbox_preds"]

        return (bev_embed, all_cls_scores, all_bbox_preds)

    return run


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "l1_small_size": 4 * 8192,
            "trace_region_size": 10000000,
            "num_command_queues": 2,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize("num_iterations", [1])
@pytest.mark.parametrize(
    "batch_size, expected_compile_time, expected_throughput_fps",
    [(1, 100.0, 0.105)],
)
@pytest.mark.models_performance_bare_metal
def test_bevformerv2_e2e_performant_2cq_trace(
    device,
    num_iterations,
    batch_size,
    expected_compile_time,
    expected_throughput_fps,
    reset_seeds,
    model_location_generator,
):
    torch.manual_seed(0)
    dtype = ttnn.bfloat16

    logger.info("Building BEVFormerV2 model...")
    torch_model = bevformer_v2.BEVFormerV2(
        use_grid_mask=True,
        img_backbone=dict(depth=50, in_channels=3, out_indices=(1, 2, 3), style="caffe"),
        img_neck=dict(in_channels=[512, 1024, 2048], out_channels=256, num_outs=5),
        pts_bbox_head=dict(bev_h=200, bev_w=200, num_query=900, num_classes=10, in_channels=256),
        video_test_mode=True,
    )

    torch_model = load_torch_model(torch_model=torch_model, model_location_generator=model_location_generator)

    encoder_layers = 6
    decoder_layers = 6
    torch_model.pts_bbox_head.transformer.encoder.layers = torch.nn.ModuleList(
        list(torch_model.pts_bbox_head.transformer.encoder.layers)[:encoder_layers]
    )
    torch_model.pts_bbox_head.transformer.encoder.num_layers = encoder_layers
    torch_model.pts_bbox_head.transformer.decoder.layers = torch.nn.ModuleList(
        list(torch_model.pts_bbox_head.transformer.decoder.layers)[:decoder_layers]
    )
    torch_model.pts_bbox_head.transformer.decoder.num_layers = decoder_layers

    tensor, img_metas = prepare_demo_sample(
        sample_idx=0, data_root="models/experimental/BEVFormerV2/demo/demo_data/nuscenes"
    )
    sample_input = tensor

    parameter = create_bevformerv2_model_parameters(
        torch_model,
        [
            False,
            [sample_input],
            img_metas,
        ],
        device,
    )

    tt_model = TtBevFormerV2(
        device=device,
        params=parameter,
        use_grid_mask=False,
        img_backbone=dict(depth=50, in_channels=3, out_indices=(1, 2, 3), style="caffe"),
        img_neck=dict(in_channels=[512, 1024, 2048], out_channels=256, num_outs=5),
        pts_bbox_head=dict(
            bev_h=200,
            bev_w=200,
            num_query=900,
            num_classes=10,
            in_channels=256,
            encoder_num_layers=torch_model.pts_bbox_head.transformer.encoder.num_layers,
            decoder_num_layers=torch_model.pts_bbox_head.transformer.decoder.num_layers,
        ),
        video_test_mode=True,
    )

    ttnn.synchronize_device(device)

    batch_size_val, num_cameras, channels, height, width = sample_input.shape
    total_height = batch_size_val * num_cameras * height
    total_width = channels * width
    sample_input_reshaped = sample_input.reshape(1, 1, total_height, total_width)

    host_input_tensor = ttnn.from_torch(
        sample_input_reshaped,
        device=None,
        dtype=dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )

    dram_input_memory_config = get_memory_config_for_persistent_dram_tensor(
        host_input_tensor.shape, ttnn.TensorMemoryLayout.HEIGHT_SHARDED, device.dram_grid_size()
    )

    l1_core_grid = ttnn.CoreGrid(x=8, y=8)
    height_dim = host_input_tensor.shape[-2]
    assert height_dim % l1_core_grid.num_cores == 0, "Expecting even sharding on L1 input tensor"
    l1_input_memory_config = ttnn.create_sharded_memory_config(
        shape=(height_dim // l1_core_grid.num_cores, host_input_tensor.shape[-1]),
        core_grid=l1_core_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    original_shape_list = [batch_size_val, num_cameras, channels, height, width]
    pipeline_model = create_bevformerv2_pipeline_model(
        tt_model, sample_input, img_metas, dtype=dtype, original_shape=original_shape_list
    )

    pipeline = create_pipeline_from_config(
        config=PipelineConfig(use_trace=False, num_command_queues=2, all_transfers_on_separate_command_queue=False),
        model=pipeline_model,
        device=device,
        dram_input_memory_config=dram_input_memory_config,
        l1_input_memory_config=l1_input_memory_config,
    )

    input_tensors = [host_input_tensor] * num_iterations

    start = time.time()
    pipeline.compile(host_input_tensor)
    end = time.time()
    compile_time = end - start

    pipeline.preallocate_output_tensors_on_host(num_iterations)

    start = time.time()
    pipeline.enqueue(input_tensors).pop_all()
    end = time.time()

    pipeline.cleanup()

    inference_time = (end - start) / num_iterations
    logger.info(f"Average model time={1000.0 * inference_time : .2f} ms")
    logger.info(f"Average model performance={num_iterations * batch_size / (end-start) : .2f} fps")

    prep_perf_report(
        model_name="bevformerv2-trace-2cq",
        batch_size=batch_size,
        inference_and_compile_time=compile_time,
        inference_time=inference_time,
        expected_compile_time=expected_compile_time,
        expected_inference_time=batch_size / expected_throughput_fps,
        comments=f"batch_{batch_size}",
    )

    logger.info("Performance test completed!")
