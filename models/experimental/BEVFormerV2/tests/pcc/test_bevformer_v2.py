# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.

# SPDX-License-Identifier: Apache-2.0

import tracy
import pytest
import torch
import ttnn
from models.experimental.BEVFormerV2.reference import bevformer_v2
from models.experimental.BEVFormerV2.tt.ttnn_bevformer_v2 import TtBevFormerV2
from models.experimental.BEVFormerV2.tt.model_preprocessing import (
    create_bevformerv2_model_parameters,
)
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.experimental.BEVFormerV2.common import load_torch_model
from models.experimental.BEVFormerV2.demo.processing import prepare_demo_sample


@pytest.mark.parametrize("device_params", [{"l1_small_size": 4 * 8192}], indirect=True)
def test_bevformerv2(
    device,
    reset_seeds,
    model_location_generator,
):
    torch_model = bevformer_v2.BEVFormerV2(
        use_grid_mask=True,
        img_backbone=dict(depth=50, in_channels=3, out_indices=(1, 2, 3), style="caffe"),
        img_neck=dict(in_channels=[512, 1024, 2048], out_channels=256, num_outs=5),
        pts_bbox_head=dict(bev_h=200, bev_w=200, num_query=900, num_classes=10, in_channels=256),
        video_test_mode=True,
    )

    torch_model = load_torch_model(torch_model=torch_model, model_location_generator=model_location_generator)

    torch_model.pts_bbox_head.transformer.encoder.layers = torch.nn.ModuleList(
        list(torch_model.pts_bbox_head.transformer.encoder.layers)[:6]
    )
    torch_model.pts_bbox_head.transformer.encoder.num_layers = 6
    torch_model.pts_bbox_head.transformer.decoder.layers = torch.nn.ModuleList(
        list(torch_model.pts_bbox_head.transformer.decoder.layers)[:6]
    )
    torch_model.pts_bbox_head.transformer.decoder.num_layers = 6

    tensor, img_metas = prepare_demo_sample(
        sample_idx=0, data_root="models/experimental/BEVFormerV2/demo/demo_data/nuscenes"
    )
    img = [tensor]

    with torch.no_grad():
        torch_img_feats = torch_model.extract_feat(tensor, img_metas)
        torch_outputs = torch_model.pts_bbox_head(torch_img_feats, img_metas, prev_bev=None)

    parameter = create_bevformerv2_model_parameters(
        torch_model,
        [
            False,
            img,
            img_metas,
        ],
        device,
    )

    tensor = ttnn.from_torch(tensor, dtype=ttnn.bfloat16, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)
    img = [tensor]

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

    tracy.signpost("start")

    ttnn_img_feats = tt_model.extract_feat(img=img, img_metas=img_metas)
    ttnn_img_feats[0] = ttnn.to_layout(ttnn_img_feats[0], layout=ttnn.TILE_LAYOUT)
    ttnn_outputs = tt_model.pts_bbox_head(ttnn_img_feats, img_metas, prev_bev=None)

    tracy.signpost("stop")

    tt_bev_embed = ttnn_outputs["bev_embed"]
    if isinstance(tt_bev_embed, ttnn.Tensor):
        tt_bev_embed = ttnn.to_torch(tt_bev_embed).float()
    else:
        tt_bev_embed = tt_bev_embed.float()

    tt_all_cls_scores = ttnn_outputs["all_cls_scores"].float()
    tt_all_bbox_preds = ttnn_outputs["all_bbox_preds"].float()

    ref_bev_embed = torch_outputs["bev_embed"].float()
    ref_all_cls_scores = torch_outputs["all_cls_scores"].float()
    ref_all_bbox_preds = torch_outputs["all_bbox_preds"].float()

    assert_with_pcc(ref_bev_embed, tt_bev_embed, 0.98)
    assert_with_pcc(ref_all_cls_scores, tt_all_cls_scores, 0.98)
    assert_with_pcc(ref_all_bbox_preds, tt_all_bbox_preds, 0.98)
