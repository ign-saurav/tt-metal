# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import copy
import torch
import torch.nn as nn
from models.experimental.MapTR.dependency import HEADS, build_loss
from models.experimental.MapTR.dependency import DETRHead
from models.experimental.MapTR.dependency import build_bbox_coder
from models.experimental.MapTR.dependency import force_fp32
from models.experimental.MapTR.dependency import Linear, bias_init_with_prob
from models.experimental.MapTR.dependency import inverse_sigmoid

# from models.experimental.MapTR.dependency import bbox_xyxy_to_cxcywh, bbox_cxcywh_to_xyxy
from models.experimental.MapTR.projects.mmdet3d_plugin.core.bbox.coders.nms_free_coder import (
    bbox_xyxy_to_cxcywh,
    bbox_cxcywh_to_xyxy,
)
from models.experimental.MapTR.dependency import LearnedPositionalEncoding
from models.experimental.MapTR.dependency import build_transformer


def normalize_2d_bbox(bboxes, pc_range):
    patch_h = pc_range[4] - pc_range[1]
    patch_w = pc_range[3] - pc_range[0]
    cxcywh_bboxes = bbox_xyxy_to_cxcywh(bboxes)
    cxcywh_bboxes[..., 0:1] = cxcywh_bboxes[..., 0:1] - pc_range[0]
    cxcywh_bboxes[..., 1:2] = cxcywh_bboxes[..., 1:2] - pc_range[1]
    factor = bboxes.new_tensor([patch_w, patch_h, patch_w, patch_h])

    normalized_bboxes = cxcywh_bboxes / factor
    return normalized_bboxes


def normalize_2d_pts(pts, pc_range):
    patch_h = pc_range[4] - pc_range[1]
    patch_w = pc_range[3] - pc_range[0]
    new_pts = pts.clone()
    new_pts[..., 0:1] = pts[..., 0:1] - pc_range[0]
    new_pts[..., 1:2] = pts[..., 1:2] - pc_range[1]
    factor = pts.new_tensor([patch_w, patch_h])
    normalized_pts = new_pts / factor
    return normalized_pts


def denormalize_2d_bbox(bboxes, pc_range):
    bboxes = bbox_cxcywh_to_xyxy(bboxes)
    bboxes[..., 0::2] = bboxes[..., 0::2] * (pc_range[3] - pc_range[0]) + pc_range[0]
    bboxes[..., 1::2] = bboxes[..., 1::2] * (pc_range[4] - pc_range[1]) + pc_range[1]

    return bboxes


def denormalize_2d_pts(pts, pc_range):
    new_pts = pts.clone()
    new_pts[..., 0:1] = pts[..., 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0]
    new_pts[..., 1:2] = pts[..., 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1]
    return new_pts


@HEADS.register_module()
class MapTRHead(DETRHead):
    """Head of Detr3D.
    Args:
        with_box_refine (bool): Whether to refine the reference points
            in the decoder. Defaults to False.
        as_two_stage (bool) : Whether to generate the proposal from
            the outputs of encoder.
        transformer (obj:`ConfigDict`): ConfigDict is used for building
            the Encoder and Decoder.
        bev_h, bev_w (int): spatial shape of BEV queries.
    """

    def __init__(
        self,
        *args,
        with_box_refine=False,
        as_two_stage=False,
        transformer=None,
        bbox_coder=None,
        num_cls_fcs=2,
        code_weights=None,
        bev_h=30,
        bev_w=30,
        num_vec=20,
        num_pts_per_vec=2,
        num_pts_per_gt_vec=2,
        query_embed_type="all_pts",
        transform_method="minmax",
        gt_shift_pts_pattern="v0",
        dir_interval=1,
        # loss_pts=dict(type='ChamferDistance',
        #             loss_src_weight=1.0,
        #             loss_dst_weight=1.0),
        # loss_dir=dict(type='PtsDirCosLoss', loss_weight=2.0),
        loss_pts=None,
        loss_dir=None,
        **kwargs,
    ):
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.fp16_enabled = False

        self.with_box_refine = with_box_refine
        self.as_two_stage = as_two_stage
        self.bev_encoder_type = transformer.encoder.type
        if self.as_two_stage:
            transformer["as_two_stage"] = self.as_two_stage
        if "code_size" in kwargs:
            self.code_size = kwargs["code_size"]
        else:
            self.code_size = 10
        if code_weights is not None:
            self.code_weights = code_weights
        else:
            self.code_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]

        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.pc_range = self.bbox_coder.pc_range
        self.real_w = self.pc_range[3] - self.pc_range[0]
        self.real_h = self.pc_range[4] - self.pc_range[1]
        self.num_cls_fcs = num_cls_fcs - 1

        self.query_embed_type = query_embed_type
        self.transform_method = transform_method
        self.gt_shift_pts_pattern = gt_shift_pts_pattern
        num_query = num_vec * num_pts_per_vec
        self.num_query = num_query
        self.num_vec = num_vec
        self.num_pts_per_vec = num_pts_per_vec
        self.num_pts_per_gt_vec = num_pts_per_gt_vec
        self.dir_interval = dir_interval

        # Get embed_dims and num_layers from transformer config (needed before _init_layers)
        if transformer is not None and isinstance(transformer, dict):
            if "embed_dims" in transformer:
                self.embed_dims = transformer["embed_dims"]
            # Get num_layers from decoder config for _init_layers
            if "decoder" in transformer and isinstance(transformer["decoder"], dict):
                if "num_layers" in transformer["decoder"]:
                    self._decoder_num_layers = transformer["decoder"]["num_layers"]
                else:
                    self._decoder_num_layers = None
            else:
                self._decoder_num_layers = None
        else:
            self._decoder_num_layers = None

        super(MapTRHead, self).__init__(*args, transformer=transformer, **kwargs)

        # Build transformer if it's a config dict
        if transformer is not None and isinstance(transformer, dict):
            if not hasattr(self, "transformer") or self.transformer is None:
                self.transformer = build_transformer(transformer)

        # Set embed_dims if not set yet
        if not hasattr(self, "embed_dims") or self.embed_dims is None:
            if hasattr(self, "in_channels"):
                self.embed_dims = self.in_channels
            else:
                raise AttributeError("embed_dims not found in transformer config or in_channels")
        self.code_weights = nn.Parameter(torch.tensor(self.code_weights, requires_grad=False), requires_grad=False)
        # Only initialize losses if they are provided (for training)
        # For inference-only, losses can be None
        if loss_pts is not None:
            self.loss_pts = build_loss(loss_pts)
        else:
            self.loss_pts = None
        if loss_dir is not None:
            self.loss_dir = build_loss(loss_dir)
        else:
            self.loss_dir = None
        num_query = num_vec * num_pts_per_vec
        self.num_query = num_query
        self.num_vec = num_vec
        self.num_pts_per_vec = num_pts_per_vec
        self.num_pts_per_gt_vec = num_pts_per_gt_vec
        self._init_layers()

    def _init_layers(self):
        """Initialize classification branch and regression branch of head."""
        cls_branch = []
        # cls_branch.append(Linear(self.embed_dims * 2, self.embed_dims))
        # cls_branch.append(nn.LayerNorm(self.embed_dims))
        # cls_branch.append(nn.ReLU(inplace=True))
        for _ in range(self.num_reg_fcs):
            cls_branch.append(Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        cls_branch.append(Linear(self.embed_dims, self.cls_out_channels))
        fc_cls = nn.Sequential(*cls_branch)

        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, self.code_size))
        reg_branch = nn.Sequential(*reg_branch)

        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

        # last reg_branch is used to generate proposal from
        # encode feature map when as_two_stage is True.
        if hasattr(self, "transformer") and hasattr(self.transformer, "decoder"):
            num_layers = self.transformer.decoder.num_layers
        elif hasattr(self, "_decoder_num_layers") and self._decoder_num_layers is not None:
            num_layers = self._decoder_num_layers
        else:
            raise AttributeError("num_layers not found in transformer decoder")
        num_pred = (num_layers + 1) if self.as_two_stage else num_layers

        if self.with_box_refine:
            self.cls_branches = _get_clones(fc_cls, num_pred)
            self.reg_branches = _get_clones(reg_branch, num_pred)
        else:
            self.cls_branches = nn.ModuleList([fc_cls for _ in range(num_pred)])
            self.reg_branches = nn.ModuleList([reg_branch for _ in range(num_pred)])

        if not self.as_two_stage:
            if self.bev_encoder_type == "BEVFormerEncoder":
                self.bev_embedding = nn.Embedding(self.bev_h * self.bev_w, self.embed_dims)
                self.positional_encoding = LearnedPositionalEncoding(
                    self.embed_dims // 2, row_num_embed=self.bev_h, col_num_embed=self.bev_w
                )
            else:
                self.bev_embedding = None
                self.positional_encoding = None
            if self.query_embed_type == "all_pts":
                self.query_embedding = nn.Embedding(self.num_query, self.embed_dims * 2)
            elif self.query_embed_type == "instance_pts":
                self.query_embedding = None
                self.instance_embedding = nn.Embedding(self.num_vec, self.embed_dims * 2)
                self.pts_embedding = nn.Embedding(self.num_pts_per_vec, self.embed_dims * 2)

    def init_weights(self):
        """Initialize weights of the DeformDETR head."""
        self.transformer.init_weights()
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.cls_branches:
                nn.init.constant_(m[-1].bias, bias_init)
        # for m in self.reg_branches:
        #     constant_init(m[-1], 0, bias=0)
        # nn.init.constant_(self.reg_branches[0][-1].bias.data[2:], 0.)

    # @auto_fp16(apply_to=('mlvl_feats'))
    @force_fp32(apply_to=("mlvl_feats", "prev_bev"))
    def forward(self, mlvl_feats, lidar_feat, img_metas, prev_bev=None, only_bev=False):
        """Forward function.
        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 5D-tensor with shape
                (B, N, C, H, W).
            prev_bev: previous bev featues
            only_bev: only compute BEV features with encoder.
        Returns:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, theta, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
        """
        batch_id = str(img_metas[0].get("sample_idx", "unknown")) if img_metas else "unknown"
        bs, num_cam, _, _, _ = mlvl_feats[0].shape
        dtype = mlvl_feats[0].dtype
        # import pdb;pdb.set_trace()
        if self.query_embed_type == "all_pts":
            object_query_embeds = self.query_embedding.weight.to(dtype)
        elif self.query_embed_type == "instance_pts":
            pts_embeds = self.pts_embedding.weight.unsqueeze(0)
            instance_embeds = self.instance_embedding.weight.unsqueeze(1)
            object_query_embeds = (pts_embeds + instance_embeds).flatten(0, 1).to(dtype)
        if self.bev_embedding is not None:
            bev_queries = self.bev_embedding.weight.to(dtype)

            bev_mask = torch.zeros((bs, self.bev_h, self.bev_w), device=bev_queries.device).to(dtype)
            if self.positional_encoding is not None:
                bev_pos = self.positional_encoding(bev_mask).to(dtype)
            else:
                bev_pos = None
        else:
            bev_queries = None
            bev_mask = None
            bev_pos = None

        if only_bev:  # only use encoder to obtain BEV features, TODO: refine the workaround
            return self.transformer.get_bev_features(
                mlvl_feats,
                lidar_feat,
                bev_queries,
                self.bev_h,
                self.bev_w,
                grid_length=(self.real_h / self.bev_h, self.real_w / self.bev_w),
                bev_pos=bev_pos,
                img_metas=img_metas,
                prev_bev=prev_bev,
            )
        else:
            outputs = self.transformer(
                mlvl_feats,
                lidar_feat,
                bev_queries,
                object_query_embeds,
                self.bev_h,
                self.bev_w,
                grid_length=(self.real_h / self.bev_h, self.real_w / self.bev_w),
                bev_pos=bev_pos,
                reg_branches=self.reg_branches if self.with_box_refine else None,  # noqa:E501
                cls_branches=self.cls_branches if self.as_two_stage else None,
                img_metas=img_metas,
                prev_bev=prev_bev,
            )

        bev_embed, hs, init_reference, inter_references = outputs
        hs = hs.permute(0, 2, 1, 3)

        # Debug: Query initialization
        if hasattr(self, "_debug_enabled") and self._debug_enabled:
            print(f"\n=== Query Initialization Debug ===")
            print(f"Query embeds shape: {object_query_embeds.shape}")
            print(f"Query embeds range: [{object_query_embeds.min():.4f}, {object_query_embeds.max():.4f}]")
            if init_reference is not None:
                print(f"Init reference points shape: {init_reference.shape}")
                print(
                    f"Init reference points (normalized) range: [{init_reference.min():.4f}, {init_reference.max():.4f}]"
                )
                print(f"First 5 reference points (normalized): {init_reference[0, :5]}")
                # Denormalize reference points to check real coordinates
                ref_pts_real = denormalize_2d_pts(init_reference[0].view(-1, 2), self.pc_range)
                print(
                    f"Reference points (denormalized) range: X[{ref_pts_real[:, 0].min():.2f}, {ref_pts_real[:, 0].max():.2f}], Y[{ref_pts_real[:, 1].min():.2f}, {ref_pts_real[:, 1].max():.2f}]"
                )
                print(f"First 5 reference points (real): {ref_pts_real[:5]}")
                print(f"PC range: {self.pc_range}")
                # Visualize reference points
                try:
                    import matplotlib.pyplot as plt

                    ref_pts_cpu = init_reference[0].detach().cpu().numpy()
                    plt.figure(figsize=(10, 5))
                    plt.scatter(ref_pts_cpu[:, 0], ref_pts_cpu[:, 1], s=1, alpha=0.5)
                    plt.xlim([0, 1])
                    plt.ylim([0, 1])
                    plt.title("Decoder Query Reference Points (normalized)")
                    plt.xlabel("X (normalized)")
                    plt.ylabel("Y (normalized)")
                    plt.savefig("debug_reference_points.png")
                    plt.close()
                    print("Saved reference points visualization to debug_reference_points.png")
                except Exception as e:
                    print(f"Could not create reference points visualization: {e}")

        outputs_classes = []
        outputs_coords = []
        outputs_pts_coords = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                # import pdb;pdb.set_trace()
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            # import pdb;pdb.set_trace()
            # vec_embedding = hs[lvl].reshape(bs, self.num_vec, -1)
            outputs_class = self.cls_branches[lvl](hs[lvl].view(bs, self.num_vec, self.num_pts_per_vec, -1).mean(2))
            tmp = self.reg_branches[lvl](hs[lvl])

            # TODO: check the shape of reference
            assert reference.shape[-1] == 2
            tmp[..., 0:2] += reference[..., 0:2]
            # tmp[..., 0:2] = tmp[..., 0:2].sigmoid()
            tmp = tmp.sigmoid()  # cx,cy,w,h

            # Debug: Prediction head output (only for last decoder layer)
            if hasattr(self, "_debug_enabled") and self._debug_enabled and lvl == hs.shape[0] - 1:
                print(f"\n=== Prediction Head Output (Layer {lvl}) ===")
                print(f"Raw coords (tmp) shape: {tmp.shape}")
                print(f"Raw coords (normalized) range: [{tmp.min():.4f}, {tmp.max():.4f}]")
                print(f"First 3 normalized coords: {tmp[0, :3, :2]}")
                # Denormalize to check
                denorm_pts = denormalize_2d_pts(tmp.view(tmp.shape[0], -1, 2), self.pc_range)
                print(
                    f"Denorm coords range: X[{denorm_pts[:, 0].min():.2f}, {denorm_pts[:, 0].max():.2f}], Y[{denorm_pts[:, 1].min():.2f}, {denorm_pts[:, 1].max():.2f}]"
                )
                print(f"First 3 denormalized coords: {denorm_pts[:3]}")
                print(f"PC range: {self.pc_range}")
                print(
                    f"Reference points (before inverse_sigmoid) range: [{reference.min():.4f}, {reference.max():.4f}]"
                )
                print(f"Reference points (after inverse_sigmoid) first 3: {reference[0, :3]}")

            # import pdb;pdb.set_trace()
            # tmp[..., 0:1] = (tmp[..., 0:1] * (self.pc_range[3] -
            #                  self.pc_range[0]) + self.pc_range[0])
            # tmp[..., 1:2] = (tmp[..., 1:2] * (self.pc_range[4] -
            #                  self.pc_range[1]) + self.pc_range[1])
            # tmp = tmp.reshape(bs, self.num_vec,-1)
            # TODO: check if using sigmoid
            outputs_coord, outputs_pts_coord = self.transform_box(tmp)
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
            outputs_pts_coords.append(outputs_pts_coord)

        outputs_classes = torch.stack(outputs_classes)
        outputs_coords = torch.stack(outputs_coords)
        outputs_pts_coords = torch.stack(outputs_pts_coords)
        outs = {
            "bev_embed": bev_embed,
            "all_cls_scores": outputs_classes,
            "all_bbox_preds": outputs_coords,
            "all_pts_preds": outputs_pts_coords,
            "enc_cls_scores": None,
            "enc_bbox_preds": None,
            "enc_pts_preds": None,
        }

        return outs

    def transform_box(self, pts, y_first=False):
        """
        Converting the points set into bounding box.

        Args:
            pts: the input points sets (fields), each points
                set (fields) is represented as 2n scalar.
            y_first: if y_fisrt=True, the point set is represented as
                [y1, x1, y2, x2 ... yn, xn], otherwise the point set is
                represented as [x1, y1, x2, y2 ... xn, yn].
        Returns:
            The bbox [cx, cy, w, h] transformed from points.
        """
        pts_reshape = pts.view(pts.shape[0], self.num_vec, self.num_pts_per_vec, 2)
        pts_y = pts_reshape[:, :, :, 0] if y_first else pts_reshape[:, :, :, 1]
        pts_x = pts_reshape[:, :, :, 1] if y_first else pts_reshape[:, :, :, 0]
        if self.transform_method == "minmax":
            # import pdb;pdb.set_trace()

            xmin = pts_x.min(dim=2, keepdim=True)[0]
            xmax = pts_x.max(dim=2, keepdim=True)[0]
            ymin = pts_y.min(dim=2, keepdim=True)[0]
            ymax = pts_y.max(dim=2, keepdim=True)[0]
            bbox = torch.cat([xmin, ymin, xmax, ymax], dim=2)
            bbox = bbox_xyxy_to_cxcywh(bbox)
        else:
            raise NotImplementedError
        return bbox, pts_reshape

    def loss(self, *args, **kwargs):
        raise NotImplementedError("MapTRHead training / loss computation has been removed in this reference build.")

    @force_fp32(apply_to=("preds_dicts"))
    def get_bboxes(self, preds_dicts, img_metas, rescale=False):
        """Generate bboxes from bbox head predictions.
        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
            img_metas (list[dict]): Point cloud and image's meta info.
        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """
        # bboxes: xmin, ymin, xmax, ymax
        preds_dicts = self.bbox_coder.decode(preds_dicts)

        num_samples = len(preds_dicts)
        ret_list = []
        for i in range(num_samples):
            preds = preds_dicts[i]
            bboxes = preds["bboxes"]
            # bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5

            # code_size = bboxes.shape[-1]
            # bboxes = img_metas[i]['box_type_3d'](bboxes, code_size)
            scores = preds["scores"]
            labels = preds["labels"]
            pts = preds["pts"]

            ret_list.append([bboxes, scores, labels, pts])

        return ret_list
