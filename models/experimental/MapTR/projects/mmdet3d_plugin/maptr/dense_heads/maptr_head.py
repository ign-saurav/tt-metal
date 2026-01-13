import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
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
from models.experimental.MapTR.dependency import multi_apply, multi_apply, reduce_mean
from models.experimental.MapTR.dependency import TORCH_VERSION, digit_version
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

    def _get_target_single(
        self, cls_score, bbox_pred, pts_pred, gt_labels, gt_bboxes, gt_shifts_pts, gt_bboxes_ignore=None
    ):
        """ "Compute regression and classification targets for one image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_score (Tensor): Box score logits from a single decoder layer
                for one image. Shape [num_query, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from a single decoder layer
                for one image, with normalized coordinate (cx, cy, w, h) and
                shape [num_query, 4].
            gt_bboxes (Tensor): Ground truth bboxes for one image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (Tensor): Ground truth class indices for one image
                with shape (num_gts, ).
            gt_bboxes_ignore (Tensor, optional): Bounding boxes
                which can be ignored. Default None.
        Returns:
            tuple[Tensor]: a tuple containing the following for one image.
                - labels (Tensor): Labels of each image.
                - label_weights (Tensor]): Label weights of each image.
                - bbox_targets (Tensor): BBox targets of each image.
                - bbox_weights (Tensor): BBox weights of each image.
                - pos_inds (Tensor): Sampled positive indices for each image.
                - neg_inds (Tensor): Sampled negative indices for each image.
        """
        # import pdb;pdb.set_trace()
        num_bboxes = bbox_pred.size(0)
        # assigner and sampler
        gt_c = gt_bboxes.shape[-1]
        # import pdb;pdb.set_trace()
        assign_result, order_index = self.assigner.assign(
            bbox_pred, cls_score, pts_pred, gt_bboxes, gt_labels, gt_shifts_pts, gt_bboxes_ignore
        )

        sampling_result = self.sampler.sample(assign_result, bbox_pred, gt_bboxes)
        # pts_sampling_result = self.sampler.sample(assign_result, pts_pred,
        #                                       gt_pts)

        # import pdb;pdb.set_trace()
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # label targets
        labels = gt_bboxes.new_full((num_bboxes,), self.num_classes, dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # bbox targets
        bbox_targets = torch.zeros_like(bbox_pred)[..., :gt_c]
        bbox_weights = torch.zeros_like(bbox_pred)
        bbox_weights[pos_inds] = 1.0

        # pts targets
        # import pdb;pdb.set_trace()
        # pts_targets = torch.zeros_like(pts_pred)
        # num_query, num_order, num_points, num_coords
        if order_index is None:
            # import pdb;pdb.set_trace()
            assigned_shift = gt_labels[sampling_result.pos_assigned_gt_inds]
        else:
            assigned_shift = order_index[sampling_result.pos_inds, sampling_result.pos_assigned_gt_inds]
        pts_targets = pts_pred.new_zeros((pts_pred.size(0), pts_pred.size(1), pts_pred.size(2)))
        pts_weights = torch.zeros_like(pts_targets)
        pts_weights[pos_inds] = 1.0

        # DETR
        bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes
        pts_targets[pos_inds] = gt_shifts_pts[sampling_result.pos_assigned_gt_inds, assigned_shift, :, :]
        return (labels, label_weights, bbox_targets, bbox_weights, pts_targets, pts_weights, pos_inds, neg_inds)

    def get_targets(
        self,
        cls_scores_list,
        bbox_preds_list,
        pts_preds_list,
        gt_bboxes_list,
        gt_labels_list,
        gt_shifts_pts_list,
        gt_bboxes_ignore_list=None,
    ):
        """"Compute regression and classification targets for a batch image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_scores_list (list[Tensor]): Box score logits from a single
                decoder layer for each image with shape [num_query,
                cls_out_channels].
            bbox_preds_list (list[Tensor]): Sigmoid outputs from a single
                decoder layer for each image, with normalized coordinate
                (cx, cy, w, h) and shape [num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            tuple: a tuple containing the following targets.
                - labels_list (list[Tensor]): Labels for all images.
                - label_weights_list (list[Tensor]): Label weights for all \
                    images.
                - bbox_targets_list (list[Tensor]): BBox targets for all \
                    images.
                - bbox_weights_list (list[Tensor]): BBox weights for all \
                    images.
                - num_total_pos (int): Number of positive samples in all \
                    images.
                - num_total_neg (int): Number of negative samples in all \
                    images.
        """
        assert gt_bboxes_ignore_list is None, "Only supports for gt_bboxes_ignore setting to None."
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [gt_bboxes_ignore_list for _ in range(num_imgs)]

        (
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            pts_targets_list,
            pts_weights_list,
            pos_inds_list,
            neg_inds_list,
        ) = multi_apply(
            self._get_target_single,
            cls_scores_list,
            bbox_preds_list,
            pts_preds_list,
            gt_labels_list,
            gt_bboxes_list,
            gt_shifts_pts_list,
            gt_bboxes_ignore_list,
        )
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            pts_targets_list,
            pts_weights_list,
            num_total_pos,
            num_total_neg,
        )

    def loss_single(
        self,
        cls_scores,
        bbox_preds,
        pts_preds,
        gt_bboxes_list,
        gt_labels_list,
        gt_shifts_pts_list,
        gt_bboxes_ignore_list=None,
    ):
        """ "Loss function for outputs from a single decoder layer of a single
        feature level.
        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images. Shape [bs, num_query, cls_out_channels].
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape [bs, num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_pts_list (list[Tensor]): Ground truth pts for each image
                with shape (num_gts, fixed_num, 2) in [x,y] format.
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        pts_preds_list = [pts_preds[i] for i in range(num_imgs)]
        # import pdb;pdb.set_trace()
        cls_reg_targets = self.get_targets(
            cls_scores_list,
            bbox_preds_list,
            pts_preds_list,
            gt_bboxes_list,
            gt_labels_list,
            gt_shifts_pts_list,
            gt_bboxes_ignore_list,
        )
        (
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            pts_targets_list,
            pts_weights_list,
            num_total_pos,
            num_total_neg,
        ) = cls_reg_targets
        # import pdb;pdb.set_trace()
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)
        pts_targets = torch.cat(pts_targets_list, 0)
        pts_weights = torch.cat(pts_weights_list, 0)

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(cls_scores.new_tensor([cls_avg_factor]))

        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_cls(cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes accross all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # import pdb;pdb.set_trace()
        # regression L1 loss
        bbox_preds = bbox_preds.reshape(-1, bbox_preds.size(-1))
        normalized_bbox_targets = normalize_2d_bbox(bbox_targets, self.pc_range)
        # normalized_bbox_targets = bbox_targets
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
        bbox_weights = bbox_weights * self.code_weights

        loss_bbox = self.loss_bbox(
            bbox_preds[isnotnan, :4],
            normalized_bbox_targets[isnotnan, :4],
            bbox_weights[isnotnan, :4],
            avg_factor=num_total_pos,
        )

        # regression pts CD loss
        # pts_preds = pts_preds
        # import pdb;pdb.set_trace()

        # num_samples, num_order, num_pts, num_coords
        normalized_pts_targets = normalize_2d_pts(pts_targets, self.pc_range)

        # num_samples, num_pts, num_coords
        pts_preds = pts_preds.reshape(-1, pts_preds.size(-2), pts_preds.size(-1))
        if self.num_pts_per_vec != self.num_pts_per_gt_vec:
            pts_preds = pts_preds.permute(0, 2, 1)
            pts_preds = F.interpolate(pts_preds, size=(self.num_pts_per_gt_vec), mode="linear", align_corners=True)
            pts_preds = pts_preds.permute(0, 2, 1).contiguous()

        # import pdb;pdb.set_trace()
        if self.loss_pts is not None:
            loss_pts = self.loss_pts(
                pts_preds[isnotnan, :, :],
                normalized_pts_targets[isnotnan, :, :],
                pts_weights[isnotnan, :, :],
                avg_factor=num_total_pos,
            )
        else:
            loss_pts = torch.tensor(0.0, device=pts_preds.device)
        dir_weights = pts_weights[:, : -self.dir_interval, 0]
        denormed_pts_preds = denormalize_2d_pts(pts_preds, self.pc_range)
        denormed_pts_preds_dir = (
            denormed_pts_preds[:, self.dir_interval :, :] - denormed_pts_preds[:, : -self.dir_interval, :]
        )
        pts_targets_dir = pts_targets[:, self.dir_interval :, :] - pts_targets[:, : -self.dir_interval, :]
        # dir_weights = pts_weights[:, indice,:-1,0]
        # import pdb;pdb.set_trace()
        if self.loss_dir is not None:
            loss_dir = self.loss_dir(
                denormed_pts_preds_dir[isnotnan, :, :],
                pts_targets_dir[isnotnan, :, :],
                dir_weights[isnotnan, :],
                avg_factor=num_total_pos,
            )
        else:
            loss_dir = torch.tensor(0.0, device=pts_preds.device)

        bboxes = denormalize_2d_bbox(bbox_preds, self.pc_range)
        # regression IoU loss, defaultly GIoU loss
        loss_iou = self.loss_iou(
            bboxes[isnotnan, :4], bbox_targets[isnotnan, :4], bbox_weights[isnotnan, :4], avg_factor=num_total_pos
        )

        if digit_version(TORCH_VERSION) >= digit_version("1.8"):
            loss_cls = torch.nan_to_num(loss_cls)
            loss_bbox = torch.nan_to_num(loss_bbox)
            loss_iou = torch.nan_to_num(loss_iou)
            loss_pts = torch.nan_to_num(loss_pts)
            loss_dir = torch.nan_to_num(loss_dir)
        return loss_cls, loss_bbox, loss_iou, loss_pts, loss_dir

    @force_fp32(apply_to=("preds_dicts"))
    def loss(self, gt_bboxes_list, gt_labels_list, preds_dicts, gt_bboxes_ignore=None, img_metas=None):
        """ "Loss function.
        Args:

            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            preds_dicts:
                all_cls_scores (Tensor): Classification score of all
                    decoder layers, has shape
                    [nb_dec, bs, num_query, cls_out_channels].
                all_bbox_preds (Tensor): Sigmoid regression
                    outputs of all decode layers. Each is a 4D-tensor with
                    normalized coordinate format (cx, cy, w, h) and shape
                    [nb_dec, bs, num_query, 4].
                enc_cls_scores (Tensor): Classification scores of
                    points on encode feature map , has shape
                    (N, h*w, num_classes). Only be passed when as_two_stage is
                    True, otherwise is None.
                enc_bbox_preds (Tensor): Regression results of each points
                    on the encode feature map, has shape (N, h*w, 4). Only be
                    passed when as_two_stage is True, otherwise is None.
            gt_bboxes_ignore (list[Tensor], optional): Bounding boxes
                which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert gt_bboxes_ignore is None, (
            f"{self.__class__.__name__} only supports " f"for gt_bboxes_ignore setting to None."
        )
        gt_vecs_list = copy.deepcopy(gt_bboxes_list)
        # import pdb;pdb.set_trace()
        all_cls_scores = preds_dicts["all_cls_scores"]
        all_bbox_preds = preds_dicts["all_bbox_preds"]
        all_pts_preds = preds_dicts["all_pts_preds"]
        enc_cls_scores = preds_dicts["enc_cls_scores"]
        enc_bbox_preds = preds_dicts["enc_bbox_preds"]
        enc_pts_preds = preds_dicts["enc_pts_preds"]

        num_dec_layers = len(all_cls_scores)
        device = gt_labels_list[0].device

        # gt_bboxes_list = [torch.cat(
        #     (gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]),
        #     dim=1).to(device) for gt_bboxes in gt_bboxes_list]
        # import pdb;pdb.set_trace()
        # gt_bboxes_list = [
        #     gt_bboxes.to(device) for gt_bboxes in gt_bboxes_list]
        gt_bboxes_list = [gt_bboxes.bbox.to(device) for gt_bboxes in gt_vecs_list]
        gt_pts_list = [gt_bboxes.fixed_num_sampled_points.to(device) for gt_bboxes in gt_vecs_list]
        if self.gt_shift_pts_pattern == "v0":
            gt_shifts_pts_list = [gt_bboxes.shift_fixed_num_sampled_points.to(device) for gt_bboxes in gt_vecs_list]
        elif self.gt_shift_pts_pattern == "v1":
            gt_shifts_pts_list = [gt_bboxes.shift_fixed_num_sampled_points_v1.to(device) for gt_bboxes in gt_vecs_list]
        elif self.gt_shift_pts_pattern == "v2":
            gt_shifts_pts_list = [gt_bboxes.shift_fixed_num_sampled_points_v2.to(device) for gt_bboxes in gt_vecs_list]
        elif self.gt_shift_pts_pattern == "v3":
            gt_shifts_pts_list = [gt_bboxes.shift_fixed_num_sampled_points_v3.to(device) for gt_bboxes in gt_vecs_list]
        elif self.gt_shift_pts_pattern == "v4":
            gt_shifts_pts_list = [gt_bboxes.shift_fixed_num_sampled_points_v4.to(device) for gt_bboxes in gt_vecs_list]
        else:
            raise NotImplementedError
        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_pts_list = [gt_pts_list for _ in range(num_dec_layers)]
        all_gt_shifts_pts_list = [gt_shifts_pts_list for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [gt_bboxes_ignore for _ in range(num_dec_layers)]
        # import pdb;pdb.set_trace()
        losses_cls, losses_bbox, losses_iou, losses_pts, losses_dir = multi_apply(
            self.loss_single,
            all_cls_scores,
            all_bbox_preds,
            all_pts_preds,
            all_gt_bboxes_list,
            all_gt_labels_list,
            all_gt_shifts_pts_list,
            all_gt_bboxes_ignore_list,
        )

        loss_dict = dict()
        # loss of proposal generated from encode feature map.
        if enc_cls_scores is not None:
            binary_labels_list = [torch.zeros_like(gt_labels_list[i]) for i in range(len(all_gt_labels_list))]
            # TODO bug here
            enc_loss_cls, enc_losses_bbox, enc_losses_iou, enc_losses_pts, enc_losses_dir = self.loss_single(
                enc_cls_scores,
                enc_bbox_preds,
                enc_pts_preds,
                gt_bboxes_list,
                binary_labels_list,
                gt_pts_list,
                gt_bboxes_ignore,
            )
            loss_dict["enc_loss_cls"] = enc_loss_cls
            loss_dict["enc_loss_bbox"] = enc_losses_bbox
            loss_dict["enc_losses_iou"] = enc_losses_iou
            loss_dict["enc_losses_pts"] = enc_losses_pts
            loss_dict["enc_losses_dir"] = enc_losses_dir

        # loss from the last decoder layer
        loss_dict["loss_cls"] = losses_cls[-1]
        loss_dict["loss_bbox"] = losses_bbox[-1]
        loss_dict["loss_iou"] = losses_iou[-1]
        loss_dict["loss_pts"] = losses_pts[-1]
        loss_dict["loss_dir"] = losses_dir[-1]
        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i, loss_iou_i, loss_pts_i, loss_dir_i in zip(
            losses_cls[:-1], losses_bbox[:-1], losses_iou[:-1], losses_pts[:-1], losses_dir[:-1]
        ):
            loss_dict[f"d{num_dec_layer}.loss_cls"] = loss_cls_i
            loss_dict[f"d{num_dec_layer}.loss_bbox"] = loss_bbox_i
            loss_dict[f"d{num_dec_layer}.loss_iou"] = loss_iou_i
            loss_dict[f"d{num_dec_layer}.loss_pts"] = loss_pts_i
            loss_dict[f"d{num_dec_layer}.loss_dir"] = loss_dir_i
            num_dec_layer += 1
        return loss_dict

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
