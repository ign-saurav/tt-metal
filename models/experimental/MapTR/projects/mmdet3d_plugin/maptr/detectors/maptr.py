import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.experimental.MapTR.dependency import DETECTORS
from models.experimental.MapTR.dependency import MVXTwoStageDetector
from models.experimental.MapTR.projects.mmdet3d_plugin.models.utils.grid_mask import GridMask
from models.experimental.MapTR.dependency import force_fp32, auto_fp16
from models.experimental.MapTR.dependency import Voxelization, DynamicScatter
from models.experimental.MapTR.dependency import builder


@DETECTORS.register_module()
class MapTR(MVXTwoStageDetector):
    """MapTR.
    Args:
        video_test_mode (bool): Decide whether to use temporal information during inference.
    """

    def __init__(
        self,
        use_grid_mask=False,
        pts_voxel_layer=None,
        pts_voxel_encoder=None,
        pts_middle_encoder=None,
        pts_fusion_layer=None,
        img_backbone=None,
        pts_backbone=None,
        img_neck=None,
        pts_neck=None,
        pts_bbox_head=None,
        img_roi_head=None,
        img_rpn_head=None,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
        video_test_mode=False,
        modality="vision",
        lidar_encoder=None,
    ):
        super(MapTR, self).__init__(
            pts_voxel_layer,
            pts_voxel_encoder,
            pts_middle_encoder,
            pts_fusion_layer,
            img_backbone,
            pts_backbone,
            img_neck,
            pts_neck,
            pts_bbox_head,
            img_roi_head,
            img_rpn_head,
            train_cfg,
            test_cfg,
            pretrained,
        )
        self.grid_mask = GridMask(True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask
        self.fp16_enabled = False

        # temporal
        self.video_test_mode = video_test_mode
        self.prev_frame_info = {
            "prev_bev": None,
            "scene_token": None,
            "prev_pos": 0,
            "prev_angle": 0,
        }
        self.modality = modality
        if self.modality == "fusion" and lidar_encoder is not None:
            if lidar_encoder["voxelize"].get("max_num_points", -1) > 0:
                voxelize_module = Voxelization(**lidar_encoder["voxelize"])
            else:
                voxelize_module = DynamicScatter(**lidar_encoder["voxelize"])
            self.lidar_modal_extractor = nn.ModuleDict(
                {
                    "voxelize": voxelize_module,
                    "backbone": builder.build_middle_encoder(lidar_encoder["backbone"]),
                }
            )
            self.voxelize_reduce = lidar_encoder.get("voxelize_reduce", True)

    def extract_img_feat(self, img, img_metas, len_queue=None):
        """Extract features of images."""
        import numpy as np
        import os

        batch_id = str(img_metas[0].get("sample_idx", "unknown")) if img_metas else "unknown"
        if img is not None:
            # Convert from HWC [B, H, W, C] to CHW [B, C, H, W] if needed
            if img.dim() == 4 and img.shape[-1] == 3:
                img = img.permute(0, 3, 1, 2)

            # Match model dtype (convert to half if model is half)
            if hasattr(self.img_backbone, "parameters"):
                model_dtype = next(self.img_backbone.parameters()).dtype
                if model_dtype != img.dtype:
                    img = img.to(model_dtype)

            B = img.size(0)
            # input_shape = img.shape[-2:]
            # # update real input shape of each single img
            # for img_meta in img_metas:
            #     img_meta.update(input_shape=input_shape)

            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_()
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.reshape(B * N, C, H, W)
            if self.use_grid_mask:
                img = self.grid_mask(img)
            # Save input img before processing
            # if "img" in kwargs and kwargs["img"] is not None:
            print("Saving input img before processing", img.shape)
            import numpy as np
            import os

            batch_id = str(img_metas[0].get("sample_idx", "unknown"))
            os.makedirs("models/experimental/MapTR/numpy_feats/input", exist_ok=True)
            img = img
            img_np = img.detach().cpu().numpy() if isinstance(img, torch.Tensor) else np.array(img)
            np.save(f"models/experimental/MapTR/numpy_feats/input/input_{batch_id}.npy", img_np)

            ####################################################
            img = np.load(
                f"models/experimental/MapTR/numpy_feats_working/input/backbone_input_3e8750f331d7499e9b5123e9eb70f2e2.npy"
            )
            # Convert to torch tensor
            device = torch.device("cpu")
            dtype = torch.float32
            img = torch.from_numpy(img).to(device=device, dtype=dtype)
            print("Saving input img after processing", img.shape)
            ####################################################

            img_feats = self.img_backbone(img)
            save_dir = "models/experimental/MapTR/numpy_feats/backbone"
            os.makedirs(save_dir, exist_ok=True)
            backbone_feat = img_feats[0] if isinstance(img_feats, (list, tuple)) else img_feats
            np.save(os.path.join(save_dir, f"backbone_{batch_id}.npy"), backbone_feat.detach().cpu().numpy())
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)
            save_dir = "models/experimental/MapTR/numpy_feats/neck"
            os.makedirs(save_dir, exist_ok=True)
            neck_feat = img_feats[0] if isinstance(img_feats, (list, tuple)) else img_feats
            np.save(os.path.join(save_dir, f"neck_{batch_id}.npy"), neck_feat.detach().cpu().numpy())

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            if len_queue is not None:
                img_feats_reshaped.append(img_feat.view(int(B / len_queue), len_queue, int(BN / B), C, H, W))
            else:
                img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        return img_feats_reshaped

    @auto_fp16(apply_to=("img"), out_fp32=True)
    def extract_feat(self, img, img_metas=None, len_queue=None):
        """Extract features from images and points."""

        img_feats = self.extract_img_feat(img, img_metas, len_queue=len_queue)

        return img_feats

    def forward_pts_train(
        self, pts_feats, lidar_feat, gt_bboxes_3d, gt_labels_3d, img_metas, gt_bboxes_ignore=None, prev_bev=None
    ):
        """Forward function'
        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.
            prev_bev (torch.Tensor, optional): BEV features of previous frame.
        Returns:
            dict: Losses of each branch.
        """

        outs = self.pts_bbox_head(pts_feats, lidar_feat, img_metas, prev_bev)
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
        losses = self.pts_bbox_head.loss(*loss_inputs, img_metas=img_metas)
        return losses

    def forward_dummy(self, img):
        dummy_metas = None
        return self.forward_test(img=img, img_metas=[[dummy_metas]])

    def forward(self, return_loss=True, **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.
        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        # # Save input img before processing
        # if "img" in kwargs and kwargs["img"] is not None:
        #     import numpy as np
        #     import os
        #     batch_id = str(kwargs.get('sample_idx', 'unknown'))
        #     os.makedirs('models/experimental/MapTR/numpy_feats/input', exist_ok=True)
        #     img = kwargs["img"]
        #     img_np = img.detach().cpu().numpy() if isinstance(img, torch.Tensor) else np.array(img)
        #     np.save(f'models/experimental/MapTR/numpy_feats/input/input_{batch_id}.npy', img_np)

        if return_loss:
            return self.forward_train(**kwargs)
        else:
            # img_metas is not in kwargs - metadata fields are directly in kwargs
            # Construct img_metas from the individual metadata fields
            # Determine number of cameras first

            num_cams = 6  # default
            if "img" in kwargs and kwargs["img"] is not None:
                img = kwargs["img"]
                if isinstance(img, list):
                    num_cams = len(img)
                elif isinstance(img, torch.Tensor):
                    # img shape: [B, N, C, H, W] or [B, C, H, W]
                    if img.dim() == 5:
                        num_cams = img.shape[1]
                    else:
                        num_cams = 1

            # Get lidar2img from kwargs or create default identity matrices
            lidar2img = kwargs.get("lidar2img", None)
            if lidar2img is None:
                # Create default identity 4x4 matrices for each camera
                lidar2img = [np.eye(4, dtype=np.float32) for _ in range(num_cams)]

            # Get img_shape and ensure it's in the correct format (list of [H, W] per camera)
            img_shape = kwargs.get("img_shape", None)
            if img_shape is None:
                # Try to infer from img tensor
                if "img" in kwargs and kwargs["img"] is not None:
                    img = kwargs["img"]
                    if isinstance(img, torch.Tensor):
                        if img.dim() == 5:  # [B, N, C, H, W]
                            h, w = img.shape[3], img.shape[4]
                            img_shape = [[h, w] for _ in range(img.shape[1])]  # One per camera
                        elif img.dim() == 4:  # [B, C, H, W]
                            h, w = img.shape[2], img.shape[3]
                            img_shape = [[h, w]]
                        else:
                            img_shape = [[900, 1600] for _ in range(num_cams)]  # default
                    else:
                        img_shape = [[900, 1600] for _ in range(num_cams)]  # default
                else:
                    img_shape = [[900, 1600] for _ in range(num_cams)]  # default
            elif not isinstance(img_shape, list) or (len(img_shape) > 0 and not isinstance(img_shape[0], list)):
                # Convert single shape to list format
                if isinstance(img_shape, (tuple, list)) and len(img_shape) >= 2:
                    img_shape = [[img_shape[0], img_shape[1]] for _ in range(num_cams)]
                else:
                    img_shape = [[900, 1600] for _ in range(num_cams)]  # default

            img_metas = [
                {
                    "scene_token": kwargs.get("scene_token", ""),
                    "can_bus": kwargs.get("can_bus", [0.0, 0.0, 0.0, 0.0]),
                    "img_shape": img_shape,
                    "ori_shape": kwargs.get("ori_shape", img_shape),
                    "pad_shape": kwargs.get("pad_shape", img_shape),
                    "scale_factor": kwargs.get("scale_factor", None),
                    "img_norm_cfg": kwargs.get("img_norm_cfg", None),
                    "sample_idx": kwargs.get("sample_idx", None),
                    "pts_filename": kwargs.get("pts_filename", None),
                    "lidar_path": kwargs.get("lidar_path", None),
                    "timestamp": kwargs.get("timestamp", None),
                    "frame_idx": kwargs.get("frame_idx", None),
                    "map_location": kwargs.get("map_location", None),
                    "lidar2img": lidar2img,
                }
            ]
            # Wrap in double list structure as expected by forward_test
            img_metas = [img_metas]

            # Fix img format: convert from HWC [B, H, W, C] to CHW [B, C, H, W]
            if "img" in kwargs and kwargs["img"] is not None:
                img = kwargs["img"]
                if isinstance(img, list):
                    img = torch.stack(img)
                if isinstance(img, torch.Tensor) and img.dim() == 4 and img.shape[-1] == 3:
                    img = img.permute(0, 3, 1, 2)
                kwargs["img"] = img

            return self.forward_test(img_metas, **kwargs)

    def obtain_history_bev(self, imgs_queue, img_metas_list):
        """Obtain history BEV features iteratively. To save GPU memory, gradients are not calculated."""
        self.eval()

        with torch.no_grad():
            prev_bev = None
            bs, len_queue, num_cams, C, H, W = imgs_queue.shape
            imgs_queue = imgs_queue.reshape(bs * len_queue, num_cams, C, H, W)
            img_feats_list = self.extract_feat(img=imgs_queue, len_queue=len_queue)
            for i in range(len_queue):
                img_metas = [each[i] for each in img_metas_list]
                if not img_metas[0]["prev_bev_exists"]:
                    prev_bev = None
                img_feats = [each_scale[:, i] for each_scale in img_feats_list]
                prev_bev = self.pts_bbox_head(img_feats, None, img_metas, prev_bev, only_bev=True)
            self.train()
            return prev_bev

    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points):
        feats, coords, sizes = [], [], []
        for k, res in enumerate(points):
            ret = self.lidar_modal_extractor["voxelize"](res)
            if len(ret) == 3:
                # hard voxelize
                f, c, n = ret
            else:
                assert len(ret) == 2
                f, c = ret
                n = None
            feats.append(f)
            coords.append(F.pad(c, (1, 0), mode="constant", value=k))
            if n is not None:
                sizes.append(n)

        feats = torch.cat(feats, dim=0)
        coords = torch.cat(coords, dim=0)
        if len(sizes) > 0:
            sizes = torch.cat(sizes, dim=0)
            if self.voxelize_reduce:
                feats = feats.sum(dim=1, keepdim=False) / sizes.type_as(feats).view(-1, 1)
                feats = feats.contiguous()

        return feats, coords, sizes

    @auto_fp16(apply_to=("points"), out_fp32=True)
    def extract_lidar_feat(self, points):
        feats, coords, sizes = self.voxelize(points)
        # voxel_features = self.lidar_modal_extractor["voxel_encoder"](feats, sizes, coords)
        batch_size = coords[-1, 0] + 1
        lidar_feat = self.lidar_modal_extractor["backbone"](feats, coords, batch_size, sizes=sizes)

        return lidar_feat

    # @auto_fp16(apply_to=('img', 'points'))
    @force_fp32(apply_to=("img", "points", "prev_bev"))
    def forward_train(
        self,
        points=None,
        img_metas=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        gt_labels=None,
        gt_bboxes=None,
        img=None,
        proposals=None,
        gt_bboxes_ignore=None,
        img_depth=None,
        img_mask=None,
    ):
        """Forward training function.
        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.
        Returns:
            dict: Losses of different branches.
        """
        lidar_feat = None
        if self.modality == "fusion":
            lidar_feat = self.extract_lidar_feat(points)

        len_queue = img.size(1)
        prev_img = img[:, :-1, ...]
        img = img[:, -1, ...]

        prev_img_metas = copy.deepcopy(img_metas)
        # prev_bev = self.obtain_history_bev(prev_img, prev_img_metas)
        # import pdb;pdb.set_trace()
        prev_bev = self.obtain_history_bev(prev_img, prev_img_metas) if len_queue > 1 else None

        img_metas = [each[len_queue - 1] for each in img_metas]
        if not img_metas[0]["prev_bev_exists"]:
            prev_bev = None
        img_feats = self.extract_feat(img=img, img_metas=img_metas)
        losses = dict()
        losses_pts = self.forward_pts_train(
            img_feats, lidar_feat, gt_bboxes_3d, gt_labels_3d, img_metas, gt_bboxes_ignore, prev_bev
        )

        losses.update(losses_pts)
        return losses

    def forward_test(self, img_metas, img=None, points=None, **kwargs):
        for var, name in [(img_metas, "img_metas")]:
            if not isinstance(var, list):
                raise TypeError("{} must be a list, but got {}".format(name, type(var)))
        # Convert img from HWC to CHW if needed before wrapping in list
        if img is not None and isinstance(img, torch.Tensor) and img.dim() == 4 and img.shape[-1] == 3:
            img = img.permute(0, 3, 1, 2)
        img = [img] if img is None else img
        points = [points] if points is None else points
        if img_metas[0][0]["scene_token"] != self.prev_frame_info["scene_token"]:
            # the first sample of each scene is truncated
            self.prev_frame_info["prev_bev"] = None
        # update idx
        self.prev_frame_info["scene_token"] = img_metas[0][0]["scene_token"]

        # do not use temporal information
        if not self.video_test_mode:
            self.prev_frame_info["prev_bev"] = None

        # Get the delta of ego position and angle between two timestamps.
        tmp_pos = copy.deepcopy(img_metas[0][0]["can_bus"][:3])
        tmp_angle = copy.deepcopy(img_metas[0][0]["can_bus"][-1])
        if self.prev_frame_info["prev_bev"] is not None:
            img_metas[0][0]["can_bus"][:3] -= self.prev_frame_info["prev_pos"]
            img_metas[0][0]["can_bus"][-1] -= self.prev_frame_info["prev_angle"]
        else:
            img_metas[0][0]["can_bus"][-1] = 0
            img_metas[0][0]["can_bus"][:3] = 0

        new_prev_bev, bbox_results = self.simple_test(
            img_metas[0], img[0], points[0], prev_bev=self.prev_frame_info["prev_bev"], **kwargs
        )
        # During inference, we save the BEV features and ego motion of each timestamp.
        self.prev_frame_info["prev_pos"] = tmp_pos
        self.prev_frame_info["prev_angle"] = tmp_angle
        self.prev_frame_info["prev_bev"] = new_prev_bev
        return bbox_results

    def pred2result(self, bboxes, scores, labels, pts, attrs=None):
        """Convert detection results to a list of numpy arrays.

        Args:
            bboxes (torch.Tensor): Bounding boxes with shape of (n, 5).
            labels (torch.Tensor): Labels with shape of (n, ).
            scores (torch.Tensor): Scores with shape of (n, ).
            attrs (torch.Tensor, optional): Attributes with shape of (n, ). \
                Defaults to None.

        Returns:
            dict[str, torch.Tensor]: Bounding box results in cpu mode.

                - boxes_3d (torch.Tensor): 3D boxes.
                - scores (torch.Tensor): Prediction scores.
                - labels_3d (torch.Tensor): Box labels.
                - attrs_3d (torch.Tensor, optional): Box attributes.
        """
        result_dict = dict(
            boxes_3d=bboxes.to("cpu"), scores_3d=scores.cpu(), labels_3d=labels.cpu(), pts_3d=pts.to("cpu")
        )

        if attrs is not None:
            result_dict["attrs_3d"] = attrs.cpu()

        return result_dict

    def simple_test_pts(self, x, lidar_feat, img_metas, prev_bev=None, rescale=False):
        """Test function"""
        outs = self.pts_bbox_head(x, lidar_feat, img_metas, prev_bev=prev_bev)

        bbox_list = self.pts_bbox_head.get_bboxes(outs, img_metas, rescale=rescale)

        bbox_results = [self.pred2result(bboxes, scores, labels, pts) for bboxes, scores, labels, pts in bbox_list]
        # import pdb;pdb.set_trace()
        return outs["bev_embed"], bbox_results

    def simple_test(self, img_metas, img=None, points=None, prev_bev=None, rescale=False, **kwargs):
        """Test function without augmentaiton."""
        lidar_feat = None
        if self.modality == "fusion":
            lidar_feat = self.extract_lidar_feat(points)
        img_feats = self.extract_feat(img=img, img_metas=img_metas)

        bbox_list = [dict() for i in range(len(img_metas))]
        new_prev_bev, bbox_pts = self.simple_test_pts(img_feats, lidar_feat, img_metas, prev_bev, rescale=rescale)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict["pts_bbox"] = pts_bbox
        return new_prev_bev, bbox_list


@DETECTORS.register_module()
class MapTR_fp16(MapTR):
    """
    The default version BEVFormer currently can not support FP16.
    We provide this version to resolve this issue.
    """

    # @auto_fp16(apply_to=('img', 'prev_bev', 'points'))
    @force_fp32(apply_to=("img", "points", "prev_bev"))
    def forward_train(
        self,
        points=None,
        img_metas=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        gt_labels=None,
        gt_bboxes=None,
        img=None,
        proposals=None,
        gt_bboxes_ignore=None,
        img_depth=None,
        img_mask=None,
        prev_bev=None,
    ):
        """Forward training function.
        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.
        Returns:
            dict: Losses of different branches.
        """

        img_feats = self.extract_feat(img=img, img_metas=img_metas)
        # import pdb;pdb.set_trace()
        losses = dict()
        losses_pts = self.forward_pts_train(
            img_feats, gt_bboxes_3d, gt_labels_3d, img_metas, gt_bboxes_ignore, prev_bev=prev_bev
        )
        losses.update(losses_pts)
        return losses

    def val_step(self, data, optimizer):
        """
        In BEVFormer_fp16, we use this `val_step` function to inference the `prev_pev`.
        This is not the standard function of `val_step`.
        """

        img = data["img"]
        img_metas = data["img_metas"]
        img_feats = self.extract_feat(img=img, img_metas=img_metas)
        prev_bev = data.get("prev_bev", None)
        prev_bev = self.pts_bbox_head(img_feats, img_metas, prev_bev=prev_bev, only_bev=True)
        return prev_bev
