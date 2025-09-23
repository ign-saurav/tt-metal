import torch
from models.experimental.functional_petr.reference.petr import PETR
import torch.onnx


class LiDARInstance3DBoxes:
    def __init__(self, tensor, box_dim=7, with_yaw=True, origin=(0.5, 0.5, 0)):
        self.tensor = tensor
        self.box_dim = box_dim
        self.with_yaw = with_yaw
        self.origin = origin

    def __call__(self, tensor, box_dim=None):
        return LiDARInstance3DBoxes(tensor, box_dim or self.box_dim)

    def to(self, device):
        """Move the underlying tensor to a device"""
        return LiDARInstance3DBoxes(
            self.tensor.to(device) if hasattr(self.tensor, "to") else self.tensor,
            self.box_dim,
            self.with_yaw,
            self.origin,
        )

    def cpu(self):
        """Move to CPU"""
        return self.to("cpu")

    @property
    def shape(self):
        """Get shape of underlying tensor"""
        return self.tensor.shape if hasattr(self.tensor, "shape") else None


def test_reference():
    batch_size = 1
    num_cams = 6  # Typical for nuScenes
    C, H, W = 3, 320, 800  # Typical image dimensions
    # inputs = torch.rand(batch_size, C, H, W)
    inputs = {"imgs": torch.randn(batch_size, num_cams, C, H, W)}
    # modified_batch_img_metas = {
    #     'img_shape': [(H, W, C)] * num_cams,
    #     'lidar2img': torch.randn(num_cams, 4, 4),  # Camera extrinsics
    #     'cam_intrinsic': torch.randn(num_cams, 3, 3),  # Camera intrinsics
    # }
    modified_batch_img_metas = []
    for i in range(batch_size):
        meta = {
            "img_shape": (H, W),
            "pad_shape": (H, W),
            # 'cam_id': (H, W),
            "ori_shape": (H, W, C),
            "batch_input_shape": (H, W),
            "cam2img": [torch.randn(3, 3) for _ in range(num_cams)],
            "lidar2cam": [torch.randn(4, 4) for _ in range(num_cams)],
            "lidar2img": [torch.randn(4, 4) for _ in range(num_cams)],
            "cam2lidar": [torch.randn(4, 4) for _ in range(num_cams)],
            "ego2global": torch.randn(4, 4),
            "img_timestamp": [0.0] * num_cams,
            "img_aug_matrix": [torch.eye(4) for _ in range(num_cams)],
            "box_type_3d": LiDARInstance3DBoxes,
            "scale_factor": 1.0,
            "flip": False,
            "lidar2ego": torch.eye(4),
            "can_bus": torch.randn(18),
        }
        modified_batch_img_metas.append(meta)
    model = PETR(use_grid_mask=True)
    # weights_state_dict = torch.load(
    #     "models/experimental/functional_petr/data.pkl"
    #     # "models/experimental/functional_petr/fcos3d_vovnet_imgbackbone-remapped.pth"
    # )["state_dict"]
    # model.load_state_dict(weights_state_dict)
    model.eval()
    output = model.predict(inputs, modified_batch_img_metas)
    print("output", output)

    class ONNXWrapper(torch.nn.Module):
        def __init__(self, model):
            # super().__init__()
            # self.model = model
            # self.img_backbone = model.img_backbone
            # self.img_neck = model.img_neck if hasattr(model, 'img_neck') else None

            super().__init__()
            self.petr = model
            self.num_cams = 6
            self.H, self.W = 320, 800

            # Register constant shapes as buffers
            self.register_buffer("img_shape", torch.tensor([self.H, self.W]))
            self.register_buffer("pad_shape", torch.tensor([self.H, self.W]))

        def forward(self, imgs):
            # For ONNX, we can only process the image tensor
            B, Nc, C, H, W = 1, 6, 3, 320, 800

            # Flatten batch and cameras
            imgs_flat = imgs.view(-1, C, H, W)

            # Process through image backbone
            # features = self.img_backbone(imgs_flat)

            # # Process through neck if available
            # if self.img_neck is not None and isinstance(features, (tuple, list)):
            #     features = self.img_neck(features)

            # # Return the last feature map if multiple
            # if isinstance(features, (tuple, list)):
            #     return features[-1]

            # outs = self.petr.pts_bbox_head([fpn_feats], img_metas)
            # if isinstance(outs, dict):
            #     return outs["all_cls_scores"], outs["all_bbox_preds"]
            # return features
            feats = self.petr.img_backbone(imgs_flat)
            if not isinstance(feats, (list, tuple)):
                feats = [feats]

            # Neck (FPN)
            fpn_feats = self.petr.img_neck(feats)
            if isinstance(fpn_feats, (list, tuple)):
                fpn_feats = fpn_feats[0]
            _, Cf, Hf, Wf = fpn_feats.shape
            fpn_feats = fpn_feats.view(B, Nc, Cf, Hf, Wf)

            # Prepare minimal metas
            # img_metas = [{
            #     "img_shape": (self.H, self.W),
            #     "pad_shape": (self.H, self.W),
            #     "batch_input_shape": (self.H, self.W),
            # }] * B

            # # Detection head
            # outs = self.petr.pts_bbox_head([fpn_feats], img_metas)
            # if isinstance(outs, dict):
            #     return outs["all_cls_scores"], outs["all_bbox_preds"]
            # return outs
            # masks = torch.ones((B, Nc, H, W), dtype=torch.bool, device=fpn_feats.device)

            # 4) Build img_metas dict without Python unpacking:
            #    Provide pad_shape and img_shape as tensors of shape [2].
            # shape_tensor = torch.tensor([H, W], dtype=torch.int64, device=fpn_feats.device)
            # img_metas = [{
            #     "img_shape": (H, W),
            #     "pad_shape": (H, W),
            #     "batch_input_shape": (H, W),
            #     # Include the masks directly
            #     "pcm_image_mask": masks[i]
            # } for i in range(B)]
            img_metas = []
            for i in range(B):
                img_metas.append(
                    {
                        "img_shape": [(H, W) for _ in range(num_cams)],
                        "pad_shape": (H, W),
                        "ori_shape": (H, W, C),
                        "batch_input_shape": (H, W),
                        "cam2img": [torch.randn(3, 3) for _ in range(num_cams)],
                        "lidar2cam": [torch.randn(4, 4) for _ in range(num_cams)],
                        "lidar2img": [torch.randn(4, 4) for _ in range(num_cams)],
                        "cam2lidar": [torch.randn(4, 4) for _ in range(num_cams)],
                        "ego2global": torch.randn(4, 4),
                        "img_timestamp": [0.0] * num_cams,
                        "img_aug_matrix": [torch.eye(4) for _ in range(num_cams)],
                        "box_type_3d": LiDARInstance3DBoxes,
                        "scale_factor": 1.0,
                        "flip": False,
                        "lidar2ego": torch.eye(4),
                        "can_bus": torch.randn(18),
                    }
                )

            # 5) Head inference
            outs = self.petr.pts_bbox_head([fpn_feats], img_metas)
            if isinstance(outs, dict):
                return outs["all_cls_scores"], outs["all_bbox_preds"]
            return outs

    print("\nExporting wrapped model to ONNX...")
    # wrapper = ONNXWrapper(model)
    # wrapper.eval()

    # dummy_imgs = torch.randn(B, num_cams, C, H, W)
    # torch.onnx.export(
    #     wrapper,
    #     inputs["imgs"],
    #     "petr_model.onnx",
    #     # input_names=['multi_camera_images'],
    #     # output_names=['features'],
    #     # dynamic_axes={'multi_camera_images': {0: 'batch_size'}},
    #     # opset_version=11
    #     input_names=["multi_camera_images"],
    #     output_names=["cls_scores", "bbox_preds"],
    #     dynamic_axes={
    #         "multi_camera_images": {0: "batch_size"},
    #         "cls_scores": {0: "batch_size"},
    #         "bbox_preds": {0: "batch_size"},
    #     },
    #     opset_version=12,
    #     do_constant_folding=True,
    # )
    # print("Model exported to petr_model.onnx")
    wrapper = ONNXWrapper(model).eval()

    dummy_input = torch.randn(1, wrapper.num_cams, 3, wrapper.H, wrapper.W)

    torch.onnx.export(
        wrapper,
        dummy_input,
        "petr_model.onnx",
        input_names=["multi_camera_images"],
        output_names=["cls_scores", "bbox_preds"],
        dynamic_axes={
            "multi_camera_images": {0: "batch_size"},
            "cls_scores": {0: "batch_size"},
            "bbox_preds": {0: "batch_size"},
        },
        opset_version=12,
        do_constant_folding=True,
    )
    print(f"Exported PETR to 'petr_model.onnx'")
