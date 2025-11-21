# Copyright (c) Megvii Inc. All rights reserved.
"""
mAP: 0.3304
mATE: 0.7021
mASE: 0.2795
mAOE: 0.5346
mAVE: 0.5530
mAAE: 0.2274
NDS: 0.4355
Eval time: 171.8s

Per-class results:
Object Class    AP      ATE     ASE     AOE     AVE     AAE
car     0.499   0.540   0.165   0.211   0.650   0.233
truck   0.278   0.719   0.218   0.265   0.547   0.215
bus     0.386   0.661   0.211   0.171   1.132   0.274
trailer 0.168   1.034   0.235   0.548   0.408   0.168
construction_vehicle    0.075   1.124   0.510   1.177   0.111   0.385
pedestrian      0.284   0.757   0.298   0.966   0.578   0.301
motorcycle      0.335   0.624   0.263   0.621   0.734   0.237
bicycle 0.305   0.554   0.264   0.653   0.263   0.006
traffic_cone    0.462   0.516   0.355   nan     nan     nan
barrier 0.512   0.491   0.275   0.200   nan     nan
"""
# from bevdepth.exps.base_cli import run_cli
from models.experimental.BevDepth.bevdepth.exps.nuscenes.base_exp import (
    BEVDepthLightningModel as BaseBEVDepthLightningModel,
)
from models.experimental.BevDepth.bevdepth.models.base_bev_depth import BaseBEVDepth


class BEVDepthLightningModel(BaseBEVDepthLightningModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.key_idxes = [-1]
        self.head_conf["bev_backbone_conf"]["in_channels"] = 80 * (len(self.key_idxes) + 1)
        self.head_conf["bev_neck_conf"]["in_channels"] = [80 * (len(self.key_idxes) + 1), 160, 320, 640]
        self.head_conf["train_cfg"]["code_weights"] = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        self.model = BaseBEVDepth(self.backbone_conf, self.head_conf, is_train_depth=True)


def get_bevdepth_model(**kwargs):
    """
    Instantiates the BEVDepthLightningModel and returns the underlying self.model.

    Args:
        **kwargs: Keyword arguments for the BEVDepthLightningModel.
    Returns:
        BaseBEVDepth: The instantiated model.
    """
    lightning_model = BEVDepthLightningModel(**kwargs)
    return lightning_model.model


if __name__ == "__main__":
    model = get_bevdepth_model()
    print(model)
    # import torch

    # # After model is built
    # model.eval()

    # print("\n" + "="*50)
    # print("Testing forward pass...")
    # print("="*50)

    # # Create dummy inputs matching BEVDepth's expected format
    # batch_size = 1
    # num_sweeps = 1  # For 2-key model, this would be 2
    # num_cameras = 1
    # img_h, img_w = 256, 704

    # # 1. Images
    # imgs = torch.randn(batch_size, num_sweeps, num_cameras, 3, img_h, img_w)

    # # 2. Transformation matrices (mats_dict)
    # mats_dict = {
    #     # Sensor to ego transformation (camera to vehicle coordinates)
    #     'sensor2ego_mats': torch.eye(4).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(
    #         batch_size, num_sweeps, num_cameras, 1, 1
    #     ),

    #     # Intrinsic camera parameters
    #     'intrin_mats': torch.eye(4).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(
    #         batch_size, num_sweeps, num_cameras, 1, 1
    #     ),

    #     # Image data augmentation matrix
    #     'ida_mats': torch.eye(4).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(
    #         batch_size, num_sweeps, num_cameras, 1, 1
    #     ),

    #     # Sensor to sensor transformation (for temporal alignment)
    #     'sensor2sensor_mats': torch.eye(4).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(
    #         batch_size, num_sweeps, num_cameras, 1, 1
    #     ),

    #     # Bird's eye view data augmentation
    #     'bda_mat': torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1),
    # }

    # print(f"Input shapes:")
    # print(f"  imgs: {imgs.shape}")
    # for key, val in mats_dict.items():
    #     print(f"  {key}: {val.shape}")

    # # Run forward pass
    # with torch.no_grad():
    #     try:
    #         output = model(imgs, mats_dict)
    #         print("\n✓ Forward pass successful!")

    #         # Print output structure
    #         if isinstance(output, list):
    #             print(f"\nOutput: List with {len(output)} elements")
    #             for i, task_output in enumerate(output):
    #                 print(f"\n  Task {i}:")
    #                 if isinstance(task_output, list):
    #                     for j, item in enumerate(task_output):
    #                         if isinstance(item, dict):
    #                             print(f"    Item {j} (dict):")
    #                             for key, val in item.items():
    #                                 if isinstance(val, torch.Tensor):
    #                                     print(f"      {key}: {val.shape}")
    #                         elif isinstance(item, torch.Tensor):
    #                             print(f"    Item {j} (tensor): {item.shape}")
    #                 elif isinstance(task_output, dict):
    #                     print(f"    Dict with keys: {task_output.keys()}")
    #                     for key, val in task_output.items():
    #                         if isinstance(val, torch.Tensor):
    #                             print(f"      {key}: {val.shape}")
    #         elif isinstance(output, dict):
    #             print(f"\nOutput: Dict with keys: {output.keys()}")
    #             for key, val in output.items():
    #                 if isinstance(val, torch.Tensor):
    #                     print(f"  {key}: {val.shape}")
    #         else:
    #             print(f"\nOutput type: {type(output)}")

    #     except Exception as e:
    #         print(f"\n✗ Forward pass failed: {e}")
    #         import traceback
    #         traceback.print_exc()
# if __name__ == '__main__':
#     run_cli(BEVDepthLightningModel,
#             'bev_depth_lss_r50_256x704_128x128_24e_2key')
