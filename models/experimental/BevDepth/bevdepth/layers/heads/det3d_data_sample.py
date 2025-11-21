# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Tuple, Union

import torch

# from mmdet.structures import DetDataSample
from mmengine.structures import InstanceData

# from .point_data import PointData
from models.experimental.BevDepth.bevdepth.layers.heads.point_data import PointData

# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional

from mmengine.structures import BaseDataElement, InstanceData, PixelData


class DetDataSample(BaseDataElement):
    """A data structure interface of MMDetection. They are used as interfaces
    between different components.

    The attributes in ``DetDataSample`` are divided into several parts:

        - ``proposals``(InstanceData): Region proposals used in two-stage
            detectors.
        - ``gt_instances``(InstanceData): Ground truth of instance annotations.
        - ``pred_instances``(InstanceData): Instances of detection predictions.
        - ``pred_track_instances``(InstanceData): Instances of tracking
            predictions.
        - ``ignored_instances``(InstanceData): Instances to be ignored during
            training/testing.
        - ``gt_panoptic_seg``(PixelData): Ground truth of panoptic
            segmentation.
        - ``pred_panoptic_seg``(PixelData): Prediction of panoptic
           segmentation.
        - ``gt_sem_seg``(PixelData): Ground truth of semantic segmentation.
        - ``pred_sem_seg``(PixelData): Prediction of semantic segmentation.

    Examples:
         >>> import torch
         >>> import numpy as np
         >>> from mmengine.structures import InstanceData
         >>> from mmdet.structures import DetDataSample

         >>> data_sample = DetDataSample()
         >>> img_meta = dict(img_shape=(800, 1196),
         ...                 pad_shape=(800, 1216))
         >>> gt_instances = InstanceData(metainfo=img_meta)
         >>> gt_instances.bboxes = torch.rand((5, 4))
         >>> gt_instances.labels = torch.rand((5,))
         >>> data_sample.gt_instances = gt_instances
         >>> assert 'img_shape' in data_sample.gt_instances.metainfo_keys()
         >>> len(data_sample.gt_instances)
         5
         >>> print(data_sample)
        <DetDataSample(

            META INFORMATION

            DATA FIELDS
            gt_instances: <InstanceData(

                    META INFORMATION
                    pad_shape: (800, 1216)
                    img_shape: (800, 1196)

                    DATA FIELDS
                    labels: tensor([0.8533, 0.1550, 0.5433, 0.7294, 0.5098])
                    bboxes:
                    tensor([[9.7725e-01, 5.8417e-01, 1.7269e-01, 6.5694e-01],
                            [1.7894e-01, 5.1780e-01, 7.0590e-01, 4.8589e-01],
                            [7.0392e-01, 6.6770e-01, 1.7520e-01, 1.4267e-01],
                            [2.2411e-01, 5.1962e-01, 9.6953e-01, 6.6994e-01],
                            [4.1338e-01, 2.1165e-01, 2.7239e-04, 6.8477e-01]])
                ) at 0x7f21fb1b9190>
        ) at 0x7f21fb1b9880>
         >>> pred_instances = InstanceData(metainfo=img_meta)
         >>> pred_instances.bboxes = torch.rand((5, 4))
         >>> pred_instances.scores = torch.rand((5,))
         >>> data_sample = DetDataSample(pred_instances=pred_instances)
         >>> assert 'pred_instances' in data_sample

         >>> pred_track_instances = InstanceData(metainfo=img_meta)
         >>> pred_track_instances.bboxes = torch.rand((5, 4))
         >>> pred_track_instances.scores = torch.rand((5,))
         >>> data_sample = DetDataSample(
         ...    pred_track_instances=pred_track_instances)
         >>> assert 'pred_track_instances' in data_sample

         >>> data_sample = DetDataSample()
         >>> gt_instances_data = dict(
         ...                        bboxes=torch.rand(2, 4),
         ...                        labels=torch.rand(2),
         ...                        masks=np.random.rand(2, 2, 2))
         >>> gt_instances = InstanceData(**gt_instances_data)
         >>> data_sample.gt_instances = gt_instances
         >>> assert 'gt_instances' in data_sample
         >>> assert 'masks' in data_sample.gt_instances

         >>> data_sample = DetDataSample()
         >>> gt_panoptic_seg_data = dict(panoptic_seg=torch.rand(2, 4))
         >>> gt_panoptic_seg = PixelData(**gt_panoptic_seg_data)
         >>> data_sample.gt_panoptic_seg = gt_panoptic_seg
         >>> print(data_sample)
        <DetDataSample(

            META INFORMATION

            DATA FIELDS
            _gt_panoptic_seg: <BaseDataElement(

                    META INFORMATION

                    DATA FIELDS
                    panoptic_seg: tensor([[0.7586, 0.1262, 0.2892, 0.9341],
                                [0.3200, 0.7448, 0.1052, 0.5371]])
                ) at 0x7f66c2bb7730>
            gt_panoptic_seg: <BaseDataElement(

                    META INFORMATION

                    DATA FIELDS
                    panoptic_seg: tensor([[0.7586, 0.1262, 0.2892, 0.9341],
                                [0.3200, 0.7448, 0.1052, 0.5371]])
                ) at 0x7f66c2bb7730>
        ) at 0x7f66c2bb7280>
        >>> data_sample = DetDataSample()
        >>> gt_segm_seg_data = dict(segm_seg=torch.rand(2, 2, 2))
        >>> gt_segm_seg = PixelData(**gt_segm_seg_data)
        >>> data_sample.gt_segm_seg = gt_segm_seg
        >>> assert 'gt_segm_seg' in data_sample
        >>> assert 'segm_seg' in data_sample.gt_segm_seg
    """

    @property
    def proposals(self) -> InstanceData:
        return self._proposals

    @proposals.setter
    def proposals(self, value: InstanceData):
        self.set_field(value, "_proposals", dtype=InstanceData)

    @proposals.deleter
    def proposals(self):
        del self._proposals

    @property
    def gt_instances(self) -> InstanceData:
        return self._gt_instances

    @gt_instances.setter
    def gt_instances(self, value: InstanceData):
        self.set_field(value, "_gt_instances", dtype=InstanceData)

    @gt_instances.deleter
    def gt_instances(self):
        del self._gt_instances

    @property
    def pred_instances(self) -> InstanceData:
        return self._pred_instances

    @pred_instances.setter
    def pred_instances(self, value: InstanceData):
        self.set_field(value, "_pred_instances", dtype=InstanceData)

    @pred_instances.deleter
    def pred_instances(self):
        del self._pred_instances

    # directly add ``pred_track_instances`` in ``DetDataSample``
    # so that the ``TrackDataSample`` does not bother to access the
    # instance-level information.
    @property
    def pred_track_instances(self) -> InstanceData:
        return self._pred_track_instances

    @pred_track_instances.setter
    def pred_track_instances(self, value: InstanceData):
        self.set_field(value, "_pred_track_instances", dtype=InstanceData)

    @pred_track_instances.deleter
    def pred_track_instances(self):
        del self._pred_track_instances

    @property
    def ignored_instances(self) -> InstanceData:
        return self._ignored_instances

    @ignored_instances.setter
    def ignored_instances(self, value: InstanceData):
        self.set_field(value, "_ignored_instances", dtype=InstanceData)

    @ignored_instances.deleter
    def ignored_instances(self):
        del self._ignored_instances

    @property
    def gt_panoptic_seg(self) -> PixelData:
        return self._gt_panoptic_seg

    @gt_panoptic_seg.setter
    def gt_panoptic_seg(self, value: PixelData):
        self.set_field(value, "_gt_panoptic_seg", dtype=PixelData)

    @gt_panoptic_seg.deleter
    def gt_panoptic_seg(self):
        del self._gt_panoptic_seg

    @property
    def pred_panoptic_seg(self) -> PixelData:
        return self._pred_panoptic_seg

    @pred_panoptic_seg.setter
    def pred_panoptic_seg(self, value: PixelData):
        self.set_field(value, "_pred_panoptic_seg", dtype=PixelData)

    @pred_panoptic_seg.deleter
    def pred_panoptic_seg(self):
        del self._pred_panoptic_seg

    @property
    def gt_sem_seg(self) -> PixelData:
        return self._gt_sem_seg

    @gt_sem_seg.setter
    def gt_sem_seg(self, value: PixelData):
        self.set_field(value, "_gt_sem_seg", dtype=PixelData)

    @gt_sem_seg.deleter
    def gt_sem_seg(self):
        del self._gt_sem_seg

    @property
    def pred_sem_seg(self) -> PixelData:
        return self._pred_sem_seg

    @pred_sem_seg.setter
    def pred_sem_seg(self, value: PixelData):
        self.set_field(value, "_pred_sem_seg", dtype=PixelData)

    @pred_sem_seg.deleter
    def pred_sem_seg(self):
        del self._pred_sem_seg


SampleList = List[DetDataSample]
OptSampleList = Optional[SampleList]


class Det3DDataSample(DetDataSample):
    """A data structure interface of MMDetection3D. They are used as interfaces
    between different components.

    The attributes in ``Det3DDataSample`` are divided into several parts:

        - ``proposals`` (InstanceData): Region proposals used in two-stage
          detectors.
        - ``ignored_instances`` (InstanceData): Instances to be ignored during
          training/testing.
        - ``gt_instances_3d`` (InstanceData): Ground truth of 3D instance
          annotations.
        - ``gt_instances`` (InstanceData): Ground truth of 2D instance
          annotations.
        - ``pred_instances_3d`` (InstanceData): 3D instances of model
          predictions.
          - For point-cloud 3D object detection task whose input modality is
            `use_lidar=True, use_camera=False`, the 3D predictions results are
            saved in `pred_instances_3d`.
          - For vision-only (monocular/multi-view) 3D object detection task
            whose input modality is `use_lidar=False, use_camera=True`, the 3D
            predictions are saved in `pred_instances_3d`.
        - ``pred_instances`` (InstanceData): 2D instances of model predictions.
          - For multi-modality 3D detection task whose input modality is
            `use_lidar=True, use_camera=True`, the 2D predictions are saved in
            `pred_instances`.
        - ``pts_pred_instances_3d`` (InstanceData): 3D instances of model
          predictions based on point cloud.
          - For multi-modality 3D detection task whose input modality is
            `use_lidar=True, use_camera=True`, the 3D predictions based on
            point cloud are saved in `pts_pred_instances_3d` to distinguish
            with `img_pred_instances_3d` which based on image.
        - ``img_pred_instances_3d`` (InstanceData): 3D instances of model
          predictions based on image.
          - For multi-modality 3D detection task whose input modality is
            `use_lidar=True, use_camera=True`, the 3D predictions based on
            image are saved in `img_pred_instances_3d` to distinguish with
            `pts_pred_instances_3d` which based on point cloud.
        - ``gt_pts_seg`` (PointData): Ground truth of point cloud segmentation.
        - ``pred_pts_seg`` (PointData): Prediction of point cloud segmentation.
        - ``eval_ann_info`` (dict or None): Raw annotation, which will be
          passed to evaluator and do the online evaluation.

    Examples:
        >>> import torch
        >>> from mmengine.structures import InstanceData

        >>> from mmdet3d.structures import Det3DDataSample
        >>> from mmdet3d.structures.bbox_3d import BaseInstance3DBoxes

        >>> data_sample = Det3DDataSample()
        >>> meta_info = dict(
        ...     img_shape=(800, 1196, 3),
        ...     pad_shape=(800, 1216, 3))
        >>> gt_instances_3d = InstanceData(metainfo=meta_info)
        >>> gt_instances_3d.bboxes_3d = BaseInstance3DBoxes(torch.rand((5, 7)))
        >>> gt_instances_3d.labels_3d = torch.randint(0, 3, (5,))
        >>> data_sample.gt_instances_3d = gt_instances_3d
        >>> assert 'img_shape' in data_sample.gt_instances_3d.metainfo_keys()
        >>> len(data_sample.gt_instances_3d)
        5
        >>> print(data_sample)
        <Det3DDataSample(
            META INFORMATION
            DATA FIELDS
            gt_instances_3d: <InstanceData(
                    META INFORMATION
                    img_shape: (800, 1196, 3)
                    pad_shape: (800, 1216, 3)
                    DATA FIELDS
                    labels_3d: tensor([1, 0, 2, 0, 1])
                    bboxes_3d: BaseInstance3DBoxes(
                            tensor([[1.9115e-01, 3.6061e-01, 6.7707e-01, 5.2902e-01, 8.0736e-01, 8.2759e-01,
                                2.4328e-01],
                                [5.6272e-01, 2.7508e-01, 5.7966e-01, 9.2410e-01, 3.0456e-01, 1.8912e-01,
                                3.3176e-01],
                                [8.1069e-01, 2.8684e-01, 7.7689e-01, 9.2397e-02, 5.5849e-01, 3.8007e-01,
                                4.6719e-01],
                                [6.6346e-01, 4.8005e-01, 5.2318e-02, 4.4137e-01, 4.1163e-01, 8.9339e-01,
                                7.2847e-01],
                                [2.4800e-01, 7.1944e-01, 3.4766e-01, 7.8583e-01, 8.5507e-01, 6.3729e-02,
                                7.5161e-05]]))
                ) at 0x7f7e29de3a00>
        ) at 0x7f7e2a0e8640>
        >>> pred_instances = InstanceData(metainfo=meta_info)
        >>> pred_instances.bboxes = torch.rand((5, 4))
        >>> pred_instances.scores = torch.rand((5, ))
        >>> data_sample = Det3DDataSample(pred_instances=pred_instances)
        >>> assert 'pred_instances' in data_sample

        >>> pred_instances_3d = InstanceData(metainfo=meta_info)
        >>> pred_instances_3d.bboxes_3d = BaseInstance3DBoxes(
        ...     torch.rand((5, 7)))
        >>> pred_instances_3d.scores_3d = torch.rand((5, ))
        >>> pred_instances_3d.labels_3d = torch.rand((5, ))
        >>> data_sample = Det3DDataSample(pred_instances_3d=pred_instances_3d)
        >>> assert 'pred_instances_3d' in data_sample

        >>> data_sample = Det3DDataSample()
        >>> gt_instances_3d_data = dict(
        ...     bboxes_3d=BaseInstance3DBoxes(torch.rand((2, 7))),
        ...     labels_3d=torch.rand(2))
        >>> gt_instances_3d = InstanceData(**gt_instances_3d_data)
        >>> data_sample.gt_instances_3d = gt_instances_3d
        >>> assert 'gt_instances_3d' in data_sample
        >>> assert 'bboxes_3d' in data_sample.gt_instances_3d

        >>> from mmdet3d.structures import PointData
        >>> data_sample = Det3DDataSample()
        >>> gt_pts_seg_data = dict(
        ...     pts_instance_mask=torch.rand(2),
        ...     pts_semantic_mask=torch.rand(2))
        >>> data_sample.gt_pts_seg = PointData(**gt_pts_seg_data)
        >>> print(data_sample)
        <Det3DDataSample(
            META INFORMATION
            DATA FIELDS
            gt_pts_seg: <PointData(
                    META INFORMATION
                    DATA FIELDS
                    pts_semantic_mask: tensor([0.7199, 0.4006])
                    pts_instance_mask: tensor([0.7363, 0.8096])
                ) at 0x7f7e2962cc40>
        ) at 0x7f7e29ff0d60>
    """  # noqa: E501

    @property
    def gt_instances_3d(self) -> InstanceData:
        return self._gt_instances_3d

    @gt_instances_3d.setter
    def gt_instances_3d(self, value: InstanceData) -> None:
        self.set_field(value, "_gt_instances_3d", dtype=InstanceData)

    @gt_instances_3d.deleter
    def gt_instances_3d(self) -> None:
        del self._gt_instances_3d

    @property
    def pred_instances_3d(self) -> InstanceData:
        return self._pred_instances_3d

    @pred_instances_3d.setter
    def pred_instances_3d(self, value: InstanceData) -> None:
        self.set_field(value, "_pred_instances_3d", dtype=InstanceData)

    @pred_instances_3d.deleter
    def pred_instances_3d(self) -> None:
        del self._pred_instances_3d

    @property
    def pts_pred_instances_3d(self) -> InstanceData:
        return self._pts_pred_instances_3d

    @pts_pred_instances_3d.setter
    def pts_pred_instances_3d(self, value: InstanceData) -> None:
        self.set_field(value, "_pts_pred_instances_3d", dtype=InstanceData)

    @pts_pred_instances_3d.deleter
    def pts_pred_instances_3d(self) -> None:
        del self._pts_pred_instances_3d

    @property
    def img_pred_instances_3d(self) -> InstanceData:
        return self._img_pred_instances_3d

    @img_pred_instances_3d.setter
    def img_pred_instances_3d(self, value: InstanceData) -> None:
        self.set_field(value, "_img_pred_instances_3d", dtype=InstanceData)

    @img_pred_instances_3d.deleter
    def img_pred_instances_3d(self) -> None:
        del self._img_pred_instances_3d

    @property
    def gt_pts_seg(self) -> PointData:
        return self._gt_pts_seg

    @gt_pts_seg.setter
    def gt_pts_seg(self, value: PointData) -> None:
        self.set_field(value, "_gt_pts_seg", dtype=PointData)

    @gt_pts_seg.deleter
    def gt_pts_seg(self) -> None:
        del self._gt_pts_seg

    @property
    def pred_pts_seg(self) -> PointData:
        return self._pred_pts_seg

    @pred_pts_seg.setter
    def pred_pts_seg(self, value: PointData) -> None:
        self.set_field(value, "_pred_pts_seg", dtype=PointData)

    @pred_pts_seg.deleter
    def pred_pts_seg(self) -> None:
        del self._pred_pts_seg


SampleList = List[Det3DDataSample]
OptSampleList = Optional[SampleList]
ForwardResults = Union[Dict[str, torch.Tensor], List[Det3DDataSample], Tuple[torch.Tensor], torch.Tensor]
