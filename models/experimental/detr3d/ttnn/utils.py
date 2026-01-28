# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
import numpy as np
from models.common.lightweightmodule import LightweightModule
from models.tt_cnn.tt.builder import TtMaxPool2d, MaxPool2dConfiguration
from models.experimental.detr3d.reference.model_3detr import BoxProcessor
from ttnn.model_preprocessing import Conv2dArgs, ConvTranspose2dArgs, MaxPool2dArgs, GroupNormArgs, ModuleArgs
from ttnn.torch_tracer import trace
from ttnn.dot_access import make_dot_access_dict


def infer_ttnn_module_args(*, model, run_model, device):
    if run_model is None:
        return None

    with trace():
        output = run_model(model)

    def _infer_ttnn_module_args(graph):
        ttnn_module_args = {}
        for node in graph:
            attributes = graph.nodes[node]
            operation = attributes["operation"]
            if isinstance(operation, ttnn.tracer.TorchModule):
                *_, module_name = operation.module.__ttnn_tracer_name__.split(".")
                (input_node, _, edge_data), *_ = graph.in_edges(node, data=True)
                input_shape = graph.nodes[input_node]["shapes"][edge_data["source_output_index"]]
                if isinstance(operation.module, torch.nn.Conv2d):
                    ttnn_module_args[module_name] = Conv2dArgs(
                        in_channels=operation.module.in_channels,
                        out_channels=operation.module.out_channels,
                        kernel_size=operation.module.kernel_size,
                        stride=operation.module.stride,
                        padding=operation.module.padding,
                        dilation=operation.module.dilation,
                        groups=operation.module.groups,
                        padding_mode=operation.module.padding_mode,
                        batch_size=input_shape[0],
                        input_height=input_shape[-2],
                        input_width=input_shape[-1],
                        math_fidelity=ttnn.MathFidelity.HiFi4,
                        dtype=ttnn.bfloat16,
                        weights_dtype=ttnn.bfloat16,
                        use_1d_systolic_array=True,
                        enable_auto_formatting=False,
                        conv_blocking_and_parallelization_config_override={},
                        device=device,
                    )
                elif isinstance(operation.module, torch.nn.ConvTranspose2d):
                    ttnn_module_args[module_name] = ConvTranspose2dArgs(
                        in_channels=operation.module.in_channels,
                        out_channels=operation.module.out_channels,
                        kernel_size=operation.module.kernel_size,
                        stride=operation.module.stride,
                        padding=operation.module.padding,
                        output_padding=operation.module.output_padding,
                        dilation=operation.module.dilation,
                        groups=operation.module.groups,
                        padding_mode=operation.module.padding_mode,
                        batch_size=input_shape[0],
                        input_height=input_shape[-2],
                        input_width=input_shape[-1],
                        math_fidelity=ttnn.MathFidelity.HiFi4,
                        dtype=ttnn.bfloat16,
                        weights_dtype=ttnn.bfloat16,
                        use_1d_systolic_array=True,
                        enable_auto_formatting=False,
                        conv_blocking_and_parallelization_config_override={},
                        device=device,
                    )
                elif isinstance(operation.module, torch.nn.MaxPool2d):
                    ttnn_module_args[module_name] = MaxPool2dArgs(
                        kernel_size=operation.module.kernel_size,
                        stride=operation.module.stride,
                        padding=operation.module.padding,
                        dilation=operation.module.dilation,
                        batch_size=input_shape[0],
                        input_channels=input_shape[1],
                        input_height=input_shape[-2],
                        input_width=input_shape[-1],
                        dtype=ttnn.bfloat16,
                    )
                elif isinstance(operation.module, torch.nn.GroupNorm):
                    ttnn_module_args[module_name] = GroupNormArgs(
                        num_groups=operation.module.num_groups,
                        num_channels=operation.module.num_channels,
                        eps=operation.module.eps,
                        affine=operation.module.affine,
                        batch_size=input_shape[0],
                        input_height=input_shape[-2],
                        input_width=input_shape[-1],
                        dtype=ttnn.bfloat16,
                    )
                else:
                    ttnn_module_args[module_name] = _infer_ttnn_module_args(operation.graph)

                if module_name.isdigit():
                    ttnn_module_args[int(module_name)] = ttnn_module_args[module_name]

        return make_dot_access_dict(ttnn_module_args, ignore_types=(ModuleArgs,))

    ttnn_module_args = _infer_ttnn_module_args(ttnn.tracer.get_graph(output))
    return ttnn_module_args[""]


def box_post_processing(
    cls_logits,
    center_offset,
    size_normalized,
    angle_logits,
    angle_residual_normalized,
    angle_residual,
    num_layers,
    torch_query_xyz,
    torch_point_cloud_dims,
    dataset_config,
):
    torch_cls_logits = ttnn.to_torch(cls_logits)
    torch_center_offset = ttnn.to_torch(center_offset)
    torch_size_normalized = ttnn.to_torch(size_normalized)
    torch_angle_logits = ttnn.to_torch(angle_logits)
    torch_angle_residual_normalized = ttnn.to_torch(angle_residual_normalized)
    torch_angle_residual = ttnn.to_torch(angle_residual)
    if not isinstance(torch_point_cloud_dims[0], torch.Tensor):
        for i in range(len(torch_point_cloud_dims)):
            torch_point_cloud_dims[i] = ttnn.to_torch(torch_point_cloud_dims[i])
    if not isinstance(torch_query_xyz, torch.Tensor):
        torch_query_xyz = ttnn.to_torch(torch_query_xyz)

    torch_box_processor = BoxProcessor(dataset_config)

    torch_outputs = []
    for l in range(num_layers):
        # box processor converts outputs so we can get a 3D bounding box
        (
            torch_center_normalized,
            torch_center_unnormalized,
        ) = torch_box_processor.compute_predicted_center(
            torch_center_offset[l], torch_query_xyz, torch_point_cloud_dims
        )
        torch_angle_continuous = torch_box_processor.compute_predicted_angle(
            torch_angle_logits[l], torch_angle_residual[l]
        )
        torch_size_unnormalized = torch_box_processor.compute_predicted_size(
            torch_size_normalized[l], torch_point_cloud_dims
        )
        torch_box_corners = torch_box_processor.box_parametrization_to_corners(
            torch_center_unnormalized, torch_size_unnormalized, torch_angle_continuous
        )

        # below are used for matching/mAP eval
        (
            torch_semcls_prob,
            torch_objectness_prob,
        ) = torch_box_processor.compute_objectness_and_cls_prob(torch_cls_logits[l])

        torch_box_prediction = {
            "sem_cls_logits": torch_cls_logits[l],
            "center_normalized": torch_center_normalized,
            "center_unnormalized": torch_center_unnormalized,
            "size_normalized": torch_size_normalized[l],
            "size_unnormalized": torch_size_unnormalized,
            "angle_logits": torch_angle_logits[l],
            "angle_residual": torch_angle_residual[l],
            "angle_residual_normalized": torch_angle_residual_normalized[l],
            "angle_continuous": torch_angle_continuous,
            "objectness_prob": torch_objectness_prob,
            "sem_cls_prob": torch_semcls_prob,
            "box_corners": torch_box_corners,
        }
        torch_outputs.append(torch_box_prediction)

    # intermediate decoder layer outputs are only used during training
    # we use them to check for any instability in PCC
    torch_aux_outputs = torch_outputs[:-1]
    torch_outputs = torch_outputs[-1]

    return {
        "outputs": torch_outputs,  # output from last layer of decoder
        "aux_outputs": torch_aux_outputs,  # output from intermediate layers of decoder
    }


class TtnnConv1D(LightweightModule):
    def __init__(
        self,
        conv,
        parameters,
        device,
        activation_dtype=ttnn.bfloat16,
        weights_dtype=ttnn.bfloat16,
        shard_layout=None,
        fp32_accum=False,
        packer_l1_acc=False,
        activation=None,
        deallocate_activation=False,
        math_fidelity=ttnn.MathFidelity.LoFi,
        return_dims=False,
        reshape_output=False,
        memory_config=None,
    ):
        super().__init__()
        self.conv = conv
        self.device = device
        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        self.kernel_size = conv.kernel_size[0]
        self.padding = conv.padding[0]
        self.stride = conv.stride[0]
        self.groups = conv.groups
        self.conv_config = ttnn.Conv1dConfig(
            weights_dtype=weights_dtype,
            shard_layout=shard_layout,
            deallocate_activation=deallocate_activation,
            activation=activation,
        )
        self.compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=math_fidelity,
            fp32_dest_acc_en=fp32_accum,
            packer_l1_acc=packer_l1_acc,
        )
        self.weight = ttnn.from_device(parameters.weight)
        self.bias = None
        if "bias" in parameters and parameters["bias"] is not None:
            bias = ttnn.from_device(parameters.bias)
            self.bias = bias
        self.activation_dtype = activation_dtype
        self.return_dims = return_dims
        self.reshape_output = reshape_output
        self.memory_config = memory_config

    def forward(self, x, shape=None):
        if shape is not None:
            batch_size = shape[0]
            input_length = shape[1]
        else:
            batch_size = x.shape[0]
            input_length = x.shape[1]

        [tt_output_tensor_on_device, out_length, [weights_device, bias_device]] = ttnn.conv1d(
            input_tensor=x,
            weight_tensor=self.weight,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            device=self.device,
            bias_tensor=self.bias,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            batch_size=batch_size,
            input_length=input_length,
            conv_config=self.conv_config,
            compute_config=self.compute_config,
            groups=self.groups,
            return_output_dim=True,
            return_weights_and_bias=True,
            memory_config=self.memory_config,
            dtype=self.activation_dtype,
        )
        shape = (batch_size, out_length, tt_output_tensor_on_device.shape[-1])
        if self.reshape_output:
            tt_output_tensor_on_device = ttnn.reshape(tt_output_tensor_on_device, shape)
        if self.return_dims:
            return tt_output_tensor_on_device, shape
        return tt_output_tensor_on_device


class TtnnMaxPool2DSlice(LightweightModule):
    def __init__(
        self,
        maxpool_args,
        num_maxpool_slice,
        device=None,
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=False,
    ):
        super().__init__()
        self.maxpool_args = maxpool_args
        self.num_maxpool_slice = num_maxpool_slice
        self.slice_h = maxpool_args.input_height // num_maxpool_slice
        self.maxpool = TtMaxPool2d(
            configuration=MaxPool2dConfiguration(
                input_height=maxpool_args.input_height // num_maxpool_slice,
                input_width=maxpool_args.input_width,
                channels=maxpool_args.input_channels,
                batch_size=maxpool_args.batch_size,
                kernel_size=maxpool_args.kernel_size,
                stride=maxpool_args.stride,
                padding=(maxpool_args.padding, maxpool_args.padding),
                dilation=(maxpool_args.dilation, maxpool_args.dilation),
                deallocate_input=True,
                output_layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat16,
            ),
            device=device,
        )

    def forward(self, x):
        x = ttnn.reshape(
            x,
            (
                self.maxpool_args.batch_size,
                self.maxpool_args.input_height,
                self.maxpool_args.input_width,
                self.maxpool_args.input_channels,
            ),
        )
        B, H, W, C = (x.shape[-4], self.slice_h, x.shape[-2], x.shape[-1])

        partial_maxpool_out = []

        for slice in range(self.num_maxpool_slice):
            slice_input = x[:, self.slice_h * slice : self.slice_h * (slice + 1), :, :]
            slice_input = ttnn.reallocate(slice_input)
            slice_input = ttnn.reshape(slice_input, (1, 1, B * H * W, C))
            partial_maxpool_out.append(ttnn.sharded_to_interleaved(self.maxpool(slice_input), ttnn.L1_MEMORY_CONFIG))

            ttnn.deallocate(slice_input)

        for i in range(len(partial_maxpool_out)):
            partial_maxpool_out[i] = ttnn.reshape(partial_maxpool_out[i], (B, H, C))
        new_features = ttnn.concat((partial_maxpool_out), dim=1, memory_config=ttnn.L1_MEMORY_CONFIG)
        new_features = ttnn.permute(
            new_features, (0, 2, 1), memory_config=ttnn.L1_MEMORY_CONFIG
        )  # (B, mlp[-1], npoint)
        for i in range(len(partial_maxpool_out)):
            ttnn.deallocate(partial_maxpool_out[i])

        return new_features


def shift_scale_points_ttnn(pred_xyz, src_range, device=None):
    """
    pred_xyz: B x N x 3
    src_range: [[B x 3], [B x 3]] - min and max XYZ coords
    dst_range: [[B x 3], [B x 3]] - min and max XYZ coords
    """

    dst_range = [
        ttnn.zeros((src_range[0].shape[0], 3), dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT),
        ttnn.ones((src_range[0].shape[0], 3), dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT),
    ]

    assert src_range[0].shape[0] == pred_xyz.shape[0]
    assert dst_range[0].shape[0] == pred_xyz.shape[0]
    assert src_range[0].shape[-1] == pred_xyz.shape[-1]
    assert src_range[0].shape == src_range[1].shape
    assert dst_range[0].shape == dst_range[1].shape
    assert src_range[0].shape == dst_range[1].shape

    src_range[0] = ttnn.unsqueeze(src_range[0], 1)
    src_range[1] = ttnn.unsqueeze(src_range[1], 1)
    dst_range[0] = ttnn.unsqueeze(dst_range[0], 1)
    dst_range[1] = ttnn.unsqueeze(dst_range[1], 1)

    src_diff = src_range[1] - src_range[0]
    dst_diff = dst_range[1] - dst_range[0]
    prop_xyz = pred_xyz - src_range[0]
    prop_xyz = prop_xyz * dst_diff
    prop_xyz = ttnn.div(
        prop_xyz,
        src_diff,
        fast_and_approximate_mode=True,
        # round_mode=None,
    )
    prop_xyz = prop_xyz + dst_range[0]

    ttnn.deallocate(src_diff)
    ttnn.deallocate(dst_diff)
    ttnn.deallocate(dst_range[0])
    ttnn.deallocate(dst_range[1])

    return prop_xyz


def scale_points(pred_xyz, mult_factor):
    if len(pred_xyz.shape) == 4:
        mult_factor = ttnn.unsqueeze(mult_factor, 1)
    mult_factor = ttnn.unsqueeze(mult_factor, 1)
    scaled_xyz = pred_xyz * mult_factor
    return scaled_xyz


class TtnnBoxProcessor(object):
    """
    Class to convert 3DETR MLP head outputs into bounding boxes
    """

    def __init__(self, dataset_config, device):
        self.dataset_config = dataset_config
        self.device = device

    def compute_predicted_center(self, center_offset, query_xyz, point_cloud_dims):
        center_unnormalized = query_xyz + center_offset
        center_normalized = shift_scale_points_ttnn(center_unnormalized, src_range=point_cloud_dims)
        return center_normalized, center_unnormalized

    def compute_predicted_size(self, size_normalized, point_cloud_dims):
        scene_scale = point_cloud_dims[1] - point_cloud_dims[0]
        scene_scale = ttnn.clamp(scene_scale, min=1e-1)
        size_unnormalized = scale_points(size_normalized, mult_factor=scene_scale)
        ttnn.deallocate(scene_scale)
        return size_unnormalized

    def compute_predicted_angle(self, angle_logits, angle_residual):
        if angle_logits.shape[-1] == 1:
            # special case for datasets with no rotation angle
            # we still use the predictions so that model outputs are used
            # in the backwards pass (DDP may complain otherwise)
            angle = angle_logits * 0 + angle_residual * 0
            angle = ttnn.squeeze(angle, -1)
            angle = ttnn.clamp(angle, min=0)
        else:
            angle_per_cls = (2 * np.pi) / self.dataset_config.num_angle_bin
            pred_angle_class = ttnn.argmax(angle_logits, dim=-1)
            angle_center = angle_per_cls * pred_angle_class
            pred_angle_class = ttnn.unsqueeze(pred_angle_class, -1)
            angle_residual_gathered = ttnn.gather(angle_residual, 2, pred_angle_class)
            angle = angle_center + ttnn.squeeze(angle_residual_gathered, -1)
            mask = angle > np.pi
            angle[mask] = angle[mask] - (2 * np.pi)
        return angle

    def compute_objectness_and_cls_prob(self, cls_logits):
        assert cls_logits.shape[-1] == self.dataset_config.num_semcls + 1
        cls_prob = ttnn.softmax(cls_logits, dim=-1)
        objectness_prob = 1 - cls_prob[..., -1]
        return cls_prob[..., :-1], objectness_prob

    def box_parametrization_to_corners(self, box_center_unnorm, box_size_unnorm, box_angle):
        torch_box_center_unnorm = ttnn.to_torch(box_center_unnorm, dtype=torch.float32)
        torch_box_size_unnorm = ttnn.to_torch(box_size_unnorm, dtype=torch.float32)
        torch_box_angle = ttnn.to_torch(box_angle, dtype=torch.float32)
        return self.dataset_config.box_parametrization_to_corners(
            torch_box_center_unnorm, torch_box_size_unnorm, torch_box_angle
        )
