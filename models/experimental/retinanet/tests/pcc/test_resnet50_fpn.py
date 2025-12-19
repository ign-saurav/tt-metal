# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
import pytest
from loguru import logger
from torchvision.models.detection import retinanet_resnet50_fpn_v2, RetinaNet_ResNet50_FPN_V2_Weights
from PIL import Image
from torchvision import transforms
from ttnn.model_preprocessing import preprocess_model_parameters
from tests.ttnn.utils_for_testing import check_with_pcc
from models.experimental.retinanet.tt.tt_fpn import resnet50Fpn, fpn_optimisations
from ttnn.model_preprocessing import fold_batch_norm2d_into_conv2d
from ttnn.model_preprocessing import infer_ttnn_module_args
from ttnn.dot_access import make_dot_access_dict
from ttnn.torch_tracer import trace, visualize


class ModuleArgs(dict):
    ...


class Conv2dArgs(ModuleArgs):
    __getattr__ = dict.__getitem__
    __delattr__ = dict.__delitem__

    def __repr__(self):
        return super().__repr__()


class ConvTranspose2dArgs(ModuleArgs):
    __getattr__ = dict.__getitem__
    __delattr__ = dict.__delitem__

    def __repr__(self):
        return super().__repr__()


class MaxPool2dArgs(ModuleArgs):
    __getattr__ = dict.__getitem__
    __delattr__ = dict.__delitem__

    def __repr__(self):
        return super().__repr__()


class GroupNormArgs(ModuleArgs):
    __getattr__ = dict.__getitem__
    __delattr__ = dict.__delitem__

    def __repr__(self):
        return super().__repr__()


def infer_ttnn_module_args(*, model, run_model, device):
    if run_model is None:
        return None

    # ------------------------------------------------------------------
    # Run model under TTNN tracing
    # ------------------------------------------------------------------
    with trace():
        output = run_model(model)

    visualize(output, file_name=ttnn.CONFIG.tmp_dir / "model_graph.svg")

    # ------------------------------------------------------------------
    # Helper: insert value into nested dict using module path
    # ------------------------------------------------------------------
    def insert_nested(d, path, value):
        for key in path[:-1]:
            key = int(key) if isinstance(key, str) and key.isdigit() else key
            d = d.setdefault(key, {})
        last = path[-1]
        last = int(last) if isinstance(last, str) and last.isdigit() else last
        d[last] = value

    # ------------------------------------------------------------------
    # Recursive graph walk
    # ------------------------------------------------------------------
    def _infer_ttnn_module_args(graph):
        ttnn_module_args = {}

        for node in graph:
            attributes = graph.nodes[node]
            operation = attributes.get("operation")

            if not isinstance(operation, ttnn.tracer.TorchModule):
                continue

            # Full hierarchical module path
            module_path = operation.module.__ttnn_tracer_name__.split(".")

            # Infer input shape (assumes single input edge)
            in_edges = list(graph.in_edges(node, data=True))
            if not in_edges:
                continue

            input_node, _, edge_data = in_edges[0]
            input_shape = graph.nodes[input_node]["shapes"][edge_data["source_output_index"]]

            module = operation.module

            # ----------------------------------------------------------
            # Conv2d
            # ----------------------------------------------------------
            if isinstance(module, torch.nn.Conv2d):
                insert_nested(
                    ttnn_module_args,
                    module_path,
                    Conv2dArgs(
                        in_channels=module.in_channels,
                        out_channels=module.out_channels,
                        kernel_size=module.kernel_size,
                        stride=module.stride,
                        padding=module.padding,
                        dilation=module.dilation,
                        groups=module.groups,
                        padding_mode=module.padding_mode,
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
                    ),
                )

            # ----------------------------------------------------------
            # ConvTranspose2d
            # ----------------------------------------------------------
            elif isinstance(module, torch.nn.ConvTranspose2d):
                insert_nested(
                    ttnn_module_args,
                    module_path,
                    ConvTranspose2dArgs(
                        in_channels=module.in_channels,
                        out_channels=module.out_channels,
                        kernel_size=module.kernel_size,
                        stride=module.stride,
                        padding=module.padding,
                        output_padding=module.output_padding,
                        dilation=module.dilation,
                        groups=module.groups,
                        padding_mode=module.padding_mode,
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
                    ),
                )

            # ----------------------------------------------------------
            # MaxPool2d
            # ----------------------------------------------------------
            elif isinstance(module, torch.nn.MaxPool2d):
                insert_nested(
                    ttnn_module_args,
                    module_path,
                    MaxPool2dArgs(
                        kernel_size=module.kernel_size,
                        stride=module.stride,
                        padding=module.padding,
                        dilation=module.dilation,
                        batch_size=input_shape[0],
                        input_channels=input_shape[1],
                        input_height=input_shape[-2],
                        input_width=input_shape[-1],
                        dtype=ttnn.bfloat16,
                    ),
                )

            # ----------------------------------------------------------
            # GroupNorm
            # ----------------------------------------------------------
            elif isinstance(module, torch.nn.GroupNorm):
                insert_nested(
                    ttnn_module_args,
                    module_path,
                    GroupNormArgs(
                        num_groups=module.num_groups,
                        num_channels=module.num_channels,
                        eps=module.eps,
                        affine=module.affine,
                        batch_size=input_shape[0],
                        input_height=input_shape[-2],
                        input_width=input_shape[-1],
                        dtype=ttnn.bfloat16,
                    ),
                )

            # ----------------------------------------------------------
            # Container / composite module → recurse
            # ----------------------------------------------------------
            else:
                nested = _infer_ttnn_module_args(operation.graph)
                if nested:
                    insert_nested(
                        ttnn_module_args,
                        module_path,
                        nested,
                    )

        return make_dot_access_dict(ttnn_module_args, ignore_types=(ModuleArgs,))

    # ------------------------------------------------------------------
    # Kick off inference from traced graph
    # ------------------------------------------------------------------
    full_args = _infer_ttnn_module_args(ttnn.tracer.get_graph(output))

    # Root module is stored under empty name ""
    return full_args.get("", full_args)


def conv_bn_to_params(conv, bn, mesh_mapper):
    """Fold BN into Conv (if present) and return TTNN weight/bias tensors."""
    if bn is None:
        weight = conv.weight.detach().clone().contiguous()
        bias = conv.bias.detach().clone().contiguous() if conv.bias is not None else torch.zeros(conv.out_channels)
    else:
        weight, bias = fold_batch_norm2d_into_conv2d(conv, bn)

    return {
        "weight": ttnn.from_torch(weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper),
        "bias": ttnn.from_torch(bias.reshape(1, 1, 1, -1), dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper),
    }


def fpn_to_params(fpn, mesh_mapper):
    """
    Extracts and folds all Conv+BN (if any) layers from a torchvision FeaturePyramidNetwork.
    Produces a nested dictionary structure like:
    {
      inner_blocks: {0: {0: {...}}, 1: {0: {...}}, 2: {0: {...}}},
      layer_blocks: {0: {0: {...}}, ...},
      extra_blocks: {p6: {...}, p7: {...}}
    }
    """

    def extract_from_module(module):
        """Extract Conv2d layers (and optional BN) from a Sequential or ModuleList."""
        params = {}
        for name, child in module.named_children():
            if isinstance(child, torch.nn.Conv2d):
                # Direct Conv2d
                params[name] = {0: conv_bn_to_params(child, None, mesh_mapper)}

            elif isinstance(child, torch.nn.Sequential):
                # Handle Sequential containing Conv2d (like Conv2dNormActivation)
                subparams = {}
                for subname, subchild in child.named_children():
                    if isinstance(subchild, torch.nn.Conv2d):
                        subparams[subname] = conv_bn_to_params(subchild, None, mesh_mapper)
                params[name] = subparams

            elif isinstance(child, torch.nn.Module):
                # Nested structure (rare for FPN)
                subparams = extract_from_module(child)
                if subparams:
                    params[name] = subparams
        return params

    layers = {}

    # Process inner_blocks (1x1 convs)
    if hasattr(fpn, "inner_blocks"):
        layers["inner_blocks"] = extract_from_module(fpn.inner_blocks)

    # Process layer_blocks (3x3 convs)
    if hasattr(fpn, "layer_blocks"):
        layers["layer_blocks"] = extract_from_module(fpn.layer_blocks)

    # Process extra_blocks (e.g., LastLevelP6P7)
    if hasattr(fpn, "extra_blocks"):
        extra_params = {}
        for name, child in fpn.extra_blocks.named_children():
            if isinstance(child, torch.nn.Conv2d):
                extra_params[name] = conv_bn_to_params(child, None, mesh_mapper)
            elif isinstance(child, torch.nn.Sequential):
                for subname, subchild in child.named_children():
                    if isinstance(subchild, torch.nn.Conv2d):
                        extra_params[name] = conv_bn_to_params(subchild, None, mesh_mapper)
        layers["extra_blocks"] = extra_params

    return layers


def create_custom_mesh_preprocessor(mesh_mapper=None):
    """Return a closure that can be passed to TTNN's preprocess pipeline."""

    def custom_fpn_preprocessor(model, name, *, ttnn_module_args=None, convert_to_ttnn=True):
        return fpn_to_params(model, mesh_mapper)

    return custom_fpn_preprocessor


class Resnet50FpnTestInfra:
    def __init__(self, device, batch_size, inplanes, planes, height, width, stride, model_config, name):
        super().__init__()
        self._init_seeds()
        self.device = device
        self.num_devices = device.get_num_devices()
        self.batch_size = batch_size * self.num_devices
        self.inputs_mesh_mapper, self.weights_mesh_mapper, self.output_mesh_composer = self.get_mesh_mappers(device)
        self.name = name

        # Build reference torch model
        weights = RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT
        preprocess = transforms.Compose(
            [
                transforms.CenterCrop((512, 512)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        img = Image.open("models/experimental/retinanet/resources/dog_800x800.jpg").convert("RGB")

        retinanet = retinanet_resnet50_fpn_v2(weights=weights).eval()
        backbone = retinanet.backbone.body
        fpn = retinanet.backbone.fpn
        torch_model = fpn

        # Preprocess parameters for TTNN
        parameters = preprocess_model_parameters(
            initialize_model=lambda: torch_model,
            custom_preprocessor=create_custom_mesh_preprocessor(self.weights_mesh_mapper),
            device=None,
        )

        # Prepare golden inputs/outputs
        backbone_input = preprocess(img).unsqueeze(0)
        self.torch_input_tensor = backbone(backbone_input)  # input dictionary for fpn layer
        self.torch_output_tensor = torch_model(self.torch_input_tensor)

        ################# MODEL ARGS ##################
        conv_args = {}
        conv_args = infer_ttnn_module_args(
            model=torch_model, run_model=lambda model: torch_model(self.torch_input_tensor), device=device
        )
        ################# MODEL ARGS ##################

        # Convert input to TTNN format
        tt_host_tensor = {}
        self.input_tensor = {}
        for key, tensor in self.torch_input_tensor.items():
            tt_host_tensor[key] = ttnn.from_torch(
                tensor.permute(0, 2, 3, 1),
                dtype=ttnn.bfloat16,
                mesh_mapper=self.inputs_mesh_mapper,
            )
            self.input_tensor[key] = ttnn.to_device(tt_host_tensor[key], device, memory_config=ttnn.L1_MEMORY_CONFIG)

        # Build TTNN model
        self.ttnn_model = resnet50Fpn(
            parameters=parameters,
            model_config=model_config,
            model_args=conv_args,
            device=device,
            layer_optimisations=fpn_optimisations,
        )

        # Run + validate
        self.run()
        self.validate(model_config)

    def _init_seeds(self):
        if not hasattr(self, "_model_initialized"):
            torch.manual_seed(42)
            torch.cuda.manual_seed_all(42)
            torch.backends.cudnn.deterministic = True
            self._model_initialized = True

    def get_mesh_mappers(self, device):
        if device.get_num_devices() != 1:
            return (
                ttnn.ShardTensorToMesh(device, dim=0),
                None,
                ttnn.ConcatMeshToTensor(device, dim=0),
            )
        return None, None, None

    def run(self):
        self.output_tensor = self.ttnn_model(self.input_tensor, self.device)
        return self.output_tensor

    def validate(self, model_config, output_tensor=None):
        tt_output = self.output_tensor if output_tensor is None else output_tensor

        valid_pcc = {"0": 0.99, "1": 0.99, "2": 0.99, "p6": 0.99, "p7": 0.99}
        self.pcc_passed_all = []
        self.pcc_message_all = []

        for key in tt_output:
            tt_output_tensor_torch = ttnn.to_torch(
                tt_output[key],
                dtype=self.torch_output_tensor[key].dtype,
                device=self.device,
                mesh_composer=self.output_mesh_composer,
            )

            # Free device memory
            ttnn.deallocate(tt_output[key])

            expected_shape = self.torch_output_tensor[key].shape
            tt_output_tensor_torch = torch.reshape(
                tt_output_tensor_torch, (expected_shape[0], expected_shape[2], expected_shape[3], expected_shape[1])
            )
            tt_output_tensor_torch = torch.permute(tt_output_tensor_torch, (0, 3, 1, 2))

            pcc_passed, pcc_message = check_with_pcc(
                self.torch_output_tensor[key], tt_output_tensor_torch, pcc=valid_pcc[key]
            )
            self.pcc_passed_all.append(pcc_passed)
            self.pcc_message_all.append(pcc_message)

        assert all(self.pcc_passed_all), logger.error(f"PCC check failed: {self.pcc_message_all}")
        logger.info(
            f"\nResNet52 fpn - batch_size={self.batch_size}, "
            f"\nact_dtype={model_config['ACTIVATIONS_DTYPE']}, "
            f"\nweight_dtype={model_config['WEIGHTS_DTYPE']}, "
            f"\nmath_fidelity={model_config['MATH_FIDELITY']}, "
            f"\nPCC={self.pcc_message_all}"
        )

        return self.pcc_passed_all, self.pcc_message_all


# Default model config
model_config = {
    "MATH_FIDELITY": ttnn.MathFidelity.LoFi,
    "WEIGHTS_DTYPE": ttnn.bfloat8_b,
    "ACTIVATIONS_DTYPE": ttnn.bfloat8_b,
}


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, inplanes, planes, height, width, stride, name",
    [
        (1, 3, 128, 512, 512, 1, "backbone.fpn"),
    ],
)
def test_fpn(device, batch_size, inplanes, planes, height, width, stride, name):
    Resnet50FpnTestInfra(
        device,
        batch_size,
        inplanes,
        planes,
        height,
        width,
        stride,
        model_config,
        name,
    )
