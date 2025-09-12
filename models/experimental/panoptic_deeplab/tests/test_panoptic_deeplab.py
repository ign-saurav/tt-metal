# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger
import ttnn
from ttnn.model_preprocessing import preprocess_model_parameters
from tests.ttnn.utils_for_testing import check_with_pcc

from models.experimental.panoptic_deeplab.reference.panoptic_deeplab import TorchPanopticDeepLab
from models.experimental.panoptic_deeplab.tt.panoptic_deeplab import TTPanopticDeepLab
from models.experimental.panoptic_deeplab.tt.custom_preprocessing import create_custom_mesh_preprocessor
from ttnn.model_preprocessing import infer_ttnn_module_args, preprocess_model_parameters
import os
import pickle
import numpy as np

model_config = {
    "MATH_FIDELITY": ttnn.MathFidelity.LoFi,
    "WEIGHTS_DTYPE": ttnn.bfloat8_b,
    "ACTIVATIONS_DTYPE": ttnn.bfloat8_b,
}


def map_single_key(checkpoint_key):
    """
    Map checkpoint keys to model keys.
    """

    if not checkpoint_key:
        return ""

    key = checkpoint_key

    # BACKBONE MAPPINGS (REVERSE)
    if key.startswith("backbone."):
        # Stem batch norm mappings (do this first to avoid conflicts)
        key = key.replace("backbone.stem.conv1.norm.", "backbone.stem.bn1.")
        key = key.replace("backbone.stem.conv2.norm.", "backbone.stem.bn2.")
        key = key.replace("backbone.stem.conv3.norm.", "backbone.stem.bn3.")

        # Layer mapping: res2/3/4/5 -> layer1/2/3/4
        key = key.replace("backbone.res2.", "backbone.layer1.")
        key = key.replace("backbone.res3.", "backbone.layer2.")
        key = key.replace("backbone.res4.", "backbone.layer3.")
        key = key.replace("backbone.res5.", "backbone.layer4.")

        # Batch norm mapping: conv1/2/3.norm -> bn1/2/3
        key = key.replace(".conv1.norm.", ".bn1.")
        key = key.replace(".conv2.norm.", ".bn2.")
        key = key.replace(".conv3.norm.", ".bn3.")

        # Downsample mapping: shortcut -> downsample
        key = key.replace(".shortcut.norm.", ".downsample.1.")
        # Handle shortcut.weight specifically to avoid matching shortcut.norm
        if ".shortcut." in key and ".shortcut.norm." not in checkpoint_key:
            key = key.replace(".shortcut.", ".downsample.0.")

        return key

    # SEMANTIC HEAD MAPPINGS (REVERSE)
    elif key.startswith("sem_seg_head."):
        # Replace base prefix
        key = key.replace("sem_seg_head.", "semantic_decoder.")

        # Head mappings (do these first to avoid conflicts)
        if ".predictor." in key:
            key = key.replace(".predictor.", ".head_1.conv3.0.")
        elif ".head.pointwise." in key:
            if ".head.pointwise.norm." in key:
                key = key.replace(".head.pointwise.norm.", ".head_1.conv2.1.")
            else:
                key = key.replace(".head.pointwise.", ".head_1.conv2.0.")
        elif ".head.depthwise." in key:
            if ".head.depthwise.norm." in key:
                key = key.replace(".head.depthwise.norm.", ".head_1.conv1.1.")
            else:
                key = key.replace(".head.depthwise.", ".head_1.conv1.0.")

        # ASPP mappings (res5 -> aspp)
        elif ".decoder.res5.project_conv." in key:
            # Special case for ASPP_3_Depthwise
            if ".convs.3.depthwise.norm." in key:
                key = key.replace(".decoder.res5.project_conv.convs.3.depthwise.norm.", ".aspp.ASPP_3_Depthwise.1.")
            elif ".convs.3.depthwise." in key:
                key = key.replace(".decoder.res5.project_conv.convs.3.depthwise.", ".aspp.ASPP_3_Depthwise.0.")

            # ASPP_0_Conv
            elif ".convs.0.norm." in key:
                key = key.replace(".decoder.res5.project_conv.convs.0.norm.", ".aspp.ASPP_0_Conv.1.")
            elif ".convs.0." in key:
                key = key.replace(".decoder.res5.project_conv.convs.0.", ".aspp.ASPP_0_Conv.0.")

            # ASPP_1 Depthwise and Pointwise
            elif ".convs.1.depthwise.norm." in key:
                key = key.replace(".decoder.res5.project_conv.convs.1.depthwise.norm.", ".aspp.ASPP_1_Depthwise.1.")
            elif ".convs.1.depthwise." in key:
                key = key.replace(".decoder.res5.project_conv.convs.1.depthwise.", ".aspp.ASPP_1_Depthwise.0.")
            elif ".convs.1.pointwise.norm." in key:
                key = key.replace(".decoder.res5.project_conv.convs.1.pointwise.norm.", ".aspp.ASPP_1_pointwise.1.")
            elif ".convs.1.pointwise." in key:
                key = key.replace(".decoder.res5.project_conv.convs.1.pointwise.", ".aspp.ASPP_1_pointwise.0.")

            # ASPP_2 Depthwise and Pointwise
            elif ".convs.2.depthwise.norm." in key:
                key = key.replace(".decoder.res5.project_conv.convs.2.depthwise.norm.", ".aspp.ASPP_2_Depthwise.1.")
            elif ".convs.2.depthwise." in key:
                key = key.replace(".decoder.res5.project_conv.convs.2.depthwise.", ".aspp.ASPP_2_Depthwise.0.")
            elif ".convs.2.pointwise.norm." in key:
                key = key.replace(".decoder.res5.project_conv.convs.2.pointwise.norm.", ".aspp.ASPP_2_pointwise.1.")
            elif ".convs.2.pointwise." in key:
                key = key.replace(".decoder.res5.project_conv.convs.2.pointwise.", ".aspp.ASPP_2_pointwise.0.")

            # ASPP_3 Pointwise
            elif ".convs.3.pointwise.norm." in key:
                key = key.replace(".decoder.res5.project_conv.convs.3.pointwise.norm.", ".aspp.ASPP_3_pointwise.1.")
            elif ".convs.3.pointwise." in key:
                key = key.replace(".decoder.res5.project_conv.convs.3.pointwise.", ".aspp.ASPP_3_pointwise.0.")

            # ASPP_4_Conv
            elif ".convs.4." in key:
                key = key.replace(".decoder.res5.project_conv.convs.4.1.", ".aspp.ASPP_4_Conv_1.0.")

            # ASPP project
            elif ".project.norm." in key:
                key = key.replace(".decoder.res5.project_conv.project.norm.", ".aspp.ASPP_project.1.")
            elif ".project." in key:
                key = key.replace(".decoder.res5.project_conv.project.", ".aspp.ASPP_project.0.")

        # Decoder res3 mappings
        elif ".decoder.res3." in key:
            if ".project_conv.norm." in key:
                key = key.replace(".decoder.res3.project_conv.norm.", ".res3.conv1.1.")
            elif ".project_conv." in key:
                key = key.replace(".decoder.res3.project_conv.", ".res3.conv1.0.")
            elif ".fuse_conv.depthwise.norm." in key:
                key = key.replace(".decoder.res3.fuse_conv.depthwise.norm.", ".res3.conv2.1.")
            elif ".fuse_conv.depthwise." in key:
                key = key.replace(".decoder.res3.fuse_conv.depthwise.", ".res3.conv2.0.")
            elif ".fuse_conv.pointwise.norm." in key:
                key = key.replace(".decoder.res3.fuse_conv.pointwise.norm.", ".res3.conv3.1.")
            elif ".fuse_conv.pointwise." in key:
                key = key.replace(".decoder.res3.fuse_conv.pointwise.", ".res3.conv3.0.")

        # Decoder res2 mappings
        elif ".decoder.res2." in key:
            if ".project_conv.norm." in key:
                key = key.replace(".decoder.res2.project_conv.norm.", ".res2.conv1.1.")
            elif ".project_conv." in key:
                key = key.replace(".decoder.res2.project_conv.", ".res2.conv1.0.")
            elif ".fuse_conv.depthwise.norm." in key:
                key = key.replace(".decoder.res2.fuse_conv.depthwise.norm.", ".res2.conv2.1.")
            elif ".fuse_conv.depthwise." in key:
                key = key.replace(".decoder.res2.fuse_conv.depthwise.", ".res2.conv2.0.")
            elif ".fuse_conv.pointwise.norm." in key:
                key = key.replace(".decoder.res2.fuse_conv.pointwise.norm.", ".res2.conv3.1.")
            elif ".fuse_conv.pointwise." in key:
                key = key.replace(".decoder.res2.fuse_conv.pointwise.", ".res2.conv3.0.")

        return key

    # INSTANCE HEAD MAPPINGS
    elif key.startswith("ins_embed_head."):
        # Replace base prefix
        key = key.replace("ins_embed_head.", "instance_decoder.")

        # Center head mappings (do these first)
        if ".center_head.0.norm." in key:
            key = key.replace(".center_head.0.norm.", ".head_2.conv1.1.")
        elif ".center_head.0." in key:
            key = key.replace(".center_head.0.", ".head_2.conv1.0.")
        elif ".center_head.1.norm." in key:
            key = key.replace(".center_head.1.norm.", ".head_2.conv2.1.")
        elif ".center_head.1." in key:
            key = key.replace(".center_head.1.", ".head_2.conv2.0.")
        elif ".center_predictor." in key:
            key = key.replace(".center_predictor.", ".head_2.conv3.0.")

        # Offset head mappings
        elif ".offset_head.depthwise.norm." in key:
            key = key.replace(".offset_head.depthwise.norm.", ".head_1.conv1.1.")
        elif ".offset_head.depthwise." in key:
            key = key.replace(".offset_head.depthwise.", ".head_1.conv1.0.")
        elif ".offset_head.pointwise.norm." in key:
            key = key.replace(".offset_head.pointwise.norm.", ".head_1.conv2.1.")
        elif ".offset_head.pointwise." in key:
            key = key.replace(".offset_head.pointwise.", ".head_1.conv2.0.")
        elif ".offset_predictor." in key:
            key = key.replace(".offset_predictor.", ".head_1.conv3.0.")

        # ASPP mappings (res5 -> aspp)
        elif ".decoder.res5.project_conv." in key:
            # Special case for ASPP_3_Depthwise
            if ".convs.3.depthwise.norm." in key:
                key = key.replace(".decoder.res5.project_conv.convs.3.depthwise.norm.", ".aspp.ASPP_3_Depthwise.1.")
            elif ".convs.3.depthwise." in key:
                key = key.replace(".decoder.res5.project_conv.convs.3.depthwise.", ".aspp.ASPP_3_Depthwise.0.")

            # ASPP_0_Conv
            elif ".convs.0.norm." in key:
                key = key.replace(".decoder.res5.project_conv.convs.0.norm.", ".aspp.ASPP_0_Conv.1.")
            elif ".convs.0." in key:
                key = key.replace(".decoder.res5.project_conv.convs.0.", ".aspp.ASPP_0_Conv.0.")

            # ASPP_1 Depthwise and Pointwise
            elif ".convs.1.depthwise.norm." in key:
                key = key.replace(".decoder.res5.project_conv.convs.1.depthwise.norm.", ".aspp.ASPP_1_Depthwise.1.")
            elif ".convs.1.depthwise." in key:
                key = key.replace(".decoder.res5.project_conv.convs.1.depthwise.", ".aspp.ASPP_1_Depthwise.0.")
            elif ".convs.1.pointwise.norm." in key:
                key = key.replace(".decoder.res5.project_conv.convs.1.pointwise.norm.", ".aspp.ASPP_1_pointwise.1.")
            elif ".convs.1.pointwise." in key:
                key = key.replace(".decoder.res5.project_conv.convs.1.pointwise.", ".aspp.ASPP_1_pointwise.0.")

            # ASPP_2 Depthwise and Pointwise
            elif ".convs.2.depthwise.norm." in key:
                key = key.replace(".decoder.res5.project_conv.convs.2.depthwise.norm.", ".aspp.ASPP_2_Depthwise.1.")
            elif ".convs.2.depthwise." in key:
                key = key.replace(".decoder.res5.project_conv.convs.2.depthwise.", ".aspp.ASPP_2_Depthwise.0.")
            elif ".convs.2.pointwise.norm." in key:
                key = key.replace(".decoder.res5.project_conv.convs.2.pointwise.norm.", ".aspp.ASPP_2_pointwise.1.")
            elif ".convs.2.pointwise." in key:
                key = key.replace(".decoder.res5.project_conv.convs.2.pointwise.", ".aspp.ASPP_2_pointwise.0.")

            # ASPP_3 Pointwise
            elif ".convs.3.pointwise.norm." in key:
                key = key.replace(".decoder.res5.project_conv.convs.3.pointwise.norm.", ".aspp.ASPP_3_pointwise.1.")
            elif ".convs.3.pointwise." in key:
                key = key.replace(".decoder.res5.project_conv.convs.3.pointwise.", ".aspp.ASPP_3_pointwise.0.")

            # ASPP_4_Conv
            elif ".convs.4." in key:
                key = key.replace(".decoder.res5.project_conv.convs.4.1.", ".aspp.ASPP_4_Conv_1.0.")

            # ASPP project
            elif ".project.norm." in key:
                key = key.replace(".decoder.res5.project_conv.project.norm.", ".aspp.ASPP_project.1.")
            elif ".project." in key:
                key = key.replace(".decoder.res5.project_conv.project.", ".aspp.ASPP_project.0.")

        # Decoder res3 mappings
        elif ".decoder.res3." in key:
            if ".project_conv.norm." in key:
                key = key.replace(".decoder.res3.project_conv.norm.", ".res3.conv1.1.")
            elif ".project_conv." in key:
                key = key.replace(".decoder.res3.project_conv.", ".res3.conv1.0.")
            elif ".fuse_conv.depthwise.norm." in key:
                key = key.replace(".decoder.res3.fuse_conv.depthwise.norm.", ".res3.conv2.1.")
            elif ".fuse_conv.depthwise." in key:
                key = key.replace(".decoder.res3.fuse_conv.depthwise.", ".res3.conv2.0.")
            elif ".fuse_conv.pointwise.norm." in key:
                key = key.replace(".decoder.res3.fuse_conv.pointwise.norm.", ".res3.conv3.1.")
            elif ".fuse_conv.pointwise." in key:
                key = key.replace(".decoder.res3.fuse_conv.pointwise.", ".res3.conv3.0.")

        # Decoder res2 mappings
        elif ".decoder.res2." in key:
            if ".project_conv.norm." in key:
                key = key.replace(".decoder.res2.project_conv.norm.", ".res2.conv1.1.")
            elif ".project_conv." in key:
                key = key.replace(".decoder.res2.project_conv.", ".res2.conv1.0.")
            elif ".fuse_conv.depthwise.norm." in key:
                key = key.replace(".decoder.res2.fuse_conv.depthwise.norm.", ".res2.conv2.1.")
            elif ".fuse_conv.depthwise." in key:
                key = key.replace(".decoder.res2.fuse_conv.depthwise.", ".res2.conv2.0.")
            elif ".fuse_conv.pointwise.norm." in key:
                key = key.replace(".decoder.res2.fuse_conv.pointwise.norm.", ".res2.conv3.1.")
            elif ".fuse_conv.pointwise." in key:
                key = key.replace(".decoder.res2.fuse_conv.pointwise.", ".res2.conv3.0.")

        return key

    return ""


class PanopticDeepLabTestInfra:
    def __init__(
        self,
        device,
        batch_size,
        in_channels,
        height,
        width,
        model_config,
        weights_path,
        real_input_path=None,
    ):
        super().__init__()
        if not hasattr(self, "_model_initialized"):
            torch.manual_seed(42)
            self._model_initialized = True
            torch.cuda.manual_seed_all(42)
            torch.backends.cudnn.deterministic = True

        self.pcc_passed = False
        self.pcc_message = "call validate()?"
        self.device = device
        self.num_devices = device.get_num_devices()
        self.batch_size = batch_size
        self.in_channels = in_channels
        self.height = height
        self.width = width
        self.inputs_mesh_mapper, self.weights_mesh_mapper, self.output_mesh_composer = self.get_mesh_mappers(device)
        self.weights_path = os.path.join(os.path.dirname(__file__), "weights", f"{weights_path}")
        self.real_input_path = real_input_path
        # Initialize torch model
        torch_model = TorchPanopticDeepLab()
        torch_model = load_torch_model_state(torch_model, "panoptic_deeplab")

        #########################################################
        # Load weights if provided
        print("self.weights_path", self.weights_path)
        if self.weights_path:
            # if self.weights_path and os.path.exists(self.weights_path):
            logger.info(f"Loading PyTorch weights from: {self.weights_path}")

            # Test manual mapping first
            logger.debug(f"Testing manual mapping with: {self.weights_path}")

            try:
                # Load checkpoint
                with open(self.weights_path, "rb") as f:
                    checkpoint = pickle.load(f, encoding="latin1")

                # Get state dict
                if "model_state_dict" in checkpoint:
                    state_dict = checkpoint["model_state_dict"]
                    logger.info("Using 'model_state_dict' key")
                elif "model" in checkpoint:
                    state_dict = checkpoint["model"]
                    logger.info("Using 'model' key")
                else:
                    state_dict = checkpoint
                    logger.info("Using checkpoint directly as state dict")

                # Convert numpy arrays to torch tensors
                converted_count = 0
                for k, v in state_dict.items():
                    if isinstance(v, np.ndarray):
                        state_dict[k] = torch.from_numpy(v)
                        converted_count += 1
                logger.debug(f"Converted {converted_count} numpy arrays to torch tensors")

                # Get model keys
                # logger.debug(f"Model keys: {torch_model.state_dict().keys()}")
                model_dict = torch_model.state_dict()
                # logger.debug(f"Model dict keys: {model_dict.keys()}")
                model_keys = set(model_dict.keys())
                checkpoint_keys = set(state_dict.keys())
                ######################################################
                for mk in model_keys:
                    print(mk)
                print("-------------------------------")
                print("-------------------------------")
                print("-------------------------------")
                print("-------------------------------")
                print("-------------------------------")
                print("-------------------------------")
                for ck in checkpoint_keys:
                    print(ck)
                #############################
                # my_state_dict = torch_model.res2.state_dict()
                # for key, value in my_state_dict.items():
                #     print(key)
                # print("--------------------------------")
                ##############################
                # Create comprehensive mapping
                logger.info("Creating comprehensive key mapping...")

                ####################################

                logger.info("Mapping keys...")
                key_mapping = {}
                for checkpoint_key in checkpoint_keys:  # pickle key
                    # logger.debug(f"IGN Model key: {model_keys} (checkpoint key: {checkpoint_key})")
                    mapped_key = map_single_key(checkpoint_key)
                    if mapped_key in model_keys:  # torch keys
                        key_mapping[checkpoint_key] = mapped_key
                    else:
                        logger.debug(f"No mapping for mapped key: {mapped_key} (checkpoint key: {checkpoint_key})")

                # logger.info(f"Key mapping {key_mapping}")
                print("-------------------------------")
                print("-------------------------------")
                print("-------------------------------")
                print("-------------------------------")
                print("-------------------------------")
                for k, v in key_mapping.items():
                    print(k)
                print("--------------------------------")
                logger.info(f"Model_keys - {len(model_keys)} , checkpoint_keys - {len(checkpoint_keys)}")

                # Apply mappings
                mapped_state_dict = {}
                for checkpoint_key, model_key in key_mapping.items():
                    mapped_state_dict[model_key] = state_dict[checkpoint_key]

                # Try loading
                try:
                    torch_model.load_state_dict(mapped_state_dict, strict=True)
                    logger.info(f"Successfully loaded all {len(mapped_state_dict)} mapped weights with strict=True")

                except RuntimeError as e:
                    logger.warning(f"Strict loading failed:")
                    logger.info("Attempting partial loading...")

                    # Partial loading
                    loaded_keys = []
                    skipped_keys = []

                    for model_key, checkpoint_tensor in mapped_state_dict.items():
                        if model_key in model_dict:
                            if model_dict[model_key].shape == checkpoint_tensor.shape:
                                model_dict[model_key] = checkpoint_tensor
                                loaded_keys.append(model_key)
                            else:
                                skipped_keys.append(f"{model_key}: shape mismatch")
                        else:
                            skipped_keys.append(f"{model_key}: not found in model")

                    # Load the updated model dict
                    torch_model.load_state_dict(model_dict)

                    total_model_params = len(model_dict)
                    load_ratio = len(loaded_keys) / total_model_params

                    logger.info(f"Loaded {len(loaded_keys)}/{total_model_params} model parameters ({load_ratio:.1%})")

                    if skipped_keys:
                        logger.warning(f"Skipped {len(skipped_keys)} incompatible weights (showing first 10):")
                        for skip_msg in skipped_keys[:10]:
                            logger.warning(f"  - {skip_msg}")

                    if load_ratio >= 0.7:
                        logger.info(f"Successfully loaded {load_ratio:.1%} of model weights")
                    elif load_ratio >= 0.5:
                        logger.warning(f"Loaded {load_ratio:.1%} of model weights")
                    else:
                        logger.error(f"Only loaded {load_ratio:.1%} of model weights")

                # Print sample weight values for a few loaded checkpoint keys and their mapped model keys
                logger.info("Sample loaded weights (checkpoint key -> model key):")
                for i, (checkpoint_key, model_key) in enumerate(key_mapping.items()):
                    if i >= 3:
                        break
                    checkpoint_key = "backbone.res4.5.conv3.weight"
                    model_key = "backbone.layer3.5.conv3.weight"
                    ckpt_val = state_dict[checkpoint_key]
                    model_val = torch_model.state_dict()[model_key]
                    logger.info(f"  {checkpoint_key} -> {model_key}")
                    logger.info(
                        f"    checkpoint value (mean/std): {ckpt_val.float().mean():f} / {ckpt_val.float().std():f}"
                    )
                    logger.info(
                        f"    model value (mean/std):      {model_val.float().mean():f} / {model_val.float().std():f}"
                    )
                conv_layer = torch_model.backbone.layer3[5].conv3
                bn_layer = torch_model.backbone.layer3[5].bn3

                print(f"BN running_mean: {bn_layer.running_mean[:5]}")  # First 5 values
                print(f"BN running_var: {bn_layer.running_var[:5]}")
                print(f"BN weight (gamma): {bn_layer.weight[:5]}")
                print(f"BN bias (beta): {bn_layer.bias[:5]}")

                # Verify sample parameters were updated
                sample_params = list(torch_model.parameters())[:3]
                if all(torch.any(p != 0) for p in sample_params):
                    logger.info("Weight verification passed - parameters contain non-zero values")
                else:
                    logger.warning("Weight verification failed - found zero parameters")

            except Exception as e:
                logger.error(f"Failed to load weights file: {str(e)}")
                logger.warning("Falling back to random initialization")

        else:
            logger.warning("No weights provided - using random initialization")

        #########################################################

        #############################

        #########################################################

        # Create or load input tensor
        if self.real_input_path and os.path.exists(self.real_input_path):
            logger.info(f"Loading real input from: {self.real_input_path}")
            self.torch_input_tensor = self.load_real_input(self.real_input_path)

            # Verify shape matches expected dimensions
            expected_shape = (batch_size * self.num_devices, in_channels, height, width)
            if self.torch_input_tensor.shape != expected_shape:
                logger.warning(
                    f"Input shape mismatch. Expected: {expected_shape}, Got: {self.torch_input_tensor.shape}"
                )
                # Optionally resize or adjust
                # if self.torch_input_tensor.shape[0] < batch_size * self.num_devices:
                #     # Repeat batch to match expected batch size
                #     repeats = (batch_size * self.num_devices) // self.torch_input_tensor.shape[0]
                #     self.torch_input_tensor = self.torch_input_tensor.repeat(repeats, 1, 1, 1)
                #     logger.info(f"Repeated input to match batch size: {self.torch_input_tensor.shape}")
        else:
            logger.info("Using random input tensor (no real input provided)")
            input_shape = (batch_size * self.num_devices, in_channels, height, width)
            self.torch_input_tensor = torch.rand(input_shape, dtype=torch.float32)

        # Log input statistics
        logger.info(f"Input tensor stats:")
        logger.info(f"  Shape: {self.torch_input_tensor.shape}")
        logger.info(f"  Mean: {self.torch_input_tensor.mean():.4f}")
        logger.info(f"  Std: {self.torch_input_tensor.std():.4f}")
        logger.info(f"  Min: {self.torch_input_tensor.min():.4f}")
        logger.info(f"  Max: {self.torch_input_tensor.max():.4f}")
        #############################
        # # Create input tensor
        # input_shape = (batch_size * self.num_devices, in_channels, height, width)
        # self.torch_input_tensor = torch.rand(input_shape, dtype=torch.float32)

        #############################
        # import onnx
        # onnx.export(torch_model, self.torch_input_tensor, "/home/ubuntu/ign-tt-sm/tt-metal/models/experimental/panoptic_deeplab/panoptic_deeplab_torch_model.onnx")
        #############################

        # Preprocess model parameters
        parameters = preprocess_model_parameters(
            initialize_model=lambda: torch_model,
            custom_preprocessor=create_custom_mesh_preprocessor(self.weights_mesh_mapper),
            device=None,
        )
        #########################################################
        # print("parameters", parameters)
        #########################################################

        parameters.conv_args = {}
        input_tensor = torch.randn(1, 2048, 32, 64)
        res3_tensor = torch.randn(1, 512, 64, 128)
        res2_tensor = torch.randn(1, 256, 128, 256)

        # For semantic decoder
        if hasattr(parameters, "semantic_decoder"):
            # ASPP
            aspp_args = infer_ttnn_module_args(
                model=torch_model.semantic_decoder.aspp, run_model=lambda model: model(input_tensor), device=None
            )
            if hasattr(parameters.semantic_decoder, "aspp"):
                parameters.semantic_decoder.aspp.conv_args = aspp_args

            # Res3
            aspp_out = torch_model.semantic_decoder.aspp(input_tensor)
            res3_args = infer_ttnn_module_args(
                model=torch_model.semantic_decoder.res3,
                run_model=lambda model: model(aspp_out, res3_tensor),
                device=None,
            )
            if hasattr(parameters.semantic_decoder, "res3"):
                parameters.semantic_decoder.res3.conv_args = res3_args

            # Res2
            res3_out = torch_model.semantic_decoder.res3(aspp_out, res3_tensor)
            res2_args = infer_ttnn_module_args(
                model=torch_model.semantic_decoder.res2,
                run_model=lambda model: model(res3_out, res2_tensor),
                device=None,
            )
            if hasattr(parameters.semantic_decoder, "res2"):
                parameters.semantic_decoder.res2.conv_args = res2_args

            # Head
            res2_out = torch_model.semantic_decoder.res2(res3_out, res2_tensor)
            head_args = infer_ttnn_module_args(
                model=torch_model.semantic_decoder.head_1, run_model=lambda model: model(res2_out), device=None
            )
            if hasattr(parameters.semantic_decoder, "head_1"):
                parameters.semantic_decoder.head_1.conv_args = head_args

        # For instance decoder
        if hasattr(parameters, "instance_decoder"):
            # ASPP
            aspp_args = infer_ttnn_module_args(
                model=torch_model.instance_decoder.aspp, run_model=lambda model: model(input_tensor), device=None
            )
            if hasattr(parameters.instance_decoder, "aspp"):
                parameters.instance_decoder.aspp.conv_args = aspp_args

            # Res3
            aspp_out = torch_model.instance_decoder.aspp(input_tensor)
            res3_args = infer_ttnn_module_args(
                model=torch_model.instance_decoder.res3,
                run_model=lambda model: model(aspp_out, res3_tensor),
                device=None,
            )
            if hasattr(parameters.instance_decoder, "res3"):
                parameters.instance_decoder.res3.conv_args = res3_args

            # Res2
            res3_out = torch_model.instance_decoder.res3(aspp_out, res3_tensor)
            res2_args = infer_ttnn_module_args(
                model=torch_model.instance_decoder.res2,
                run_model=lambda model: model(res3_out, res2_tensor),
                device=None,
            )
            if hasattr(parameters.instance_decoder, "res2"):
                parameters.instance_decoder.res2.conv_args = res2_args

            # Head
            res2_out = torch_model.instance_decoder.res2(res3_out, res2_tensor)
            head_args_1 = infer_ttnn_module_args(
                model=torch_model.instance_decoder.head_1, run_model=lambda model: model(res2_out), device=None
            )
            head_args_2 = infer_ttnn_module_args(
                model=torch_model.instance_decoder.head_2, run_model=lambda model: model(res2_out), device=None
            )
            if hasattr(parameters.instance_decoder, "head_1"):
                parameters.instance_decoder.head_1.conv_args = head_args_1
            if hasattr(parameters.instance_decoder, "head_2"):
                parameters.instance_decoder.head_2.conv_args = head_args_2

        # Run torch model with bfloat16
        logger.info("Running PyTorch model...")
        self.torch_output_tensor, self.torch_output_tensor_2, self.torch_output_tensor_3 = torch_model(
            self.torch_input_tensor
        )

        # Convert input to TTNN format (NHWC)
        logger.info("Converting input to TTNN format...")
        tt_host_tensor = ttnn.from_torch(
            self.torch_input_tensor.permute(0, 2, 3, 1),
            dtype=ttnn.bfloat16,
            mesh_mapper=self.inputs_mesh_mapper,
        )

        ##################################################################
        if hasattr(parameters, "backbone"):
            if hasattr(parameters.backbone, "layer3"):
                weight = parameters.backbone.layer3[5].conv3.weight
                weight_torch = ttnn.to_torch(weight)
                print(f"backbone.layer3.5.conv3.weight stats:")
                print(f"  Shape: {weight_torch.shape}")
                print(f"  Min: {weight_torch.min().item():.4f}")
                print(f"  Max: {weight_torch.max().item():.4f}")
                print(f"  Mean: {weight_torch.mean().item():.4f}")
                print(f"  Std: {weight_torch.std().item():.4f}")

        torch_conv = torch_model.backbone.layer3[5].conv3
        torch_bn = torch_model.backbone.layer3[5].bn3
        from ttnn.model_preprocessing import fold_batch_norm2d_into_conv2d

        # Fold BN for fair comparison
        folded_weight, _ = fold_batch_norm2d_into_conv2d(torch_conv, torch_bn)

        # Get TTNN weight
        ttnn_weight = ttnn.to_torch(parameters.backbone.layer3[5].conv3.weight)

        print(f"Folded PyTorch: mean={folded_weight.mean():.4f}, std={folded_weight.std():.4f}")
        print(f"TTNN processed: mean={ttnn_weight.mean():.4f}, std={ttnn_weight.std():.4f}")
        #############################################################################
        # Initialize TTNN model
        logger.info("Initializing TTNN model...")
        print("Initializing TTNN model...")
        self.ttnn_model = TTPanopticDeepLab(
            parameters=parameters,
            model_config=model_config,
        )

        logger.info("Running first TTNN model pass (JIT configuration)...")
        # first run configures convs JIT
        self.input_tensor = ttnn.to_device(tt_host_tensor, device)
        self.run()
        self.validate()

        logger.info("Running optimized TTNN model pass...")
        # Optimized run
        self.input_tensor = ttnn.to_device(tt_host_tensor, device)
        self.run()
        self.validate()

    def get_mesh_mappers(self, device):
        if device.get_num_devices() != 1:
            inputs_mesh_mapper = ttnn.ShardTensorToMesh(device, dim=0)
            weights_mesh_mapper = None
            output_mesh_composer = ttnn.ConcatMeshToTensor(device, dim=0)
        else:
            inputs_mesh_mapper = None
            weights_mesh_mapper = None
            output_mesh_composer = None
        return inputs_mesh_mapper, weights_mesh_mapper, output_mesh_composer

    def run(self):
        self.output_tensor, self.output_tensor_2, self.output_tensor_3 = self.ttnn_model(self.input_tensor, self.device)
        return self.output_tensor, self.output_tensor_2, self.output_tensor_3

    def validate(self, output_tensor=None):
        output_tensor = self.output_tensor if output_tensor is None else output_tensor
        output_tensor = ttnn.to_torch(output_tensor, device=self.device, mesh_composer=self.output_mesh_composer)
        expected_shape = self.torch_output_tensor.shape
        output_tensor = torch.reshape(
            output_tensor, (expected_shape[0], expected_shape[2], expected_shape[3], expected_shape[1])
        )
        output_tensor = torch.permute(output_tensor, (0, 3, 1, 2))

        batch_size = output_tensor.shape[0]

        valid_pcc = 0.97
        self.pcc_passed, self.pcc_message = check_with_pcc(self.torch_output_tensor, output_tensor, pcc=valid_pcc)
        # assert self.pcc_passed, logger.error(f"Semantic Segmentation Head PCC check failed: {self.pcc_message}")
        logger.info(
            f"Panoptic DeepLab - Semantic Segmentation Head: batch_size={self.batch_size}, "
            f"act_dtype={model_config['ACTIVATIONS_DTYPE']}, weight_dtype={model_config['WEIGHTS_DTYPE']}, "
            f"math_fidelity={model_config['MATH_FIDELITY']}, PCC={self.pcc_message}, shape={self.output_tensor.shape}"
        )

        # Validate instance segmentation head outputs
        output_tensor = self.output_tensor_2
        output_tensor = ttnn.to_torch(output_tensor, device=self.device, mesh_composer=self.output_mesh_composer)
        expected_shape = self.torch_output_tensor_2.shape
        output_tensor = torch.reshape(
            output_tensor, (expected_shape[0], expected_shape[2], expected_shape[3], expected_shape[1])
        )
        output_tensor = torch.permute(output_tensor, (0, 3, 1, 2))

        batch_size = output_tensor.shape[0]

        valid_pcc = 0.97
        self.pcc_passed, self.pcc_message = check_with_pcc(self.torch_output_tensor_2, output_tensor, pcc=valid_pcc)
        # assert self.pcc_passed, logger.error(f"Instance Segmentation Head PCC check failed: {self.pcc_message}")
        logger.info(
            f"Panoptic DeepLab - Instance Segmentation Offset Head: batch_size={self.batch_size}, "
            f"act_dtype={model_config['ACTIVATIONS_DTYPE']}, weight_dtype={model_config['WEIGHTS_DTYPE']}, "
            f"math_fidelity={model_config['MATH_FIDELITY']}, PCC={self.pcc_message}, shape={self.output_tensor_2.shape}"
        )

        output_tensor = self.output_tensor_3
        output_tensor = ttnn.to_torch(output_tensor, device=self.device, mesh_composer=self.output_mesh_composer)
        expected_shape = self.torch_output_tensor_3.shape
        output_tensor = torch.reshape(
            output_tensor, (expected_shape[0], expected_shape[2], expected_shape[3], expected_shape[1])
        )
        output_tensor = torch.permute(output_tensor, (0, 3, 1, 2))

        batch_size = output_tensor.shape[0]

        valid_pcc = 0.97
        self.pcc_passed, self.pcc_message = check_with_pcc(self.torch_output_tensor_3, output_tensor, pcc=valid_pcc)
        # assert self.pcc_passed, logger.error(f"Instance Segmentation Head 2 PCC check failed: {self.pcc_message}")
        logger.info(
            f"Panoptic DeepLab - Instance Segmentation Center Head: batch_size={self.batch_size}, "
            f"act_dtype={model_config['ACTIVATIONS_DTYPE']}, weight_dtype={model_config['WEIGHTS_DTYPE']}, "
            f"math_fidelity={model_config['MATH_FIDELITY']}, PCC={self.pcc_message}, shape={self.output_tensor_3.shape}"
        )

        return self.pcc_passed, self.pcc_message

    def load_real_input(self, input_path: str) -> torch.Tensor:
        """Load real input from saved file"""

        if input_path.endswith(".pt"):
            # Load PyTorch tensor
            data = torch.load(input_path, map_location="cpu")
            if isinstance(data, dict):
                tensor = data["tensor"]
                logger.info(f"Loaded input metadata: {data.keys()}")
                if "stats" in data:
                    logger.info(f"Original input stats: {data['stats']}")
            else:
                tensor = data
        elif input_path.endswith(".npy"):
            # Load numpy array
            np_array = np.load(input_path)
            tensor = torch.from_numpy(np_array)
        else:
            raise ValueError(f"Unsupported input file format: {input_path}")

        # Ensure tensor is float32
        if tensor.dtype != torch.float32:
            tensor = tensor.to(torch.float32)

        return tensor


model_config = {
    "MATH_FIDELITY": ttnn.MathFidelity.LoFi,
    "WEIGHTS_DTYPE": ttnn.bfloat8_b,
    "ACTIVATIONS_DTYPE": ttnn.bfloat8_b,
}


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, in_channels, height, width, weights_path, real_input_path",
    [
        (
            1,
            3,
            512,
            1024,
            "/home/ubuntu/ign-tt-sm/tt-metal/models/experimental/panoptic_deeplab/model_final_23d03a.pkl",
            "/home/ubuntu/ign-tt-sm/tt-metal/models/experimental/panoptic_deeplab/result/fullnet/test_inputs/frankfurt_000000_005543_leftImg8bit_torch_input.pt",
        ),
    ],
)
def test_panoptic_deeplab(
    device,
    batch_size,
    in_channels,
    height,
    width,
    weights_path,
    real_input_path,
):
    PanopticDeepLabTestInfra(
        device,
        batch_size,
        in_channels,
        height,
        width,
        model_config,
        weights_path,
        real_input_path,
    )
