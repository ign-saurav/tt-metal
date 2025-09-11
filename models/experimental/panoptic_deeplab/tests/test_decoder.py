# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger
from ttnn.model_preprocessing import preprocess_model_parameters
from ttnn.model_preprocessing import infer_ttnn_module_args, preprocess_model_parameters
import ttnn
from tests.ttnn.utils_for_testing import check_with_pcc
from models.experimental.panoptic_deeplab.tt.decoder import (
    TTDecoder,
    decoder_layer_optimisations,
)
from models.experimental.panoptic_deeplab.tt.custom_preprocessing import create_custom_mesh_preprocessor
from models.experimental.panoptic_deeplab.reference.decoder import (
    DecoderModel,
)
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

    # # BACKBONE MAPPINGS (REVERSE)
    # if key.startswith("backbone."):
    #     # Stem batch norm mappings (do this first to avoid conflicts)
    #     key = key.replace("backbone.stem.conv1.norm.", "backbone.stem.bn1.")
    #     key = key.replace("backbone.stem.conv2.norm.", "backbone.stem.bn2.")
    #     key = key.replace("backbone.stem.conv3.norm.", "backbone.stem.bn3.")

    #     # Layer mapping: res2/3/4/5 -> layer1/2/3/4
    #     key = key.replace("backbone.res2.", "backbone.layer1.")
    #     key = key.replace("backbone.res3.", "backbone.layer2.")
    #     key = key.replace("backbone.res4.", "backbone.layer3.")
    #     key = key.replace("backbone.res5.", "backbone.layer4.")

    #     # Batch norm mapping: conv1/2/3.norm -> bn1/2/3
    #     key = key.replace(".conv1.norm.", ".bn1.")
    #     key = key.replace(".conv2.norm.", ".bn2.")
    #     key = key.replace(".conv3.norm.", ".bn3.")

    #     # Downsample mapping: shortcut -> downsample
    #     key = key.replace(".shortcut.norm.", ".downsample.1.")
    #     # Handle shortcut.weight specifically to avoid matching shortcut.norm
    #     if ".shortcut." in key and ".shortcut.norm." not in checkpoint_key:
    #         key = key.replace(".shortcut.", ".downsample.0.")

    #     return key

    # SEMANTIC HEAD MAPPINGS (REVERSE)
    # if key.startswith("sem_seg_head."):
    #     # Replace base prefix

    #     key = key.replace("sem_seg_head.", "semantic_decoder.")
    ##################################################
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
    ############################################
    # # INSTANCE HEAD MAPPINGS (REVERSE)
    # elif key.startswith("ins_embed_head."):
    #     # Replace base prefix
    #     key = key.replace("ins_embed_head.", "instance_decoder.")

    #     # Center head mappings (do these first)
    #     if ".center_head.0.norm." in key:
    #         key = key.replace(".center_head.0.norm.", ".head_2.conv1.1.")
    #     elif ".center_head.0." in key:
    #         key = key.replace(".center_head.0.", ".head_2.conv1.0.")
    #     elif ".center_head.1.norm." in key:
    #         key = key.replace(".center_head.1.norm.", ".head_2.conv2.1.")
    #     elif ".center_head.1." in key:
    #         key = key.replace(".center_head.1.", ".head_2.conv2.0.")
    #     elif ".center_predictor." in key:
    #         key = key.replace(".center_predictor.", ".head_2.conv3.0.")

    #     # Offset head mappings
    #     elif ".offset_head.depthwise.norm." in key:
    #         key = key.replace(".offset_head.depthwise.norm.", ".head_1.conv1.1.")
    #     elif ".offset_head.depthwise." in key:
    #         key = key.replace(".offset_head.depthwise.", ".head_1.conv1.0.")
    #     elif ".offset_head.pointwise.norm." in key:
    #         key = key.replace(".offset_head.pointwise.norm.", ".head_1.conv2.1.")
    #     elif ".offset_head.pointwise." in key:
    #         key = key.replace(".offset_head.pointwise.", ".head_1.conv2.0.")
    #     elif ".offset_predictor." in key:
    #         key = key.replace(".offset_predictor.", ".head_1.conv3.0.")

    #     # ASPP mappings (res5 -> aspp)
    #     elif ".decoder.res5.project_conv." in key:
    #         # Special case for ASPP_3_Depthwise
    #         if ".convs.3.depthwise.norm." in key:
    #             key = key.replace(".decoder.res5.project_conv.convs.3.depthwise.norm.", ".aspp.ASPP_3_Depthwise.1.")
    #         elif ".convs.3.depthwise." in key:
    #             key = key.replace(".decoder.res5.project_conv.convs.3.depthwise.", ".aspp.ASPP_3_Depthwise.0.")

    #         # ASPP_0_Conv
    #         elif ".convs.0.norm." in key:
    #             key = key.replace(".decoder.res5.project_conv.convs.0.norm.", ".aspp.ASPP_0_Conv.1.")
    #         elif ".convs.0." in key:
    #             key = key.replace(".decoder.res5.project_conv.convs.0.", ".aspp.ASPP_0_Conv.0.")

    #         # ASPP_1 Depthwise and Pointwise
    #         elif ".convs.1.depthwise.norm." in key:
    #             key = key.replace(".decoder.res5.project_conv.convs.1.depthwise.norm.", ".aspp.ASPP_1_Depthwise.1.")
    #         elif ".convs.1.depthwise." in key:
    #             key = key.replace(".decoder.res5.project_conv.convs.1.depthwise.", ".aspp.ASPP_1_Depthwise.0.")
    #         elif ".convs.1.pointwise.norm." in key:
    #             key = key.replace(".decoder.res5.project_conv.convs.1.pointwise.norm.", ".aspp.ASPP_1_pointwise.1.")
    #         elif ".convs.1.pointwise." in key:
    #             key = key.replace(".decoder.res5.project_conv.convs.1.pointwise.", ".aspp.ASPP_1_pointwise.0.")

    #         # ASPP_2 Depthwise and Pointwise
    #         elif ".convs.2.depthwise.norm." in key:
    #             key = key.replace(".decoder.res5.project_conv.convs.2.depthwise.norm.", ".aspp.ASPP_2_Depthwise.1.")
    #         elif ".convs.2.depthwise." in key:
    #             key = key.replace(".decoder.res5.project_conv.convs.2.depthwise.", ".aspp.ASPP_2_Depthwise.0.")
    #         elif ".convs.2.pointwise.norm." in key:
    #             key = key.replace(".decoder.res5.project_conv.convs.2.pointwise.norm.", ".aspp.ASPP_2_pointwise.1.")
    #         elif ".convs.2.pointwise." in key:
    #             key = key.replace(".decoder.res5.project_conv.convs.2.pointwise.", ".aspp.ASPP_2_pointwise.0.")

    #         # ASPP_3 Pointwise
    #         elif ".convs.3.pointwise.norm." in key:
    #             key = key.replace(".decoder.res5.project_conv.convs.3.pointwise.norm.", ".aspp.ASPP_3_pointwise.1.")
    #         elif ".convs.3.pointwise." in key:
    #             key = key.replace(".decoder.res5.project_conv.convs.3.pointwise.", ".aspp.ASPP_3_pointwise.0.")

    #         # ASPP_4_Conv
    #         elif ".convs.4." in key:
    #             key = key.replace(".decoder.res5.project_conv.convs.4.1.", ".aspp.ASPP_4_Conv_1.0.")

    #         # ASPP project
    #         elif ".project.norm." in key:
    #             key = key.replace(".decoder.res5.project_conv.project.norm.", ".aspp.ASPP_project.1.")
    #         elif ".project." in key:
    #             key = key.replace(".decoder.res5.project_conv.project.", ".aspp.ASPP_project.0.")

    #     # Decoder res3 mappings
    #     elif ".decoder.res3." in key:
    #         if ".project_conv.norm." in key:
    #             key = key.replace(".decoder.res3.project_conv.norm.", ".res3.conv1.1.")
    #         elif ".project_conv." in key:
    #             key = key.replace(".decoder.res3.project_conv.", ".res3.conv1.0.")
    #         elif ".fuse_conv.depthwise.norm." in key:
    #             key = key.replace(
    #                 ".decoder.res3.fuse_conv.depthwise.norm.", ".res3.conv2.1."
    #             )
    #         elif ".fuse_conv.depthwise." in key:
    #             key = key.replace(
    #                 ".decoder.res3.fuse_conv.depthwise.", ".res3.conv2.0."
    #             )
    #         elif ".fuse_conv.pointwise.norm." in key:
    #             key = key.replace(
    #                 ".decoder.res3.fuse_conv.pointwise.norm.", ".res3.conv3.1."
    #             )
    #         elif ".fuse_conv.pointwise." in key:
    #             key = key.replace(
    #                 ".decoder.res3.fuse_conv.pointwise.", ".res3.conv3.0."
    #             )

    #     # Decoder res2 mappings
    #     elif ".decoder.res2." in key:
    #         if ".project_conv.norm." in key:
    #             key = key.replace(".decoder.res2.project_conv.norm.", ".res2.conv1.1.")
    #         elif ".project_conv." in key:
    #             key = key.replace(".decoder.res2.project_conv.", ".res2.conv1.0.")
    #         elif ".fuse_conv.depthwise.norm." in key:
    #             key = key.replace(
    #                 ".decoder.res2.fuse_conv.depthwise.norm.", ".res2.conv2.1."
    #             )
    #         elif ".fuse_conv.depthwise." in key:
    #             key = key.replace(
    #                 ".decoder.res2.fuse_conv.depthwise.", ".res2.conv2.0."
    #             )
    #         elif ".fuse_conv.pointwise.norm." in key:
    #             key = key.replace(
    #                 ".decoder.res2.fuse_conv.pointwise.norm.", ".res2.conv3.1."
    #             )
    #         elif ".fuse_conv.pointwise." in key:
    #             key = key.replace(
    #                 ".decoder.res2.fuse_conv.pointwise.", ".res2.conv3.0."
    #             )

    #     return key

    return ""


class HeadTestInfra:
    def __init__(
        self,
        device,
        batch_size,
        model_config,
        in_channels,
        res3_intermediate_channels,
        res2_intermediate_channels,
        out_channels,
        upsample_channels,
        height,
        width,
        name,
        weights_path,
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
        self.res3_intermediate_channels = res3_intermediate_channels
        self.res2_intermediate_channels = res2_intermediate_channels
        self.out_channels = out_channels
        self.upsample_channels = upsample_channels
        self.height = height
        self.width = width
        self.name = name
        self.inputs_mesh_mapper, self.weights_mesh_mapper, self.output_mesh_composer = self.get_mesh_mappers(device)
        self.weights_path = os.path.join(os.path.dirname(__file__), "weights", f"{weights_path}")
        # Create input tensors
        self.torch_input_tensor = torch.randn(
            (self.batch_size, self.in_channels, self.height, self.width), dtype=torch.float
        )

        # Create res3 and res2 feature maps with appropriate dimensions
        self.torch_res3_tensor = torch.randn((self.batch_size, 512, self.height * 2, self.width * 2), dtype=torch.float)

        self.torch_res2_tensor = torch.randn(
            (self.batch_size, upsample_channels, self.height * 4, self.width * 4), dtype=torch.float
        )

        # torch model
        torch_model = DecoderModel(self.name)
        torch_model = load_torch_model_state(torch_model, name)

        #########################################################
        # Load weights if provided
        # print("self.weights_path", self.weights_path)
        # if self.weights_path:
        if self.weights_path and os.path.exists(self.weights_path):
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
                print("self.name::::::::::::", self.name)
                if "Semantics_head" in self.name:
                    print("semantics_head::::::::::::")
                    checkpoint_dict = {
                        k[len("sem_seg_head.") :] if k.startswith("sem_seg_head.") else k: v
                        for k, v in state_dict.items()
                        if k.startswith("sem_seg_head.")
                    }
                    # checkpoint_dict = {k: v for k, v in state_dict.items() if k.startswith("sem_seg_head")}
                elif "instance" in self.name:
                    print("instance_head::::::::::::")
                    checkpoint_dict = {k: v for k, v in state_dict.items() if k.startswith("ins_embed_head")}
                checkpoint_keys = set(checkpoint_dict.keys())
                #####################################################
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
                ############################
                # my_state_dict = torch_model.res2.state_dict()
                # for key, value in my_state_dict.items():
                #     print(key)
                # print("--------------------------------")
                #############################
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
                # print("-------------------------------")
                # print("-------------------------------")
                # print("-------------------------------")
                # print("-------------------------------")
                # print("-------------------------------")
                # for k, v in key_mapping.items():
                #     print(k)
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
                        logger.info(f"Successfully loaded {load_ratio:.1%} of model weights - excellent coverage!")
                    elif load_ratio >= 0.5:
                        logger.warning(f"Loaded {load_ratio:.1%} of model weights - decent coverage")
                    else:
                        logger.error(f"Only loaded {load_ratio:.1%} of model weights - poor coverage")

                # Print sample weight values for a few loaded checkpoint keys and their mapped model keys
                logger.info("Sample loaded weights (checkpoint key -> model key):")
                for i, (checkpoint_key, model_key) in enumerate(key_mapping.items()):
                    if i >= 3:
                        break
                    # checkpoint_key = "backbone.res4.5.conv3.weight"
                    # model_key = "backbone.layer3.5.conv3.weight"
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

                # print(f"Conv weight: {conv_layer.weight[:5]}")
                # print(f"Conv bias: {conv_layer.bias[:5]}")
                # print(f"BN running_mean: {bn_layer.running_mean[:5]}")  # First 5 values
                # print(f"BN running_var: {bn_layer.running_var[:5]}")
                # print(f"BN weight (gamma): {bn_layer.weight[:5]}")
                # print(f"BN bias (beta): {bn_layer.bias[:5]}")

                # Verify sample parameters were updated
                sample_params = list(torch_model.parameters())[:3]
                if all(torch.any(p != 0) for p in sample_params):
                    logger.info("Weight verification passed - parameters contain non-zero values")
                else:
                    logger.warning("Weight verification failed - found zero parameters")

            except Exception as e:
                logger.error(f"Failed to load weights file: {str(e)}")
                logger.warning("Falling back to random initialization")

            # else:
            #     logger.warning("Manual mapping failed - using random weights")
            #     logger.warning("The checkpoint architecture is incompatible with your model")
        else:
            logger.warning("No weights provided - using random initialization")

        # if torch_model is not None:
        #     # Get original conv weight (before BN folding)
        #     original_conv = torch_model.backbone.layer3[5].conv3
        #     original_bn = torch_model.backbone.layer3[5].bn3

        #     print("\nOriginal Conv3 weight (before BN folding):")
        #     print(f"  Mean: {original_conv.weight.mean():.4f}")
        #     print(f"  Std: {original_conv.weight.std():.4f}")

        #     # Manually fold BN to see the effect
        #     from ttnn.model_preprocessing import fold_batch_norm2d_into_conv2d

        #     folded_weight, folded_bias = fold_batch_norm2d_into_conv2d(original_conv, original_bn)

        #     print("\nFolded Conv3 weight (after BN folding):")
        #     print(f"  Mean: {folded_weight.mean():.4f}")
        #     print(f"  Std: {folded_weight.std():.4f}")

        # logger.info("PyTorch model initialization completed")
        #########################################################

        parameters = preprocess_model_parameters(
            initialize_model=lambda: torch_model,
            custom_preprocessor=create_custom_mesh_preprocessor(self.weights_mesh_mapper),
            device=None,
        )

        # For ASPP
        aspp_args = infer_ttnn_module_args(
            model=torch_model.aspp, run_model=lambda model: model(self.torch_input_tensor), device=None
        )
        if hasattr(parameters, "aspp"):
            parameters.aspp.conv_args = aspp_args

        # For res3
        res3_output = torch_model.aspp(self.torch_input_tensor)
        res3_args = infer_ttnn_module_args(
            model=torch_model.res3, run_model=lambda model: model(res3_output, self.torch_res3_tensor), device=None
        )
        if hasattr(parameters, "res3"):
            parameters.res3.conv_args = res3_args

        # For res2
        res2_input = torch_model.res3(res3_output, self.torch_res3_tensor)
        res2_args = infer_ttnn_module_args(
            model=torch_model.res2, run_model=lambda model: model(res2_input, self.torch_res2_tensor), device=None
        )
        if hasattr(parameters, "res2"):
            parameters.res2.conv_args = res2_args

        # For head
        head_input = torch_model.res2(res2_input, self.torch_res2_tensor)
        head_1_args = infer_ttnn_module_args(
            model=torch_model.head_1, run_model=lambda model: model(head_input), device=None
        )
        if hasattr(parameters, "head_1"):
            parameters.head_1.conv_args = head_1_args

        if "instance" in name:
            head_2_args = infer_ttnn_module_args(
                model=torch_model.head_2, run_model=lambda model: model(head_input), device=None
            )
            if hasattr(parameters, "head_2"):
                parameters.head_2.conv_args = head_2_args

        # Get torch output with all three inputs
        self.torch_output_tensor, self.torch_output_tensor_2 = torch_model(
            self.torch_input_tensor, self.torch_res3_tensor, self.torch_res2_tensor
        )
        return

        # Convert torch tensors to TTNN host tensors
        def to_ttnn_host(tensor):
            return ttnn.from_torch(
                tensor.permute(0, 2, 3, 1),
                dtype=ttnn.bfloat8_b,
                device=device,
                mesh_mapper=self.inputs_mesh_mapper,
            )

        tt_host_tensor = to_ttnn_host(self.torch_input_tensor)
        tt_res3_tensor = to_ttnn_host(self.torch_res3_tensor)
        tt_res2_tensor = to_ttnn_host(self.torch_res2_tensor)

        # Move TTNN host tensors to device
        self.input_tensor = ttnn.to_device(tt_host_tensor, device)
        self.res3_tensor = ttnn.to_device(tt_res3_tensor, device)
        self.res2_tensor = ttnn.to_device(tt_res2_tensor, device)

        # ttnn model
        logger.info("Initializing TTNN model...")
        self.ttnn_model = TTDecoder(
            parameters, model_config, layer_optimisations=decoder_layer_optimisations[self.name], name=self.name
        )

        # run and validate
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

    # Compute golden output for the selected block using a helper function
    def run(self):
        self.output_tensor, self.output_tensor_2 = self.ttnn_model(
            self.input_tensor,
            self.res3_tensor,
            self.res2_tensor,
            self.upsample_channels,
            self.device,
        )

        return self.output_tensor, self.output_tensor_2

    def validate(self, output_tensor=None):
        output_tensor = self.output_tensor if output_tensor is None else output_tensor
        output_tensor = ttnn.to_torch(output_tensor, device=self.device, mesh_composer=self.output_mesh_composer)
        expected_shape = self.torch_output_tensor.shape
        output_tensor = torch.reshape(
            output_tensor, (expected_shape[0], expected_shape[2], expected_shape[3], expected_shape[1])
        )
        output_tensor = torch.permute(output_tensor, (0, 3, 1, 2))

        batch_size = output_tensor.shape[0]

        valid_pcc = 0.99
        self.pcc_passed, self.pcc_message = check_with_pcc(self.torch_output_tensor, output_tensor, pcc=valid_pcc)
        assert self.pcc_passed, logger.error(f"PCC check failed: {self.pcc_message}")
        logger.info(
            f"{self.name},  batch_size={batch_size}, act_dtype={model_config['ACTIVATIONS_DTYPE']}, weight_dtype={model_config['WEIGHTS_DTYPE']}, math_fidelity={model_config['MATH_FIDELITY']}, PCC={self.pcc_message}"
        )

        if "instance" in self.name:
            output_tensor = self.output_tensor_2
            output_tensor = ttnn.to_torch(output_tensor, device=self.device, mesh_composer=self.output_mesh_composer)
            expected_shape = self.torch_output_tensor_2.shape
            output_tensor = torch.reshape(
                output_tensor, (expected_shape[0], expected_shape[2], expected_shape[3], expected_shape[1])
            )
            output_tensor = torch.permute(output_tensor, (0, 3, 1, 2))

            batch_size = output_tensor.shape[0]

            valid_pcc = 0.99
            self.pcc_passed, self.pcc_message = check_with_pcc(self.torch_output_tensor_2, output_tensor, pcc=valid_pcc)
            assert self.pcc_passed, logger.error(f"PCC check failed: {self.pcc_message}")
            logger.info(
                f"{self.name},  batch_size={batch_size}, act_dtype={model_config['ACTIVATIONS_DTYPE']}, weight_dtype={model_config['WEIGHTS_DTYPE']}, math_fidelity={model_config['MATH_FIDELITY']}, PCC={self.pcc_message}"
            )

        return self.pcc_passed, self.pcc_message


model_config = {
    "MATH_FIDELITY": ttnn.MathFidelity.LoFi,
    "WEIGHTS_DTYPE": ttnn.bfloat8_b,
    "ACTIVATIONS_DTYPE": ttnn.bfloat8_b,
}


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, in_channels, res3_intermediate_channels, res2_intermediate_channels, out_channels, upsample_channels, height, width, name, weights_path",
    [
        (
            1,
            2048,
            320,
            288,
            (19,),
            256,
            32,
            64,
            "Semantics_head",
            "/home/ubuntu/ign-tt-sm/tt-metal/models/experimental/panoptic_deeplab/model_final_23d03a.pkl",
        ),  # semantic head
        # (1, 2048, 320, 160, (2, 1), 256, 32, 64, "instance_head", "/home/ubuntu/ign-tt-sm/tt-metal/models/experimental/panoptic_deeplab/model_final_23d03a.pkl"),  # instance offset head
    ],
)
def test_decoder(
    device,
    batch_size,
    in_channels,
    res3_intermediate_channels,
    res2_intermediate_channels,
    out_channels,
    upsample_channels,
    height,
    width,
    name,
    weights_path,
):
    DecoderTestInfra(
        device,
        batch_size,
        model_config,
        in_channels,
        res3_intermediate_channels,
        res2_intermediate_channels,
        out_channels,
        upsample_channels,
        height,
        width,
        name,
        weights_path,
    )
