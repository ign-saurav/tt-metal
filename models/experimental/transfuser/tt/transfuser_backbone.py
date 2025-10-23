# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
from models.experimental.transfuser.tt.utils import TTConv2D
from loguru import logger
from models.experimental.transfuser.tt.gpt import TTGpt
from models.experimental.transfuser.tt.stages import Ttstages


class TtTransfuserBackbone:
    def __init__(
        self,
        device,
        parameters,
        stride,
        model_config,
        config,
    ) -> None:
        self.device = device
        self.config = config
        self.inplanes = 32
        self.conv1 = TTConv2D(
            kernel_size=3,
            stride=2,
            padding=1,
            parameters=parameters.image_encoder.features.conv1,
            kernel_fidelity=model_config,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            deallocate_activation=True,
            reallocate_halo_output=True,
            reshard_if_not_optimal=True,
            enable_act_double_buffer=True,
            enable_weights_double_buffer=True,
            dtype=ttnn.bfloat16,
        )
        self.lidar_conv1 = TTConv2D(
            kernel_size=3,
            stride=2,
            padding=1,
            parameters=parameters.lidar_encoder._model.conv1,
            kernel_fidelity=model_config,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            deallocate_activation=True,
            reallocate_halo_output=True,
            reshard_if_not_optimal=True,
            enable_act_double_buffer=True,
            enable_weights_double_buffer=True,
            dtype=ttnn.bfloat16,
        )
        # Layer1 for both encoders
        self.image_layer1 = Ttstages._make_layer(
            parameters=parameters.image_encoder.features.layer1,
            planes=72,
            blocks=2,  # b1 and b2
            stride=2,
            groups=3,  # conv2
            model_config=model_config,
            stage_name="layer1",
        )

        self.lidar_layer1 = Ttstages._make_layer(
            parameters=parameters.lidar_encoder._model.layer1,
            planes=72,
            blocks=2,
            stride=2,
            groups=3,
            model_config=model_config,
            stage_name="layer1",
        )

        self.transformer1 = TTGpt(
            device=self.device,
            parameters=parameters["transformer1"],
            n_head=config.n_head,
            n_layer=config.n_layer,
            use_velocity=config.use_velocity,
            img_vert_anchors=config.img_vert_anchors,
            img_horz_anchors=config.img_horz_anchors,
            lidar_vert_anchors=config.lidar_vert_anchors,
            lidar_horz_anchors=config.lidar_horz_anchors,
            seq_len=config.seq_len,
            n_embd=72,  # layer1 output channels
            dtype=ttnn.bfloat16,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

    # def _make_layer(
    #     self,
    #     parameters,
    #     planes: int,
    #     blocks: int,
    #     stride: int,
    #     groups: int = 1,
    #     model_config=None,
    #     stage_name=None,
    # ) -> List[TTRegNetBottleneck]:
    #     layers = []

    #     # Determine shard layout based on stage name
    #     if stage_name == "layer1":
    #         shard_layout = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
    #     elif stage_name == "layer2":
    #         shard_layout = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
    #     elif stage_name == "layer3":
    #         shard_layout = ttnn.TensorMemoryLayout.WIDTH_SHARDED
    #     elif stage_name == "layer4":
    #         shard_layout = ttnn.TensorMemoryLayout.WIDTH_SHARDED
    #     else:
    #         # Default to HEIGHT_SHARDED for backward compatibility
    #         shard_layout = ttnn.TensorMemoryLayout.HEIGHT_SHARDED

    #     # First block (may have downsample)
    #     downsample = stride != 1 or self.inplanes != planes
    #     layers.append(
    #         TTRegNetBottleneck(
    #             parameters=parameters["b1"],
    #             model_config=model_config,
    #             stride=stride,
    #             downsample=downsample,
    #             groups=groups,
    #             shard_layout=shard_layout,
    #         )
    #     )
    #     self.inplanes = planes

    #     # Remaining blocks
    #     for block_num in range(1, blocks):
    #         block_name = f"b{block_num + 1}"
    #         layers.append(
    #             TTRegNetBottleneck(
    #                 parameters=parameters[block_name],
    #                 model_config=model_config,
    #                 stride=1,
    #                 downsample=False,
    #                 groups=groups,
    #                 shard_layout=shard_layout,
    #             )
    #         )

    #     return layers

    def normalize_imagenet_ttnn(self, x):
        """Optimized normalization that avoids slice/concat overhead"""
        # Convert from [0,255] to [0,1]
        x = ttnn.multiply(x, 1.0 / 255.0)

        # Create normalization constants as tensors
        # Mean: [0.485, 0.456, 0.406], Std: [0.229, 0.224, 0.225]
        mean = ttnn.from_torch(
            torch.tensor([0.485, 0.456, 0.406]).reshape(1, 1, 1, 3),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
        )
        std_inv = ttnn.from_torch(
            torch.tensor([1.0 / 0.229, 1.0 / 0.224, 1.0 / 0.225]).reshape(1, 1, 1, 3),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
        )

        # Normalize all channels at once (no slice/concat needed)
        x = ttnn.subtract(x, mean)
        x = ttnn.multiply(x, std_inv)

        return x

    def __call__(self, image_x, lidar_x, velocity, device):
        # Process image input
        logger.info(f"image_encoder_conv1")
        image_x = self.normalize_imagenet_ttnn(image_x)
        image_out, image_shape = self.conv1(device, image_x, image_x.shape)
        # Reshape to spatial dimensions: 80 * 352 = 28160
        # out = ttnn.reshape(out, (1, 80, 352, 32))
        # out = ttnn.permute(out, (0, 3, 1, 2))
        logger.info(f"lidar_encoder_conv1")
        # Process lidar input
        lidar_out, lidar_shape = self.lidar_conv1(device, lidar_x, lidar_x.shape)
        print("..........................................")
        print(f"{image_out.shape=}")
        print(image_shape)
        # print(im
        # age_shape)
        print(f"{lidar_out.shape=}")
        print(lidar_shape)

        logger.info(f"image_encoder_layer1")
        # image_out = ttnn.reshape(image_out, (1, 80, 352, 32))
        # image_out = ttnn.reshape(image_out, image_shape)
        print(f"{image_out.shape=}")
        # Process layer1 blocks
        for block in self.image_layer1:
            image_out, image_shape = block(image_out, device, image_shape)

        logger.info(f"lidar_encoder_layer1")
        # lidar_out = ttnn.reshape(lidar_out, lidar_shape)
        # lidar_out = ttnn.reshape(lidar_out, (1, 128, 128, 32))
        for block in self.lidar_layer1:
            lidar_out, lidar_shape = block(lidar_out, device, lidar_shape)

        logger.info(f"img_avgpool")

        # image_h = image_out.shape[1]
        image_h = image_shape[1]
        # image_w = image_out.shape[2]
        image_w = image_shape[2]
        # image_w = image_out.shape[2]
        image_c = image_shape[3]

        image_features_flat = image_out
        print(f"eeeeeeeeeeeeeee{image_features_flat.shape=}")
        # image_features_flat = ttnn.reshape(image_out, (1, 1, image_shape[0] * image_h * image_w, image_c))
        image_embd_layer1 = ttnn.adaptive_avg_pool2d(
            input_tensor=image_features_flat,
            batch_size=image_shape[0],
            input_h=image_h,
            input_w=image_w,
            channels=image_c,
            output_size=[self.config.img_vert_anchors, self.config.img_horz_anchors],
        )
        print(f"{image_embd_layer1.shape=}")
        logger.info(f"lidar_avgpool")
        lidar_h = lidar_shape[1]
        lidar_w = lidar_shape[2]
        lidar_c = lidar_shape[3]

        # lidar_features_flat = ttnn.reshape(lidar_out, (1, 1, lidar_out.shape[0] * lidar_h * lidar_w, lidar_c))
        lidar_features_flat = lidar_out
        print(f"eeeeeeeeeeeeeee{lidar_features_flat.shape=}")
        lidar_embd_layer1 = ttnn.adaptive_avg_pool2d(
            input_tensor=lidar_features_flat,
            batch_size=lidar_shape[0],
            input_h=lidar_h,
            input_w=lidar_w,
            channels=lidar_c,
            output_size=[self.config.lidar_vert_anchors, self.config.lidar_horz_anchors],
        )
        logger.info(f"Layer1 transformer")

        image_embd_layer1 = ttnn.to_memory_config(image_embd_layer1, ttnn.DRAM_MEMORY_CONFIG)
        image_embd_layer1 = ttnn.to_layout(image_embd_layer1, ttnn.TILE_LAYOUT)

        lidar_embd_layer1 = ttnn.to_memory_config(lidar_embd_layer1, ttnn.DRAM_MEMORY_CONFIG)
        lidar_embd_layer1 = ttnn.to_layout(lidar_embd_layer1, ttnn.TILE_LAYOUT)

        image_features_layer1, lidar_features_layer1 = self.transformer1(
            image_embd_layer1, lidar_embd_layer1, velocity, 72
        )
        print(f"{image_features_layer1.shape=}")
        image_features_layer1 = ttnn.permute(image_features_layer1, (0, 2, 3, 1))
        lidar_features_layer1 = ttnn.permute(lidar_features_layer1, (0, 2, 3, 1))

        logger.info(f"Layer1 image and lidar interpolation- bilinear")
        logger.info(f"bilinear_image")
        image_features_layer1 = ttnn.to_layout(image_features_layer1, ttnn.ROW_MAJOR_LAYOUT)
        image_features_layer1 = ttnn.to_memory_config(image_features_layer1, ttnn.DRAM_MEMORY_CONFIG)
        image_features_layer1 = ttnn.pad(
            image_features_layer1, padding=((0, 0), (0, 0), (0, 0), (0, 24)), value=0.0  # Pad 24 channels (96 - 72)
        )
        image_features_layer1 = ttnn.upsample(
            image_features_layer1, scale_factor=(8, 8), mode="bilinear", memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        # Slice back to original 72 channels
        image_features_layer1 = ttnn.slice(image_features_layer1, [0, 0, 0, 0], [1, 40, 176, 72])
        image_features_layer1 = ttnn.to_layout(image_features_layer1, ttnn.TILE_LAYOUT)

        logger.info(f"bilinear_lidar")
        lidar_features_layer1 = ttnn.to_layout(lidar_features_layer1, ttnn.ROW_MAJOR_LAYOUT)
        lidar_features_layer1 = ttnn.to_memory_config(lidar_features_layer1, ttnn.DRAM_MEMORY_CONFIG)
        lidar_features_layer1 = ttnn.pad(lidar_features_layer1, padding=((0, 0), (0, 0), (0, 0), (0, 24)), value=0.0)
        lidar_features_layer1 = ttnn.upsample(
            lidar_features_layer1, scale_factor=(8, 8), mode="bilinear", memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        # Slice back to original 72 channels
        lidar_features_layer1 = ttnn.slice(lidar_features_layer1, [0, 0, 0, 0], [1, 64, 64, 72])
        lidar_features_layer1 = ttnn.to_layout(lidar_features_layer1, ttnn.TILE_LAYOUT)
        print(f"{image_out.shape=}")
        print(f"{image_features_layer1.shape=}")
        print(f"{lidar_out.shape=}")
        print(f"{lidar_features_layer1.shape=}")
        logger.info("Image and lidar - add")
        image_features_layer1 = ttnn.reshape(image_features_layer1, image_out.shape)
        lidar_features_layer1 = ttnn.reshape(lidar_features_layer1, lidar_out.shape)
        image_features = ttnn.add(image_out, image_features_layer1)
        lidar_features = ttnn.add(lidar_out, lidar_features_layer1)

        return image_features, lidar_features
