# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import List, Optional

import ttnn
from models.tt_cnn.tt.builder import (
    Conv2dConfiguration,
    MaxPool2dConfiguration,
)
from models.experimental.bevformerv2.tt.model_configs import BevFormerV2ModelConfig


@dataclass
class TtResNetStemConfigs:
    """Configuration for ResNet stem (conv1 + maxpool)."""

    conv1: Conv2dConfiguration
    maxpool: MaxPool2dConfiguration


@dataclass
class TtBottleneckConfigs:
    """Configuration for a single ResNet bottleneck block."""

    conv1: Conv2dConfiguration
    conv2: Conv2dConfiguration
    conv3: Conv2dConfiguration
    downsample: Optional[Conv2dConfiguration] = None


@dataclass
class TtResNetLayerConfigs:
    """Configuration for a ResNet layer (multiple bottleneck blocks)."""

    bottlenecks: List[TtBottleneckConfigs]


@dataclass
class TtResNet50Configs:
    """Complete configuration for ResNet-50 backbone."""

    stem: TtResNetStemConfigs
    layer1: TtResNetLayerConfigs
    layer2: TtResNetLayerConfigs
    layer3: TtResNetLayerConfigs
    layer4: TtResNetLayerConfigs


@dataclass
class TtFPNConvConfigs:
    """Configuration for a single FPN convolution."""

    conv: Conv2dConfiguration


@dataclass
class TtFPNConfigs:
    """Complete configuration for FPN."""

    lateral_convs: List[TtFPNConvConfigs]
    fpn_convs: List[TtFPNConvConfigs]


@dataclass
class TtBEVFormerV2Configs:
    """Complete configuration for BEVFormerV2 model."""

    resnet: TtResNet50Configs
    fpn: Optional[TtFPNConfigs] = None


class TtResNet50ConfigBuilder:
    """Builder for ResNet-50 configuration."""

    def __init__(
        self,
        conv_args,
        conv_pth,
        device: ttnn.Device,
        model_configs: Optional[BevFormerV2ModelConfig] = None,
    ):
        self.conv_args = conv_args
        self.conv_pth = conv_pth
        self.device = device
        self.model_configs = model_configs

    def build_configs(self) -> TtResNet50Configs:
        """Build all ResNet-50 configurations."""
        return TtResNet50Configs(
            stem=self._build_stem_configs(),
            layer1=self._build_layer_configs("layer1", 3),
            layer2=self._build_layer_configs("layer2", 4),
            layer3=self._build_layer_configs("layer3", 6),
            layer4=self._build_layer_configs("layer4", 3),
        )

    def _build_stem_configs(self) -> TtResNetStemConfigs:
        """Build stem (conv1 + maxpool) configurations."""
        # Conv1
        conv1_config = self._create_conv_config(
            self.conv_args.conv1,
            self.conv_pth.conv1,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            act_block_h=32,
            layer_path="stem.conv1",
        )

        # MaxPool
        conv1_channels = conv1_config.out_channels
        maxpool_config = self._create_maxpool_config(
            self.conv_args.maxpool,
            channels=conv1_channels,
        )

        return TtResNetStemConfigs(conv1=conv1_config, maxpool=maxpool_config)

    def _build_layer_configs(self, layer_name: str, num_blocks: int) -> TtResNetLayerConfigs:
        """Build configurations for a ResNet layer."""
        bottlenecks = []
        # Access layer as attribute (e.g., conv_args.layer1)
        layer_args = getattr(self.conv_args, layer_name)
        for block_idx in range(num_blocks):
            block_path = f"{layer_name}.{block_idx}"
            is_downsample = block_idx == 0

            bottleneck_configs = self._build_bottleneck_configs(
                layer_args[block_idx],
                getattr(self.conv_pth, f"{layer_name}_{block_idx}"),
                is_downsample=is_downsample,
                block_path=block_path,
            )
            bottlenecks.append(bottleneck_configs)

        return TtResNetLayerConfigs(bottlenecks=bottlenecks)

    def _build_bottleneck_configs(
        self,
        conv_args,
        conv_pth,
        is_downsample: bool = False,
        block_path: Optional[str] = None,
    ) -> TtBottleneckConfigs:
        """Build configurations for a bottleneck block."""
        from models.experimental.bevformerv2.tt.tt_bottleneck import (
            get_bottleneck_optimisation,
        )

        layer_optimisations = get_bottleneck_optimisation(block_path)

        # Conv1
        conv1_opts = layer_optimisations.conv1.copy()
        conv1_config = self._create_conv_config(
            conv_args.conv1,
            conv_pth.conv1,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            layer_path=f"{block_path}.conv1" if block_path else None,
            **conv1_opts,
        )

        # Conv2
        conv2_opts = layer_optimisations.conv2.copy()
        conv2_config = self._create_conv_config(
            conv_args.conv2,
            conv_pth.conv2,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            layer_path=f"{block_path}.conv2" if block_path else None,
            **conv2_opts,
        )

        # Conv3
        conv3_opts = layer_optimisations.conv3.copy()
        conv3_config = self._create_conv_config(
            conv_args.conv3,
            conv_pth.conv3,
            activation=None,
            layer_path=f"{block_path}.conv3" if block_path else None,
            **conv3_opts,
        )

        # Downsample (if needed)
        downsample_config = None
        if is_downsample:
            downsample_opts = layer_optimisations.downsample.copy()
            downsample_config = self._create_conv_config(
                conv_args.downsample[0],
                conv_pth.downsample,
                activation=None,
                layer_path=f"{block_path}.downsample" if block_path else None,
                **downsample_opts,
            )

        return TtBottleneckConfigs(
            conv1=conv1_config,
            conv2=conv2_config,
            conv3=conv3_config,
            downsample=downsample_config,
        )

    def _create_conv_config(
        self,
        conv_args,
        conv_pth,
        activation: Optional[ttnn.UnaryWithParam] = None,
        activation_dtype: ttnn.DataType = ttnn.bfloat16,
        weights_dtype: ttnn.DataType = ttnn.bfloat8_b,
        shard_layout: ttnn.TensorMemoryLayout = ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        is_blk: bool = False,
        dealloc_act: bool = False,
        act_block_h: Optional[int] = None,
        layer_path: Optional[str] = None,
        **kwargs,
    ) -> Conv2dConfiguration:
        """Create a Conv2dConfiguration."""
        from models.experimental.bevformerv2.tt.utils import create_conv2d_configuration

        return create_conv2d_configuration(
            conv_args=conv_args,
            conv_pth=conv_pth,
            device=self.device,
            activation=activation,
            activation_dtype=activation_dtype,
            weights_dtype=weights_dtype,
            shard_layout=shard_layout,
            is_blk=is_blk,
            dealloc_act=dealloc_act,
            act_block_h=act_block_h,
            model_configs=self.model_configs,
            layer_path=layer_path,
            **kwargs,
        )

    def _create_maxpool_config(
        self,
        maxpool_args,
        channels: int,
        **kwargs,
    ) -> MaxPool2dConfiguration:
        """Create a MaxPool2dConfiguration."""
        from models.experimental.bevformerv2.tt.utils import create_maxpool2d_configuration

        return create_maxpool2d_configuration(
            maxpool_args=maxpool_args,
            channels=channels,
            **kwargs,
        )


class TtFPNConfigBuilder:
    """Builder for FPN configuration."""

    def __init__(
        self,
        conv_args,
        conv_pth,
        device: ttnn.Device,
        model_configs: Optional[BevFormerV2ModelConfig] = None,
    ):
        self.conv_args = conv_args
        self.conv_pth = conv_pth
        self.device = device
        self.model_configs = model_configs

    def build_configs(self) -> TtFPNConfigs:
        """Build all FPN configurations."""
        lateral_convs = []
        fpn_convs = []

        # Build lateral convolutions
        num_lateral = len(self.conv_args.lateral_convs)
        for i in range(num_lateral):
            lat_args = self.conv_args.lateral_convs[i]
            lat_pth = self.conv_pth.lateral_convs[i]

            # Extract .conv from both args and pth for FPN structure
            conv_config = self._create_conv_config(
                lat_args.conv,
                lat_pth.conv,
                dealloc_act=True,
                layer_path=f"fpn.lateral_convs.{i}.conv",
            )
            lateral_convs.append(TtFPNConvConfigs(conv=conv_config))

        # Build FPN convolutions
        num_fpn = len(self.conv_args.fpn_convs)
        for i in range(num_fpn):
            fpn_args = self.conv_args.fpn_convs[i]
            fpn_pth = self.conv_pth.fpn_convs[i]

            is_extra_level = i >= num_lateral
            dealloc_act = not is_extra_level

            # Extract .conv from both args and pth for FPN structure
            conv_config = self._create_conv_config(
                fpn_args.conv,
                fpn_pth.conv,
                is_blk=False,
                dealloc_act=dealloc_act,
                layer_path=f"fpn.fpn_convs.{i}.conv",
            )
            fpn_convs.append(TtFPNConvConfigs(conv=conv_config))

        return TtFPNConfigs(lateral_convs=lateral_convs, fpn_convs=fpn_convs)

    def _create_conv_config(
        self,
        conv_args,
        conv_pth,
        activation: Optional[ttnn.UnaryWithParam] = None,
        activation_dtype: ttnn.DataType = ttnn.bfloat16,
        weights_dtype: ttnn.DataType = ttnn.bfloat8_b,
        shard_layout: ttnn.TensorMemoryLayout = ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        is_blk: bool = False,
        dealloc_act: bool = False,
        act_block_h: Optional[int] = None,
        layer_path: Optional[str] = None,
        **kwargs,
    ) -> Conv2dConfiguration:
        """Create a Conv2dConfiguration."""
        from models.experimental.bevformerv2.tt.utils import create_conv2d_configuration

        return create_conv2d_configuration(
            conv_args=conv_args,
            conv_pth=conv_pth,
            device=self.device,
            activation=activation,
            activation_dtype=activation_dtype,
            weights_dtype=weights_dtype,
            shard_layout=shard_layout,
            is_blk=is_blk,
            dealloc_act=dealloc_act,
            act_block_h=act_block_h,
            model_configs=self.model_configs,
            layer_path=layer_path,
            **kwargs,
        )


def create_resnet50_configs(
    conv_args,
    conv_pth,
    device: ttnn.Device,
    model_configs: Optional[BevFormerV2ModelConfig] = None,
) -> TtResNet50Configs:
    """
    Create ResNet-50 configuration object given weights and input tensor dimensions.

    Parameters
    ----------
    conv_args:
        Arguments containing convolution parameters (from infer_ttnn_module_args)
    conv_pth:
        Preprocessed TTNN weights
    device:
        TTNN device
    model_configs:
        Optional model configuration overrides

    Returns
    -------
    TtResNet50Configs:
        Complete ResNet-50 configuration
    """
    builder = TtResNet50ConfigBuilder(conv_args, conv_pth, device, model_configs)
    return builder.build_configs()


def create_fpn_configs(
    conv_args,
    conv_pth,
    device: ttnn.Device,
    model_configs: Optional[BevFormerV2ModelConfig] = None,
) -> TtFPNConfigs:
    """
    Create FPN configuration object given weights and input tensor dimensions.

    Parameters
    ----------
    conv_args:
        Arguments containing convolution parameters (from infer_ttnn_module_args)
    conv_pth:
        Preprocessed TTNN weights
    device:
        TTNN device
    model_configs:
        Optional model configuration overrides

    Returns
    -------
    TtFPNConfigs:
        Complete FPN configuration
    """
    builder = TtFPNConfigBuilder(conv_args, conv_pth, device, model_configs)
    return builder.build_configs()


def create_bevformerv2_configs(
    resnet_conv_args,
    resnet_conv_pth,
    device: ttnn.Device,
    fpn_conv_args: Optional = None,
    fpn_conv_pth: Optional = None,
    model_configs: Optional[BevFormerV2ModelConfig] = None,
) -> TtBEVFormerV2Configs:
    """
    Create complete BEVFormerV2 configuration object.

    Parameters
    ----------
    resnet_conv_args:
        ResNet arguments containing convolution parameters
    resnet_conv_pth:
        Preprocessed ResNet TTNN weights
    device:
        TTNN device
    fpn_conv_args:
        Optional FPN arguments containing convolution parameters
    fpn_conv_pth:
        Optional preprocessed FPN TTNN weights
    model_configs:
        Optional model configuration overrides

    Returns
    -------
    TtBEVFormerV2Configs:
        Complete BEVFormerV2 configuration
    """
    resnet_configs = create_resnet50_configs(resnet_conv_args, resnet_conv_pth, device, model_configs)

    fpn_configs = None
    if fpn_conv_args is not None and fpn_conv_pth is not None:
        fpn_configs = create_fpn_configs(fpn_conv_args, fpn_conv_pth, device, model_configs)

    return TtBEVFormerV2Configs(resnet=resnet_configs, fpn=fpn_configs)
