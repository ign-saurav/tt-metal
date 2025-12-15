# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import ttnn

from ...layers.conv2d import Conv2d
from ...layers.normalization import GroupNorm
from ...layers.linear import ColParallelLinear, Linear
from ...utils.substate import substate, indexed_substates
from ...parallel.config import vae_all_gather

if TYPE_CHECKING:
    pass

"""Adapted from models/experimental/tt_dit/models/vae/vae_sd35.py"""


class ResnetBlock:
    def __init__(
        self,
        mesh_device=None,
        parallel_config=None,
        ccl_manager=None,
        torch_ref=None,
    ):
        """
        Initialize the ResnetBlock.
        Args:
            torch_ref: The reference to the torch model.
            mesh_device: The device to use for the model.
            parallel_config: The parallel config to use for the model.
            ccl_manager: The ccl manager to use for the model.
        """
        self.norm1 = GroupNorm.from_torch(
            torch_ref=torch_ref.norm1,
            mesh_device=mesh_device,
            mesh_axis=parallel_config.tensor_parallel.mesh_axis,
        )
        self.norm2 = GroupNorm.from_torch(
            torch_ref=torch_ref.norm2,
            mesh_device=mesh_device,
            mesh_axis=parallel_config.tensor_parallel.mesh_axis,
        )
        self.conv1 = Conv2d.from_torch(
            torch_ref.conv1,
            mesh_device=mesh_device,
            out_mesh_axis=parallel_config.tensor_parallel.mesh_axis,
            ccl_manager=ccl_manager,
        )
        self.conv2 = Conv2d.from_torch(
            torch_ref.conv2,
            mesh_device=mesh_device,
            out_mesh_axis=parallel_config.tensor_parallel.mesh_axis,
            ccl_manager=ccl_manager,
        )
        self.conv_shortcut = None
        if torch_ref.conv_shortcut is not None:
            self.conv_shortcut = Conv2d.from_torch(
                torch_ref.conv_shortcut,
                mesh_device=mesh_device,
                out_mesh_axis=parallel_config.tensor_parallel.mesh_axis,
                ccl_manager=ccl_manager,
            )
        else:
            self.conv_shortcut = None

    def load_torch_state_dict(self, state_dict):
        self.norm1.load_torch_state_dict(state_dict["norm1"])
        self.norm2.load_torch_state_dict(state_dict["norm2"])
        self.conv1.load_torch_state_dict(state_dict["conv1"])
        self.conv2.load_torch_state_dict(state_dict["conv2"])

        if "conv_shortcut" in state_dict:
            self.conv_shortcut.load_torch_state_dict(state_dict["conv_shortcut"])

    @classmethod
    def from_torch(
        cls,
        torch_ref,
        mesh_device=None,
        parallel_config=None,
        ccl_manager=None,
    ):
        resnet_block = cls(
            torch_ref=torch_ref,
            mesh_device=mesh_device,
            parallel_config=parallel_config,
            ccl_manager=ccl_manager,
        )

        return resnet_block

    def __call__(self, x):
        residual = ttnn.clone(x)
        x = self.norm1(x)
        x = ttnn.silu(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = ttnn.silu(x)
        x = self.conv2(x)
        if self.conv_shortcut is not None:
            residual = self.conv_shortcut(residual)
        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)  # Following binary op requires tile layout
        return x + residual


class Upsample2D:
    def __init__(
        self,
        mesh_device=None,
        parallel_config=None,
        ccl_manager=None,
        torch_ref=None,
    ):
        """
        Initialize the Upsample2D block.
        Args:
            torch_ref: The reference to the torch model.
            mesh_device: The device to use for the model.
            parallel_config: The parallel config to use for the model.
            ccl_manager: The ccl manager to use for the model.
        """
        self.conv = Conv2d.from_torch(
            torch_ref.conv,
            mesh_device=mesh_device,
            out_mesh_axis=parallel_config.tensor_parallel.mesh_axis,
            ccl_manager=ccl_manager,
        )

    # Fix to align with constructor
    @classmethod
    def from_torch(cls, torch_ref, mesh_device=None, mesh_axis=None, parallel_manager=None):
        layer = cls(
            mesh_device=mesh_device,
            mesh_axis=mesh_axis,
            parallel_manager=parallel_manager,
            torch_ref=torch_ref,
        )
        return layer

    def load_torch_state_dict(self, state_dict):
        self.conv.load_torch_state_dict(state_dict["conv"])

    def __call__(self, x):
        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)  # Upsample requires row major.
        x = ttnn.upsample(x, scale_factor=2)
        x = self.conv(x)
        return x


class Downsample2D:
    def __init__(
        self,
        mesh_device=None,
        parallel_config=None,
        ccl_manager=None,
        torch_ref=None,
    ):
        """
        Initialize the Downsample2D block.
        Args:
            torch_ref: The reference to the torch model.
            mesh_device: The device to use for the model.
            parallel_config: The parallel config to use for the model.
            ccl_manager: The ccl manager to use for the model.
        """
        self.conv = Conv2d(
            torch_ref.conv.in_channels,
            torch_ref.conv.out_channels,
            kernel_size=torch_ref.conv.kernel_size,
            stride=torch_ref.conv.stride,
            padding=(0, 1, 0, 1) if not torch_ref.padding else 0,
            mesh_device=mesh_device,
            out_mesh_axis=parallel_config.tensor_parallel.mesh_axis,
            ccl_manager=ccl_manager,
        )
        if torch_ref is not None:
            self.conv.load_torch_state_dict(torch_ref.conv.state_dict())

    # Fix to align with constructor
    @classmethod
    def from_torch(cls, torch_ref, mesh_device=None, mesh_axis=None, parallel_manager=None):
        layer = cls(
            mesh_device=mesh_device,
            mesh_axis=mesh_axis,
            parallel_manager=parallel_manager,
            torch_ref=torch_ref,
        )
        return layer

    def load_torch_state_dict(self, state_dict):
        self.conv.load_torch_state_dict(state_dict["conv"])

    def __call__(self, x):
        x = self.conv(x)

        return x


class UpDecoderBlock2D:
    def __init__(
        self,
        mesh_device=None,
        parallel_config=None,
        ccl_manager=None,
        torch_ref=None,
    ):
        """
        Initialize the UpDecoderBlock2D.
        Args:
            torch_ref: The reference to the torch model.
            mesh_device: The device to use for the model.
            parallel_config: The parallel config to use for the model.
            ccl_manager: The ccl manager to use for the model.
        """
        self.resnets = [
            ResnetBlock(
                torch_ref=resnet,
                mesh_device=mesh_device,
                parallel_config=parallel_config,
                ccl_manager=ccl_manager,
            )
            for resnet in torch_ref.resnets
        ]

        self.upsamplers = [
            Upsample2D(
                torch_ref=upsampler,
                mesh_device=mesh_device,
                parallel_config=parallel_config,
                ccl_manager=ccl_manager,
            )
            for upsampler in torch_ref.upsamplers or []
        ]

    @classmethod
    def from_torch(cls, torch_ref, mesh_device=None, parallel_config=None, ccl_manager=None):
        layer = cls(
            torch_ref=torch_ref,
            mesh_device=mesh_device,
            parallel_config=parallel_config,
            ccl_manager=ccl_manager,
        )
        return layer

    def load_torch_state_dict(self, state_dict):
        for i, state in enumerate(indexed_substates(state_dict, "resnets")):
            self.resnets[i].load_torch_state_dict(state)

        for i, state in enumerate(indexed_substates(state_dict, "upsamplers")):
            self.upsamplers[i].load_torch_state_dict(state)

    def __call__(self, x):
        for resnet in self.resnets:
            x = resnet(x)
        for upsampler in self.upsamplers:
            x = upsampler(x)
        return x


class DownEncoderBlock2D:
    def __init__(
        self,
        mesh_device=None,
        parallel_config=None,
        ccl_manager=None,
        torch_ref=None,
    ):
        """
        Initialize the DownEncoderBlock2D block.
        Args:
            torch_ref: The reference to the torch model.
            mesh_device: The device to use for the model.
            parallel_config: The parallel config to use for the model.
            ccl_manager: The ccl manager to use for the model.
        """
        self.resnets = [
            ResnetBlock(
                torch_ref=resnet,
                mesh_device=mesh_device,
                parallel_config=parallel_config,
                ccl_manager=ccl_manager,
            )
            for resnet in torch_ref.resnets
        ]

        self.downsamplers = [
            Downsample2D(
                torch_ref=downsampler,
                mesh_device=mesh_device,
                parallel_config=parallel_config,
                ccl_manager=ccl_manager,
            )
            for downsampler in torch_ref.downsamplers or []
        ]

    @classmethod
    def from_torch(cls, torch_ref, mesh_device=None, parallel_config=None, ccl_manager=None):
        layer = cls(
            torch_ref=torch_ref,
            mesh_device=mesh_device,
            parallel_config=parallel_config,
            ccl_manager=ccl_manager,
        )
        return layer

    def load_torch_state_dict(self, state_dict):
        for i, state in enumerate(indexed_substates(state_dict, "resnets")):
            self.resnets[i].load_torch_state_dict(state)

        for i, state in enumerate(indexed_substates(state_dict, "downsamplers")):
            self.downsamplers[i].load_torch_state_dict(state)

    def __call__(self, x):
        for resnet in self.resnets:
            x = resnet(x)
        for downsampler in self.downsamplers:
            x = downsampler(x)
        return x


class Attention:
    def __init__(
        self,
        mesh_device=None,
        parallel_config=None,
        ccl_manager=None,
        torch_ref=None,
    ):
        """
        Initialize the Attention block.
        Args:
            torch_ref: The reference to the torch model.
            mesh_device: The device to use for the model.
            parallel_config: The parallel config to use for the model.
            ccl_manager: The ccl manager to use for the model.
        """
        self.query_dim = torch_ref.to_q.in_features
        self.num_heads = torch_ref.heads
        self.head_dim = torch_ref.to_q.out_features // self.num_heads
        self.inner_dim = self.head_dim * self.num_heads
        self.mesh_device = mesh_device
        self.parallel_config = parallel_config
        self.ccl_manager = ccl_manager
        self.to_q = Linear(in_features=self.query_dim, out_features=self.inner_dim, mesh_device=mesh_device)
        self.to_k = Linear(in_features=self.query_dim, out_features=self.inner_dim, mesh_device=mesh_device)
        self.to_v = Linear(in_features=self.query_dim, out_features=self.inner_dim, mesh_device=mesh_device)
        self.to_out = [
            ColParallelLinear(
                in_features=self.inner_dim,
                out_features=self.query_dim,
                mesh_device=mesh_device,
                mesh_axis=parallel_config.tensor_parallel.mesh_axis,
            )
        ]
        self.group_norm = GroupNorm(
            num_groups=(torch_ref.group_norm.num_groups),
            num_channels=self.query_dim,
            eps=torch_ref.group_norm.eps,
            mesh_device=mesh_device,
            mesh_axis=parallel_config.tensor_parallel.mesh_axis,
        )

        if torch_ref is not None:
            self.load_torch_state_dict(torch_ref.state_dict())

    @classmethod
    def from_torch(cls, torch_ref, mesh_device=None, parallel_config=None, ccl_manager=None):
        layer = cls(
            torch_ref=torch_ref, mesh_device=mesh_device, parallel_config=parallel_config, ccl_manager=ccl_manager
        )
        return layer

    @staticmethod
    def reorder_for_attention(x, batch_size, n_heads, head_dim):
        return ttnn.permute(ttnn.reshape(x, (batch_size, -1, n_heads, head_dim)), (0, 2, 1, 3))

    def load_torch_state_dict(self, state_dict):
        self.to_q.load_torch_state_dict(substate(state_dict, "to_q"))
        self.to_k.load_torch_state_dict(substate(state_dict, "to_k"))
        self.to_v.load_torch_state_dict(substate(state_dict, "to_v"))
        for i, state in enumerate(indexed_substates(state_dict, "to_out")):
            self.to_out[i].load_torch_state_dict(state)
        self.group_norm.load_torch_state_dict(substate(state_dict, "group_norm"))

    def gather_if_sharded(self, x):
        if x.shape[3] < self.to_q.in_features:
            x = vae_all_gather(self.ccl_manager, x, self.parallel_config.tensor_parallel.mesh_axis)
        return x

    def __call__(self, x):
        assert len(x.shape) == 4
        residual = x
        # elementwise required to be tilized
        in_layout = x.layout
        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)

        [b, h, w, c] = list(x.shape)

        # No need to transpose like reference. x is alredy channel last
        x = self.group_norm(x)
        x = self.gather_if_sharded(x)

        # output will be bxhxwx(num_heads*head_dims)
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)
        inner_dim = k.shape[-1]
        head_dim = inner_dim // self.num_heads

        q = self.reorder_for_attention(q, b, self.num_heads, head_dim)
        k = self.reorder_for_attention(k, b, self.num_heads, head_dim)
        v = self.reorder_for_attention(v, b, self.num_heads, head_dim)

        x = ttnn.transformer.scaled_dot_product_attention(q, k, v, is_causal=False)
        x = ttnn.reshape(ttnn.permute(x, (0, 2, 1, 3)), (b, h, w, inner_dim))

        for to_out in self.to_out:
            x = to_out(x)

        x = x + residual

        x = ttnn.to_layout(x, in_layout)
        return x


class UnetMidBlock2D:
    def __init__(
        self,
        mesh_device=None,
        parallel_config=None,
        ccl_manager=None,
        torch_ref=None,
    ):
        """
        Initialize the UnetMidBlock2D.
        Args:
            torch_ref: The reference to the torch model.
            mesh_device: The device to use for the model.
            parallel_config: The parallel config to use for the model.
            ccl_manager: The ccl manager to use for the model.
        """
        self.attentions = [
            Attention(
                torch_ref=attention,
                mesh_device=mesh_device,
                parallel_config=parallel_config,
                ccl_manager=ccl_manager,
            )
            for attention in torch_ref.attentions
        ]
        self.resnets = [
            ResnetBlock(
                torch_ref=resnet, mesh_device=mesh_device, parallel_config=parallel_config, ccl_manager=ccl_manager
            )
            for resnet in torch_ref.resnets
        ]

    @classmethod
    def from_torch(cls, torch_ref, mesh_device=None, parallel_config=None, ccl_manager=None):
        layer = cls(
            torch_ref=torch_ref, mesh_device=mesh_device, parallel_config=parallel_config, ccl_manager=ccl_manager
        )
        return layer

    def load_torch_state_dict(self, state_dict):
        for i, state in enumerate(indexed_substates(state_dict, "attentions")):
            self.attentions[i].load_torch_state_dict(state)
        for i, state in enumerate(indexed_substates(state_dict, "resnets")):
            self.resnets[i].load_torch_state_dict(state)

    def __call__(self, x):
        x = self.resnets[0](x)
        x = self.attentions[0](x)
        return self.resnets[1](x)


class VAEDecoder:
    def __init__(
        self,
        torch_ref=None,
        mesh_device=None,
        parallel_config=None,
        ccl_manager=None,
    ):
        """
        Initialize the VAEDecoder.
        Args:
            torch_ref: The reference to the torch model.
            mesh_device: The device to use for the model.
            parallel_config: The parallel config to use for the model.
            ccl_manager: The ccl manager to use for the model.
        """
        self.conv_in = Conv2d.from_torch(
            torch_ref.conv_in,
            mesh_device=mesh_device,
            out_mesh_axis=parallel_config.tensor_parallel.mesh_axis,
            ccl_manager=ccl_manager,
        )
        self.mid_block = UnetMidBlock2D.from_torch(
            torch_ref=torch_ref.mid_block,
            mesh_device=mesh_device,
            parallel_config=parallel_config,
            ccl_manager=ccl_manager,
        )

        self.up_blocks = [
            UpDecoderBlock2D.from_torch(
                torch_ref=up_block,
                mesh_device=mesh_device,
                parallel_config=parallel_config,
                ccl_manager=ccl_manager,
            )
            for up_block in torch_ref.up_blocks
        ]

        self.conv_norm_out = GroupNorm.from_torch(
            torch_ref=torch_ref.conv_norm_out,
            mesh_device=mesh_device,
            mesh_axis=parallel_config.tensor_parallel.mesh_axis,
        )

        self.conv_out = Conv2d.from_torch(
            torch_ref.conv_out,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
        )

        self._tp_axis = parallel_config.tensor_parallel.mesh_axis
        self._ccl_manager = ccl_manager

    @classmethod
    def from_torch(cls, torch_ref, mesh_device=None, parallel_config=None, ccl_manager=None):
        vae_model = cls(
            torch_ref=torch_ref, mesh_device=mesh_device, parallel_config=parallel_config, ccl_manager=ccl_manager
        )
        return vae_model

    def load_torch_state_dict(self, state_dict):
        self.conv_in.load_torch_state_dict(substate(state_dict, "conv_in"))
        self.mid_block.load_torch_state_dict(substate(state_dict, "mid_block"))
        for i, state in enumerate(indexed_substates(state_dict, "up_blocks")):
            self.up_blocks[i].load_torch_state_dict(state)
        self.conv_norm_out.load_torch_state_dict(substate(state_dict, "conv_norm_out"))
        self.conv_out.load_torch_state_dict(substate(state_dict, "conv_out"))

    def __call__(self, x):
        x = self.conv_in(x)
        x = self.mid_block(x)
        for up_block in self.up_blocks:
            x = up_block(x)
        x = self.conv_norm_out(x)
        x = ttnn.silu(x)
        x = vae_all_gather(self._ccl_manager, x, cluster_axis=self._tp_axis)
        x = self.conv_out(x)
        return x


class VAEEncoder:
    def __init__(
        self,
        torch_ref=None,
        mesh_device=None,
        parallel_config=None,
        ccl_manager=None,
    ):
        """
        Initialize the VAEEncoder.
        Args:
            torch_ref: The reference to the torch model.
            mesh_device: The device to use for the model.
            parallel_config: The parallel config to use for the model.
            ccl_manager: The ccl manager to use for the model.
        """
        self.conv_in = Conv2d.from_torch(
            torch_ref.conv_in,
            mesh_device=mesh_device,
            out_mesh_axis=parallel_config.tensor_parallel.mesh_axis,
            ccl_manager=ccl_manager,
        )
        self.mid_block = UnetMidBlock2D.from_torch(
            torch_ref=torch_ref.mid_block,
            mesh_device=mesh_device,
            parallel_config=parallel_config,
            ccl_manager=ccl_manager,
        )

        self.down_blocks = [
            DownEncoderBlock2D.from_torch(
                torch_ref=down_block,
                mesh_device=mesh_device,
                parallel_config=parallel_config,
                ccl_manager=ccl_manager,
            )
            for down_block in torch_ref.down_blocks
        ]

        self.conv_norm_out = GroupNorm.from_torch(
            torch_ref=torch_ref.conv_norm_out,
            mesh_device=mesh_device,
            mesh_axis=parallel_config.tensor_parallel.mesh_axis,
        )

        self.conv_out = Conv2d.from_torch(
            torch_ref.conv_out,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
        )

        self._tp_axis = parallel_config.tensor_parallel.mesh_axis
        self._ccl_manager = ccl_manager

    @classmethod
    def from_torch(cls, torch_ref, mesh_device=None, parallel_config=None, ccl_manager=None):
        vae_model = cls(
            torch_ref=torch_ref, mesh_device=mesh_device, parallel_config=parallel_config, ccl_manager=ccl_manager
        )
        return vae_model

    def load_torch_state_dict(self, state_dict):
        self.conv_in.load_torch_state_dict(substate(state_dict, "conv_in"))
        self.mid_block.load_torch_state_dict(substate(state_dict, "mid_block"))
        for i, state in enumerate(indexed_substates(state_dict, "down_blocks")):
            self.down_blocks[i].load_torch_state_dict(state)
        self.conv_norm_out.load_torch_state_dict(substate(state_dict, "conv_norm_out"))
        self.conv_out.load_torch_state_dict(substate(state_dict, "conv_out"))

    def __call__(self, x):
        x = self.conv_in(x)
        for down_block in self.down_blocks:
            x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
            x = down_block(x)
        x = self.mid_block(x)
        x = self.conv_norm_out(x)
        x = ttnn.silu(x)
        x = vae_all_gather(self._ccl_manager, x, cluster_axis=self._tp_axis)
        x = self.conv_out(x)
        return x
