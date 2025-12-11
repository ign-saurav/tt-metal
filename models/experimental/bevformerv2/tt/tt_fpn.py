# SPDX-FileCopyrightText: © 2025
# SPDX-License-Identifier: Apache-2.0

import ttnn
from typing import Optional

from models.tt_cnn.tt.builder import TtConv2d
from models.experimental.bevformerv2.tt.utils import create_conv2d_configuration
from models.experimental.bevformerv2.tt.model_configs import BevFormerV2ModelConfig
from models.experimental.bevformerv2.tt.config import TtFPNConvConfigs, TtFPNConfigs


class TtConvModule:
    """
    Lightweight wrapper around :class:`TtConv2d` for FPN.

    Plugs into the configurable :class:`BevFormerV2ModelConfig`.
    """

    def __init__(
        self,
        conv_args=None,
        conv_pth=None,
        *,
        device=None,
        model_configs: BevFormerV2ModelConfig | None = None,
        layer_path: str | None = None,
        is_blk: bool = False,
        dealloc_act: bool = True,
        configs: Optional[TtFPNConvConfigs] = None,
    ):
        self.device = device

        # Use provided configs or build them inline
        if configs is not None:
            # Use configs from config.py
            self.meta = None  # Metadata not needed when using configs
            self.conv = TtConv2d(configs.conv, self.device)
        else:
            # Build configs inline (backward compatibility)
            if conv_args is None or conv_pth is None:
                raise ValueError("Either configs must be provided, or conv_args and conv_pth must be provided")

            # Keep a handle to the inferred conv metadata so we can recover (B, H, W)
            # for reshape / upsample inside the FPN top‑down pathway.
            self.meta = conv_args.conv

            conv_config = create_conv2d_configuration(
                conv_args.conv,
                conv_pth.conv,
                device=self.device,
                dealloc_act=dealloc_act,
                is_blk=is_blk,
                model_configs=model_configs,
                layer_path=layer_path,
            )
            self.conv = TtConv2d(conv_config, self.device)

    def __call__(self, x):
        # TtConv2d returns just the output tensor
        x = self.conv(x)
        return x


class TtFPN:
    """
    TTNN implementation of the MMDetection‑style FPN used by BEVFormerV2.

    The behaviour mirrors :class:`models.experimental.bevformerv2.reference.fpn.FPN`
    for the common BEVFormer configuration:

      - inputs: C3, C4, C5 feature maps from the backbone
      - lateral 1x1 convolutions on each input level
      - top‑down pathway with nearest‑neighbour upsampling and elementwise add
      - per‑level 3x3 FPN convolutions
      - optional extra levels implemented as stride‑2 convolutions on the last output

    Notes
    -----
    * ``conv_args`` is expected to be the ``infer_ttnn_module_args`` output for the
      FPN module, with ``conv_args.lateral_convs`` and ``conv_args.fpn_convs`` lists.
    * ``conv_pth`` should carry the preprocessed TTNN weights in matching structure:
      ``conv_pth.lateral_convs[i].conv`` / ``conv_pth.fpn_convs[i].conv``.
    """

    def __init__(
        self,
        conv_args=None,
        conv_pth=None,
        device=None,
        *,
        model_configs: BevFormerV2ModelConfig | None = None,
        configs: Optional[TtFPNConfigs] = None,
    ):
        self.device = device
        self.start_level = 0

        # Use provided configs or build them inline
        if configs is not None:
            self.configs = configs
            # Metadata not needed when using configs, but initialize empty list
            self._lateral_meta = []
        else:
            # Build configs inline (backward compatibility)
            if conv_args is None or conv_pth is None or device is None:
                raise ValueError("Either configs must be provided, or conv_args, conv_pth, and device must be provided")

            from models.experimental.bevformerv2.tt.config import create_fpn_configs

            self.configs = create_fpn_configs(conv_args, conv_pth, device, model_configs)

            # Store metadata for backward compatibility
            self._lateral_meta = []
            for i in range(len(conv_args.lateral_convs)):
                self._lateral_meta.append(conv_args.lateral_convs[i].conv)

        # Lateral and FPN convs are stored as Python lists for cheap iteration.
        self.lateral_convs: list[TtConvModule] = []
        self.fpn_convs: list[TtConvModule] = []

        num_lateral = len(self.configs.lateral_convs)
        num_fpn = len(self.configs.fpn_convs)
        assert num_fpn >= num_lateral, "FPN must have at least as many fpn_convs as lateral_convs"

        # ------------------------
        # Build lateral convolutions
        # ------------------------
        for i in range(num_lateral):
            self.lateral_convs.append(
                TtConvModule(
                    device=device,
                    configs=self.configs.lateral_convs[i],
                )
            )

        # ------------------------
        # Build FPN convolutions
        # ------------------------
        for i in range(num_fpn):
            self.fpn_convs.append(
                TtConvModule(
                    device=device,
                    configs=self.configs.fpn_convs[i],
                )
            )

        self._num_lateral = num_lateral
        self._num_fpn = num_fpn

    def _upsample_and_add(self, top: ttnn.Tensor, bottom: ttnn.Tensor, level: int) -> ttnn.Tensor:
        """
        Upsample ``top`` feature map and add it to ``bottom``.

        Both tensors are in the flattened [1, 1, B * H * W, C] format coming out of
        :class:`TtConv2d`. We:

          1. convert to ROW_MAJOR layout
          2. reshape to [B, H, W, C]
          3. upsample by factor 2 (nearest‑neighbour)
          4. crop to the spatial size of the *bottom* level
          5. reshape back to [1, 1, B * H * W, C] and switch to TILE layout
        """
        # Metadata for the current and previous lateral levels.
        # Try to get from stored metadata, otherwise infer from configs
        if len(self._lateral_meta) > level:
            coarse_meta = self._lateral_meta[level]
            fine_meta = self._lateral_meta[level - 1]
            coarse_b = coarse_meta.batch_size
            coarse_h = coarse_meta.input_height
            coarse_w = coarse_meta.input_width
            fine_h = fine_meta.input_height
            fine_w = fine_meta.input_width
        else:
            # Infer from configs if metadata not available
            coarse_config = self.configs.lateral_convs[level].conv
            fine_config = self.configs.lateral_convs[level - 1].conv
            coarse_b = coarse_config.batch_size
            coarse_h = coarse_config.input_height
            coarse_w = coarse_config.input_width
            fine_h = fine_config.input_height
            fine_w = fine_config.input_width

        # Convert to row‑major layout and unflatten.
        top = ttnn.to_layout(top, ttnn.ROW_MAJOR_LAYOUT)
        top = ttnn.reshape(top, (coarse_b, coarse_h, coarse_w, top.shape[-1]))

        # Nearest‑neighbour upsample by 2 (mirrors MMDetection FPN default).
        top = ttnn.upsample(top, 2)

        # Spatial crop to match the target resolution if necessary.
        # (odd feature sizes may cause +1 rows/cols after upsample).
        if top.shape[1] != fine_h or top.shape[2] != fine_w:
            top = top[:, :fine_h, :fine_w, :]

        # Re‑flatten back to [1, 1, B * H * W, C].
        top = ttnn.reshape(
            top,
            (
                1,
                1,
                coarse_b * fine_h * fine_w,
                top.shape[-1],
            ),
        )

        # Move to DRAM + TILE layout for elementwise add with ``bottom``.
        top = ttnn.sharded_to_interleaved(top, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        top = ttnn.to_layout(top, ttnn.TILE_LAYOUT)

        return ttnn.add(bottom, top)

    def __call__(self, inputs):
        """
        Parameters
        ----------
        inputs:
            List of TTNN tensors corresponding to backbone feature maps
            (e.g. [C3, C4, C5]) in the flattened [1, 1, B * H * W, C] format.
        """
        assert len(inputs) >= self._num_lateral, "Not enough backbone levels for FPN"

        # ------------------------
        # Step 1: Lateral connections
        # ------------------------
        laterals: list[ttnn.Tensor] = []
        for i, lateral_conv in enumerate(self.lateral_convs):
            x = lateral_conv(inputs[i + self.start_level])
            ttnn.deallocate(inputs[i + self.start_level])
            laterals.append(x)

        used_backbone_levels = len(laterals)

        # ------------------------
        # Step 2: Top‑down pathway
        # ------------------------
        for i in range(used_backbone_levels - 1, 0, -1):
            laterals[i - 1] = self._upsample_and_add(laterals[i], laterals[i - 1], level=i)

        # ------------------------
        # Step 3: Final FPN convs for lateral levels
        # ------------------------
        outs: list[ttnn.Tensor] = []
        for i in range(used_backbone_levels):
            # Ensure DRAM memory before feeding into convs.
            lateral = ttnn.to_memory_config(laterals[i], memory_config=ttnn.DRAM_MEMORY_CONFIG)
            out = self.fpn_convs[i](lateral)
            outs.append(out)

        # ------------------------
        # Step 4: Extra levels (P6, P7, ...)
        # ------------------------
        for i in range(used_backbone_levels, self._num_fpn):
            outs.append(self.fpn_convs[i](outs[-1]))

        return tuple(outs)
