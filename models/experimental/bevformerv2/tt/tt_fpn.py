# SPDX-FileCopyrightText: © 2025
# SPDX-License-Identifier: Apache-2.0

import ttnn

from models.experimental.bevformerv2.tt.utils import TTConv2D
from models.experimental.bevformerv2.tt.model_configs import BevFormerV2ModelConfig


class TtConvModule:
    """
    Lightweight wrapper around :class:`TTConv2D` for FPN.

    This mirrors the small helper used in other experimental models (e.g. UniAD, VAD),
    but plugs into the configurable :class:`BevFormerV2ModelConfig`.
    """

    def __init__(
        self,
        conv_args,
        conv_pth,
        *,
        device=None,
        model_configs: BevFormerV2ModelConfig | None = None,
        layer_path: str | None = None,
        is_blk: bool = False,
        dealloc_act: bool = True,
    ):
        self.device = device
        # Keep a handle to the inferred conv metadata so we can recover (B, H, W)
        # for reshape / upsample inside the FPN top‑down pathway.
        self.meta = conv_args.conv

        self.conv = TTConv2D(
            conv_args.conv,
            conv_pth.conv,
            device=self.device,
            dealloc_act=dealloc_act,
            is_blk=is_blk,
            model_configs=model_configs,
            layer_path=layer_path,
        )

    def __call__(self, x):
        # TTConv2D conventionally returns (output, out_h, out_w)
        x, _, _ = self.conv(x)
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
    * This class intentionally follows the layout and tensor reshaping strategy used
      in :mod:`models.experimental.uniad.tt.ttnn_fpn` for ease of cross‑model parity.
    """

    def __init__(
        self,
        conv_args,
        conv_pth,
        device,
        *,
        model_configs: BevFormerV2ModelConfig | None = None,
    ):
        self.device = device
        self.start_level = 0

        # Lateral and FPN convs are stored as Python lists for cheap iteration.
        self.lateral_convs: list[TtConvModule] = []
        self.fpn_convs: list[TtConvModule] = []

        # Metadata for each lateral level (B, H, W) inferred from conv_args.
        self._lateral_meta = []

        num_lateral = len(conv_args.lateral_convs)
        num_fpn = len(conv_args.fpn_convs)
        assert num_fpn >= num_lateral, "FPN must have at least as many fpn_convs as lateral_convs"

        # ------------------------
        # Build lateral convolutions
        # ------------------------
        for i in range(num_lateral):
            lat_args = conv_args.lateral_convs[i]
            lat_pth = conv_pth.lateral_convs[i]

            self._lateral_meta.append(lat_args.conv)

            self.lateral_convs.append(
                TtConvModule(
                    lat_args,
                    lat_pth,
                    device=device,
                    model_configs=model_configs,
                    layer_path=f"fpn.lateral_convs.{i}.conv",
                    # FPN laterals are cheap; we typically deallocate their activations
                    # once they are consumed by the next stage.
                    dealloc_act=True,
                )
            )

        # ------------------------
        # Build FPN convolutions
        # ------------------------
        for i in range(num_fpn):
            fpn_args = conv_args.fpn_convs[i]
            fpn_pth = conv_pth.fpn_convs[i]

            is_extra_level = i >= num_lateral

            # Extra levels (P6, P7, ...) often feed into many downstream heads,
            # so we keep their activations alive by default (dealloc_act=False).
            dealloc_act = not is_extra_level

            self.fpn_convs.append(
                TtConvModule(
                    fpn_args,
                    fpn_pth,
                    device=device,
                    model_configs=model_configs,
                    layer_path=f"fpn.fpn_convs.{i}.conv",
                    is_blk=False,
                    dealloc_act=dealloc_act,
                )
            )

        self._num_lateral = num_lateral
        self._num_fpn = num_fpn

    def _upsample_and_add(self, top: ttnn.Tensor, bottom: ttnn.Tensor, level: int) -> ttnn.Tensor:
        """
        Upsample ``top`` feature map and add it to ``bottom``.

        Both tensors are in the flattened [1, 1, B * H * W, C] format coming out of
        :class:`TTConv2D`. We:

          1. convert to ROW_MAJOR layout
          2. reshape to [B, H, W, C]
          3. upsample by factor 2 (nearest‑neighbour)
          4. crop to the spatial size of the *bottom* level
          5. reshape back to [1, 1, B * H * W, C] and switch to TILE layout
        """
        # Metadata for the current and previous lateral levels.
        coarse_meta = self._lateral_meta[level]
        fine_meta = self._lateral_meta[level - 1]

        coarse_b = coarse_meta.batch_size
        coarse_h = coarse_meta.input_height
        coarse_w = coarse_meta.input_width

        fine_h = fine_meta.input_height
        fine_w = fine_meta.input_width

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
            # We do not need the raw backbone feature after the lateral conv.
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
        # We follow the common RetinaNet / BEVFormer configuration:
        #   - extra levels are built from the last FPN output
        for i in range(used_backbone_levels, self._num_fpn):
            outs.append(self.fpn_convs[i](outs[-1]))

        return tuple(outs)
