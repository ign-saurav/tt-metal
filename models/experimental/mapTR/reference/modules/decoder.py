# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
from typing import Optional, Tuple


def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Inverse function of sigmoid.

    Args:
        x: Tensor with values in range (0, 1).
        eps: Small value for numerical stability.

    Returns:
        Tensor after inverse sigmoid.
    """
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


class MapTRDecoder(nn.Module):
    """MapTR Decoder for iterative bounding box refinement (inference-only).

    Args:
        layers (nn.ModuleList): List of decoder layers.
        return_intermediate (bool): Whether to return intermediate outputs.
            Default: False.
    """

    def __init__(
        self,
        layers: nn.ModuleList,
        return_intermediate: bool = False,
    ):
        super().__init__()
        self.layers = layers
        self.num_layers = len(layers)
        self.return_intermediate = return_intermediate

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        query_pos: Optional[torch.Tensor] = None,
        reference_points: Optional[torch.Tensor] = None,
        reg_branches: Optional[nn.ModuleList] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward function for MapTRDecoder.

        Args:
            query: Input query with shape (num_query, bs, embed_dims).
            key: Key tensor. Default: None.
            value: Value tensor. Default: None.
            query_pos: Positional encoding for query. Default: None.
            reference_points: Reference points with shape (bs, num_query, 2).
            reg_branches: Regression branches for box refinement. Default: None.
            key_padding_mask: Key padding mask. Default: None.

        Returns:
            Tuple of:
                - output: Decoder output with shape (num_layers, num_query, bs, embed_dims)
                    if return_intermediate else (1, num_query, bs, embed_dims).
                - reference_points: Final or intermediate reference points.
        """
        output = query
        intermediate = []
        intermediate_reference_points = []

        for lid, layer in enumerate(self.layers):
            # Prepare reference points input (bs, num_query, num_level, 2)
            reference_points_input = reference_points[..., :2].unsqueeze(2)

            output = layer(
                output,
                key=key,
                value=value,
                query_pos=query_pos,
                reference_points=reference_points_input,
                key_padding_mask=key_padding_mask,
                **kwargs,
            )
            output = output.permute(1, 0, 2)

            # Iterative bounding box refinement
            if reg_branches is not None:
                tmp = reg_branches[lid](output)
                assert reference_points.shape[-1] == 2

                new_reference_points = torch.zeros_like(reference_points)
                new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points[..., :2])
                new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            output = output.permute(1, 0, 2)

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)

        return output, reference_points
