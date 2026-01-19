# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""TTNN MapTR implementation modules.

This package provides a complete end-to-end TTNN implementation of MapTR
for map element detection in autonomous driving applications.

Components:
- TtResNet50: TTNN ResNet50 backbone
- TtFPN: TTNN Feature Pyramid Network
- TtBEVFormerEncoder: TTNN BEVFormer encoder
- TtMapTRDecoder: TTNN MapTR decoder
- TtMapTRPerceptionTransformer: TTNN perception transformer
- TtMapTRHead: TTNN detection head
- TtMapTR: Complete TTNN MapTR model

Usage:
    from models.experimental.MapTR.tt import TtMapTR, create_ttnn_maptr_model
    from models.experimental.MapTR.tt.weight_loading import (
        create_maptr_parameters_from_torch_model,
        load_maptr_checkpoint,
    )
"""

from models.experimental.MapTR.tt.backbone import TtResNet50
from models.experimental.MapTR.tt.fpn import TtFPN
from models.experimental.MapTR.tt.encoder import TtBEVFormerEncoder, TtBEVFormerLayer
from models.experimental.MapTR.tt.decoder import TtMapTRDecoder
from models.experimental.MapTR.tt.transformer import TtMapTRPerceptionTransformer
from models.experimental.MapTR.tt.head import TtMapTRHead, TtLearnedPositionalEncoding
from models.experimental.MapTR.tt.maptr import TtMapTR, create_ttnn_maptr_model

# Utility modules
from models.experimental.MapTR.tt.ffn import TtFFN
from models.experimental.MapTR.tt.mha import TtMultiheadAttention
from models.experimental.MapTR.tt.detr_transformer_decoder_layer import TtDetrTransformerDecoderLayer
from models.experimental.MapTR.tt.temporal_self_attention import TtTemporalSelfAttention
from models.experimental.MapTR.tt.spatial_cross_attention import TtSpatialCrossAttention, TtMSDeformableAttention3D
from models.experimental.MapTR.tt.custom_defrmble_attention import TtCustomMSDeformableAttention
from models.experimental.MapTR.tt.bottleneck import TtBottleneck

__all__ = [
    # Main model
    "TtMapTR",
    "create_ttnn_maptr_model",
    # Backbone
    "TtResNet50",
    "TtBottleneck",
    # Neck
    "TtFPN",
    # Transformer
    "TtMapTRPerceptionTransformer",
    "TtBEVFormerEncoder",
    "TtBEVFormerLayer",
    "TtMapTRDecoder",
    # Head
    "TtMapTRHead",
    "TtLearnedPositionalEncoding",
    # Attention modules
    "TtFFN",
    "TtMultiheadAttention",
    "TtDetrTransformerDecoderLayer",
    "TtTemporalSelfAttention",
    "TtSpatialCrossAttention",
    "TtMSDeformableAttention3D",
    "TtCustomMSDeformableAttention",
]
