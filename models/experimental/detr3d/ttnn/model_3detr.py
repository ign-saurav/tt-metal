# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import math
import ttnn
import torch
import numpy as np

from models.common.lightweightmodule import LightweightModule
from models.experimental.detr3d.reference.model_3detr import BoxProcessor
from models.experimental.detr3d.ttnn.masked_transformer_encoder import (
    TtnnTransformerEncoderLayer,
    TtnnMaskedTransformerEncoder,
    EncoderLayerArgs,
)
from models.experimental.detr3d.ttnn.transformer_decoder import (
    TtnnTransformerDecoder,
    TtnnTransformerDecoderLayer,
    DecoderLayerArgs,
)
from models.experimental.detr3d.ttnn.generic_mlp import TtnnGenericMLP
from models.experimental.detr3d.ttnn.pointnet_samodule_votes import TtnnPointnetSAModuleVotes, TtnnFurthestPointSampling
from models.experimental.detr3d.reference.torch_pointnet2_ops import furthest_point_sample
from models.experimental.detr3d.ttnn.position_embedding import TtnnPositionEmbeddingCoordsSine
from models.experimental.detr3d.ttnn.constant import ON_DEVICE


class TtnnModel3DETR(LightweightModule):
    """
    NOTE: The Encoder and Decoder layers use batch first as compared to batch second in reference model,
          this helps remove lot of unnecessary permute operations within the network.

    Main 3DETR model. Consists of the following learnable sub-models
    - pre_encoder: takes raw point cloud, subsamples it and projects into "D" dimensions
                Input is a Nx3 matrix of N point coordinates
                Output is a N'xD matrix of N' point features
    - encoder: series of self-attention blocks to extract point features
                Input is a N'xD matrix of N' point features
                Output is a N''xD matrix of N'' point features.
                N'' = N' for regular encoder; N'' = N'//2 for masked encoder
    - query computation: samples a set of B coordinates from the N'' points
                and outputs a BxD matrix of query features.
    - decoder: series of self-attention and cross-attention blocks to produce BxD box features
                Takes N''xD features from the encoder and BxD query features.
    - mlp_heads: Predicts bounding box parameters and classes from the BxD box features
    """

    def __init__(
        self,
        pre_encoder,
        encoder,
        decoder,
        dataset_config,
        encoder_dim=256,
        decoder_dim=256,
        position_embedding="fourier",
        num_queries=256,
        parameters=None,
        device=None,
    ):
        # NOTE: Layers and tensors with "torch_" in their names are on host
        super().__init__()
        self.pre_encoder = pre_encoder
        self.encoder = encoder
        self.parameters = parameters
        self.device = device
        self.encoder_to_decoder_projection = TtnnGenericMLP(
            parameters.encoder_to_decoder_projection,
            device,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            output_use_activation=True,
        )
        self.pos_embedding = TtnnPositionEmbeddingCoordsSine(
            pos_type=position_embedding, normalize=True, parameters=parameters.pos_embedding, device=self.device
        )
        self.query_projection = TtnnGenericMLP(
            parameters.query_projection, device, output_use_activation=True, deallocate_activation=True
        )
        self.decoder = decoder
        self.build_mlp_heads()

        self.num_queries = num_queries
        self.torch_box_processor = BoxProcessor(dataset_config)
        # self.box_processor = TtnnBoxProcessor(dataset_config, device=self.device)

    def build_mlp_heads(self):
        self.mlp_heads = {
            "sem_cls_head": TtnnGenericMLP(
                self.parameters.mlp_heads.sem_cls_head,
                self.device,
            ),
            "center_head": TtnnGenericMLP(
                self.parameters.mlp_heads.center_head,
                self.device,
            ),
            "size_head": TtnnGenericMLP(
                self.parameters.mlp_heads.size_head,
                self.device,
            ),
            "angle_cls_head": TtnnGenericMLP(
                self.parameters.mlp_heads.angle_cls_head,
                self.device,
            ),
            "angle_residual_head": TtnnGenericMLP(
                self.parameters.mlp_heads.angle_residual_head,
                self.device,
            ),
        }

    def get_query_embeddings_ttnn(self, torch_encoder_xyz, point_cloud_dims):
        # torch_query_inds = furthest_point_sample(torch_encoder_xyz, self.num_queries)
        torch_query_inds = TtnnFurthestPointSampling()(torch_encoder_xyz, self.num_queries, device=self.device)
        # torch_query_inds = torch_query_inds.long()
        torch_query_inds = ttnn.typecast(torch_query_inds, dtype=ttnn.uint32)
        # torch_query_xyz = [torch.gather(torch_encoder_xyz[..., x], 1, torch_query_inds) for x in range(3)]
        ttnn_query_xyz = []
        for x in range(3):
            # Extract the x-th dimension and gather
            dim_xyz = ttnn.unsqueeze(torch_encoder_xyz[..., x], -1)  # Add last dim for gather
            gathered = ttnn.gather(dim_xyz, 1, torch_query_inds)
            ttnn_query_xyz.append(ttnn.squeeze(gathered, -1))  # Remove the extra dim
        ttnn_query_xyz = ttnn.stack(ttnn_query_xyz, dim=0)
        ttnn_query_xyz = ttnn.permute(ttnn_query_xyz, (1, 2, 0))

        pos_embed = self.pos_embedding(ttnn_query_xyz, input_range=point_cloud_dims)
        query_embed = self.query_projection(pos_embed)
        ttnn.deallocate(pos_embed)
        return ttnn_query_xyz, query_embed

    def get_query_embeddings(self, torch_encoder_xyz, point_cloud_dims):
        torch_query_inds = furthest_point_sample(torch_encoder_xyz, self.num_queries)
        torch_query_inds = torch_query_inds.long()
        torch_query_xyz = [torch.gather(torch_encoder_xyz[..., x], 1, torch_query_inds) for x in range(3)]
        torch_query_xyz = torch.stack(torch_query_xyz)
        torch_query_xyz = torch_query_xyz.permute(1, 2, 0)

        pos_embed = self.pos_embedding(torch_query_xyz, input_range=point_cloud_dims)
        query_embed = self.query_projection(pos_embed)
        ttnn.deallocate(pos_embed)
        return torch_query_xyz, query_embed

    def _break_up_pc(self, torch_pc):
        # pc may contain color/normals.

        torch_xyz = torch_pc[..., 0:3].contiguous()
        return torch_xyz

    def _break_up_pc_ttnn(self, torch_pc):
        # Safe host-side slice
        torch_xyz = torch_pc[..., :3]

        return torch_xyz

    def run_encoder(self, torch_point_clouds):
        if ON_DEVICE:
            torch_xyz = self._break_up_pc_ttnn(torch_point_clouds)
        else:
            torch_xyz = self._break_up_pc(torch_point_clouds)
        torch_pre_enc_xyz, pre_enc_features, _ = self.pre_encoder(torch_xyz)
        # xyz: batch x npoints x 3
        # features: batch x channel x npoints
        # inds: batch x npoints

        # MultiHeadAttention in encoder expects batch x npoints x channel features
        pre_enc_features = ttnn.permute(pre_enc_features, (0, 2, 1))

        # xyz points are batch x npoint x channel order torch tensor
        torch_enc_xyz, enc_features, _ = self.encoder(pre_enc_features, xyz=torch_pre_enc_xyz)
        ttnn.deallocate(pre_enc_features)

        return torch_enc_xyz, enc_features, _

    def get_box_predictions(self, torch_query_xyz, torch_point_cloud_dims, box_features):
        """
        Parameters:
            torch_query_xyz: batch x nqueries x 3 tensor of query XYZ coords
            torch_point_cloud_dims: List of [min, max] dims of point cloud
                              min: batch x 3 tensor of min XYZ coords
                              max: batch x 3 tensor of max XYZ coords
            box_features: num_layers x num_queries x batch x channel
        """
        # box_features change to (num_layers x batch) x num_queries x channel
        num_layers, batch, num_queries, channel = box_features.shape
        box_features = ttnn.reshape(box_features, (num_layers * batch, num_queries, channel))
        box_features = ttnn.to_memory_config(box_features, ttnn.DRAM_MEMORY_CONFIG)

        # mlp head outputs are (num_layers x batch) x nqueries x noutput
        cls_logits = self.mlp_heads["sem_cls_head"](box_features)
        center_offset = self.mlp_heads["center_head"](box_features)
        size_normalized = self.mlp_heads["size_head"](box_features)
        angle_logits = self.mlp_heads["angle_cls_head"](box_features)
        angle_residual_normalized = self.mlp_heads["angle_residual_head"](box_features)
        ttnn.deallocate(box_features)

        center_offset = ttnn.sigmoid(center_offset) - 0.5
        size_normalized = ttnn.sigmoid(size_normalized)

        # reshape outputs to num_layers x batch x nqueries x noutput
        cls_logits = ttnn.reshape(cls_logits, (num_layers, batch, num_queries, cls_logits.shape[-1]))
        center_offset = ttnn.reshape(center_offset, (num_layers, batch, num_queries, center_offset.shape[-1]))
        size_normalized = ttnn.reshape(size_normalized, (num_layers, batch, num_queries, size_normalized.shape[-1]))
        angle_logits = ttnn.reshape(angle_logits, (num_layers, batch, num_queries, angle_logits.shape[-1]))
        angle_residual_normalized = ttnn.reshape(
            angle_residual_normalized, (num_layers, batch, num_queries, angle_residual_normalized.shape[-1])
        )
        angle_residual = angle_residual_normalized * (np.pi / angle_residual_normalized.shape[-1])

        # send outputs to torch for box processing
        return (
            cls_logits,
            center_offset,
            size_normalized,
            angle_logits,
            angle_residual_normalized,
            angle_residual,
            num_layers,
            torch_query_xyz,
            torch_point_cloud_dims,
        )

    def forward(self, inputs, encoder_only=False):
        torch_point_clouds = inputs["point_clouds"]

        torch_enc_xyz, enc_features, _ = self.run_encoder(torch_point_clouds)
        enc_features = self.encoder_to_decoder_projection(enc_features)
        # encoder features: batch x npoints x channel
        # encoder xyz: npoints x batch x 3

        if encoder_only:
            # return: batch x npoints x channels
            return torch_enc_xyz, enc_features

        torch_point_cloud_dims = [
            inputs["point_cloud_dims_min"],
            inputs["point_cloud_dims_max"],
        ]
        if ON_DEVICE:
            torch_query_xyz, query_embed = self.get_query_embeddings_ttnn(torch_enc_xyz, torch_point_cloud_dims)
            enc_pos = self.pos_embedding(torch_enc_xyz, input_range=torch_point_cloud_dims)
        else:
            point_cloud_dims = [
                ttnn.from_torch(t, dtype=ttnn.bfloat16, device=self.device, layout=ttnn.TILE_LAYOUT)
                for t in torch_point_cloud_dims
            ]

            torch_query_xyz, query_embed = self.get_query_embeddings(torch_enc_xyz, point_cloud_dims)
            # query_embed: batch x npoint x channel
            enc_pos = self.pos_embedding(torch_enc_xyz, input_range=point_cloud_dims)

        # decoder expects: batch x npoints x channel
        tgt = ttnn.zeros_like(query_embed, dtype=ttnn.bfloat16)
        box_features = self.decoder(tgt, enc_features, query_pos=query_embed, pos=enc_pos)
        ttnn.deallocate(tgt)
        ttnn.deallocate(enc_features)
        ttnn.deallocate(query_embed)
        ttnn.deallocate(enc_pos)

        (
            cls_logits,
            center_offset,
            size_normalized,
            angle_logits,
            angle_residual_normalized,
            angle_residual,
            num_layers,
            torch_query_xyz,
            torch_point_cloud_dims,
        ) = self.get_box_predictions(torch_query_xyz, torch_point_cloud_dims, box_features)
        return (
            cls_logits,
            center_offset,
            size_normalized,
            angle_logits,
            angle_residual_normalized,
            angle_residual,
            num_layers,
            torch_query_xyz,
            torch_point_cloud_dims,
        )


def build_ttnn_preencoder(args):
    mlp_dims = [3 * int(args.use_color), 64, 128, args.enc_dim]
    preencoder = TtnnPointnetSAModuleVotes(
        radius=0.2,
        nsample=64,
        npoint=args.preenc_npoints,
        mlp=mlp_dims,
        normalize_xyz=True,
        parameters=args.parameters.pre_encoder,
        layer_params=args.parameters.layer_args.pre_encoder,
        device=args.device,
    )
    return preencoder


def build_ttnn_encoder(args):
    if args.enc_type in ["masked"]:
        encoder_layer = TtnnTransformerEncoderLayer
        interim_downsampling = TtnnPointnetSAModuleVotes(
            radius=0.4,
            nsample=32,
            npoint=args.preenc_npoints // 2,
            mlp=[args.enc_dim, 256, 256, args.enc_dim],
            normalize_xyz=True,
            parameters=args.parameters.encoder.interim_downsampling,
            layer_params=args.parameters.layer_args.encoder.interim_downsampling,
            device=args.device,
        )

        masking_radius = [math.pow(x, 2) for x in [0.4, 0.8, 1.2]]
        encoder = TtnnMaskedTransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=args.enc_nlayers,
            interim_downsampling=interim_downsampling,
            masking_radius=masking_radius,
            device=args.device,
            encoder_args=EncoderLayerArgs(
                d_model=args.enc_dim,
                nhead=args.enc_nhead,
                dim_feedforward=args.enc_ffn_dim,
            ),
            parameters=args.parameters.encoder,
        )
    else:
        raise ValueError(f"Unknown encoder type {args.enc_type}")
    return encoder


def build_ttnn_decoder(args):
    decoder = TtnnTransformerDecoder(
        decoder_layer=TtnnTransformerDecoderLayer,
        num_layers=args.dec_nlayers,
        device=args.device,
        return_intermediate=True,
        decoder_args=DecoderLayerArgs(
            d_model=args.dec_dim,
            nhead=args.dec_nhead,
            dim_feedforward=args.dec_ffn_dim,
            normalize_before=True,
        ),
        parameters=args.parameters.decoder,
    )
    return decoder


def build_ttnn_3detr(args, dataset_config):
    pre_encoder = build_ttnn_preencoder(args)
    encoder = build_ttnn_encoder(args)
    decoder = build_ttnn_decoder(args)
    model = TtnnModel3DETR(
        pre_encoder,
        encoder,
        decoder,
        dataset_config,
        encoder_dim=args.enc_dim,
        decoder_dim=args.dec_dim,
        num_queries=args.nqueries,
        parameters=args.parameters,
        device=args.device,
    )
    torch_output_processor = BoxProcessor(dataset_config)
    # output_processor = TtnnBoxProcessor(dataset_config, device=args.device)
    return model, torch_output_processor
