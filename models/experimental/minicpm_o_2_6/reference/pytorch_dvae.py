# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
PyTorch reference implementation of DVAE for PCC validation.

Simplified implementation focusing on encoder/decoder architecture.
"""

import torch
import torch.nn as nn


class PyTorchDVAE(nn.Module):
    """
    Simplified PyTorch reference implementation of DVAE.
    """

    def __init__(
        self,
        num_encoder_layers: int = 12,  # Production: 12 layers
        num_decoder_layers: int = 12,  # Production: 12 layers
        hidden_dim: int = 256,
        num_mel_bins: int = 100,
        bn_dim: int = 128,  # Production: 128
        enable_gfsq: bool = True,  # Enable/disable GFSQ quantization
    ):
        """
        PyTorch reference DVAE with production configuration.

        Production Configuration (from MiniCPM-o-2_6):
        - Encoder: 12 ConvNeXt blocks, hidden_dim=256, bn_dim=128
        - Decoder: 12 ConvNeXt blocks, hidden_dim=256, bn_dim=128

        Args:
            num_encoder_layers: Number of encoder ConvNeXt blocks (default 12 for production)
            num_decoder_layers: Number of decoder ConvNeXt blocks (default 12 for production)
            hidden_dim: Hidden dimension (default 256)
            num_mel_bins: Number of mel bins (default 100)
            bn_dim: Bottleneck dimension (default 128)
            enable_gfsq: Enable GFSQ quantization (default True)
        """
        super().__init__()
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.hidden_dim = hidden_dim
        self.num_mel_bins = num_mel_bins
        self.enable_gfsq = enable_gfsq
        self.bn_dim = bn_dim

        # Coefficient
        self.coef = nn.Parameter(torch.randn(1, num_mel_bins, 1))

        # Encoder downsampling (1D conv)
        self.encoder_downsample = nn.Sequential(
            nn.Conv1d(num_mel_bins, 512, 3, padding=1),  # 1D conv
            nn.GELU(),  # Production: GELU instead of ReLU
            nn.Conv1d(512, 512, 4, stride=2, padding=1),  # 1D conv with stride
            nn.GELU(),  # Production: GELU instead of ReLU
        )

        # Encoder input (Production: bn_dim=128)
        self.encoder_input = nn.Sequential(
            nn.Conv1d(512, bn_dim, 3, padding=1),
            nn.GELU(),  # Production: GELU instead of ReLU
            nn.Conv1d(bn_dim, hidden_dim, 3, padding=1),
            # nn.GELU(),  # Production: GELU instead of ReLU
        )

        # Encoder blocks (1D ConvNeXt)
        self.encoder_blocks = nn.ModuleList([ConvNeXtBlock1D(hidden_dim) for _ in range(num_encoder_layers)])

        # Encoder output
        self.encoder_output = nn.Conv1d(hidden_dim, 1024, 1)  # 1x1 conv

        # Decoder input (Production: decoder processes 1024 channels from encoder)
        self.decoder_input = nn.Sequential(
            nn.Conv1d(512, bn_dim, 3, padding=1),  # Production: 512 input channels from encoder
            nn.GELU(),  # Production: GELU instead of ReLU
            nn.Conv1d(bn_dim, hidden_dim, 3, padding=1),
            # nn.GELU(),  # Production: GELU instead of ReLU
        )

        # Decoder blocks
        self.decoder_blocks = nn.ModuleList([ConvNeXtBlock1D(hidden_dim) for _ in range(num_decoder_layers)])

        # Decoder projection: hidden_dim -> 512 channels (NEW layer)
        self.decoder_proj = nn.Conv1d(hidden_dim, 512, 1)  # 1x1 conv

        # Decoder output (Production: 512 -> num_mel_bins)
        self.decoder_output = nn.Conv1d(512, num_mel_bins, 3, padding=1)

    def forward(self, mel_spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            mel_spectrogram: [batch_size, num_mel_bins, time_steps]

        Returns:
            torch.Tensor: Reconstructed mel spectrogram
        """
        # Input is already in [B, C, T] format for Conv1d
        # Encoder
        encoded = self._encode(mel_spectrogram)

        # Apply GFSQ quantization (or bypass if disabled)
        if self.enable_gfsq:
            # Apply GFSQ quantization (simplified - pass through for now)
            quantized = encoded
        else:
            # Bypass quantization - pass through unchanged
            quantized = encoded

        print("quantized.shape : ", quantized.shape)

        quantized = (
            quantized.view(
                (1, quantized.size(1) // 2, 2, quantized.size(2)),
            ).flatten(2)
            # .permute(0, 2, 1)
        )

        # Decoder
        reconstructed = self._decode(quantized)

        return reconstructed

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encoder forward pass.
        Input x: [batch, num_mel_bins, time_steps] (NCT format for Conv1d)
        """
        # Skip coefficient for testing basic conv operations
        # coef_expanded = self.coef.unsqueeze(-1)  # [1, num_mel_bins, 1]
        # x = x * coef_expanded

        # Downsampling (NCT format: [B, C, T])
        x = self.encoder_downsample(x)

        # Input processing (NCT format)
        x = self.encoder_input(x)

        # ConvNeXt blocks (PRODUCTION: 12 blocks enabled)
        # Convert to [B, T, C] for ConvNeXt blocks
        x = x.permute(0, 2, 1)  # [B, T, C]

        for block in self.encoder_blocks:
            x = block(x)

        # Convert back to [B, C, T] for encoder_output
        x = x.permute(0, 2, 1)  # [B, C, T]

        # Output
        x = self.encoder_output(x)

        return x

    def _decode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decoder forward pass.
        Production: processes 1024-channel features from encoder, applies 12 ConvNeXt blocks
        Input x: [batch, 1024, time_steps] (encoder output, NCT format)
        """
        # Input processing (1024 -> bn_dim -> hidden_dim, NCT format)
        x = self.decoder_input(x)

        # ConvNeXt blocks (PRODUCTION: 12 blocks enabled)
        # Convert to [B, T, C] for ConvNeXt blocks
        x = x.permute(0, 2, 1)  # [B, T, C]

        for block in self.decoder_blocks:
            x = block(x)

        # Convert back to [B, C, T] for decoder output
        x = x.permute(0, 2, 1)  # [B, C, T]

        # Decoder projection: hidden_dim -> 512 channels
        x = self.decoder_proj(x)

        # Output (512 -> num_mel_bins, NCT format)
        x = self.decoder_output(x)

        return x


class ConvNeXtBlock1D(nn.Module):
    """
    Simplified ConvNeXt block for 1D convolutions.
    Adapted for mel spectrogram processing: [batch, time_steps, channels]
    """

    def __init__(self, dim: int):
        super().__init__()
        # Depthwise 1D conv: groups=dim for depthwise
        self.dwconv = nn.Conv1d(dim, dim, 7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: [batch_size, time_steps, dim] (NTC format)

        Returns:
            torch.Tensor: [batch_size, time_steps, dim] (NTC format)
        """
        residual = x

        # Depthwise conv: Convert to [B, C, T] for Conv1d
        x_nct = x.permute(0, 2, 1)  # [B, C, T]
        x_nct = self.dwconv(x_nct)  # [B, C, T]

        # Convert back to [B, T, C] for LayerNorm
        x = x_nct.permute(0, 2, 1)  # [B, T, C]

        # LayerNorm and pointwise convs
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)

        # Residual
        x = x + residual

        return x
