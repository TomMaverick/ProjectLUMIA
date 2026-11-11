"""Multi-epoch spatial-temporal tracklet classification network with swappable aggregators."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from LUMIA.models.backbones import ResNetLite

if TYPE_CHECKING:
    from LUMIA.training.config import TrackletModelConfig


class MaskedPoolAggregator(nn.Module):
    """Zero-parameter baseline computing masked temporal mean and max pooling."""

    def __init__(self, spatial_embed_dim: int = 128) -> None:
        super().__init__()
        self.spatial_embed_dim = spatial_embed_dim
        self.output_dim = 2 * spatial_embed_dim

    def forward(
        self,
        tokens: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Aggregate sequence tokens into a fixed-size vector.

        Args:
            tokens: FloatTensor of shape (B, N_max, D_spatial).
            padding_mask: BoolTensor of shape (B, N_max) where True indicates valid frame.

        Returns:
            Concatenated mean and max pooled representation of shape (B, 2 * D_spatial).
        """
        mask = padding_mask.unsqueeze(-1)  # (B, N_max, 1)
        denom = mask.sum(dim=1).clamp(min=1e-6)

        # Masked mean
        z_mean = (tokens * mask.float()).sum(dim=1) / denom

        # Masked max
        masked_tokens = tokens.masked_fill(~mask, -1e9)
        z_max = masked_tokens.max(dim=1).values

        return torch.cat([z_mean, z_max], dim=-1)


class BiGRUAggregator(nn.Module):
    """Bidirectional GRU aggregator with packed sequence processing."""

    def __init__(
        self,
        spatial_embed_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.spatial_embed_dim = spatial_embed_dim
        self.hidden_dim = hidden_dim
        self.output_dim = 2 * hidden_dim

        self.gru = nn.GRU(
            input_size=spatial_embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(
        self,
        tokens: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Process variable-length sequences via packed bidirectional GRU.

        Args:
            tokens: FloatTensor of shape (B, N_max, D_spatial).
            padding_mask: BoolTensor of shape (B, N_max) where True indicates valid frame.

        Returns:
            Aggregated bidirectional hidden states of shape (B, 2 * hidden_dim).
        """
        batch_size = tokens.size(0)
        lengths = padding_mask.sum(dim=1).clamp(min=1).cpu()

        packed = pack_padded_sequence(
            tokens,
            lengths,
            batch_first=True,
            enforce_sorted=False,
        )
        out_packed, _ = self.gru(packed)
        out, _ = pad_packed_sequence(
            out_packed,
            batch_first=True,
            total_length=tokens.size(1),
        )

        batch_idx = torch.arange(batch_size, device=tokens.device)
        last_idx = (lengths.to(tokens.device) - 1).long()

        forward_feat = out[batch_idx, last_idx, : self.hidden_dim]
        backward_feat = out[batch_idx, 0, self.hidden_dim :]

        return torch.cat([forward_feat, backward_feat], dim=-1)


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed 1D temporal sinusoidal positional encodings."""

    def __init__(self, d_model: int, max_len: int = 256) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        pe_sub: torch.Tensor = self.pe[:, :seq_len]  # type: ignore[index]
        return x + pe_sub


class TransformerAggregator(nn.Module):
    """Transformer Encoder aggregator with 1D temporal positional encodings."""

    def __init__(
        self,
        spatial_embed_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_len: int = 256,
    ) -> None:
        super().__init__()
        self.spatial_embed_dim = spatial_embed_dim
        self.hidden_dim = hidden_dim
        self.output_dim = 2 * hidden_dim

        self.pos_encoder = SinusoidalPositionalEncoding(spatial_embed_dim, max_len=max_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=spatial_embed_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.proj = nn.Linear(2 * spatial_embed_dim, self.output_dim)

    def forward(
        self,
        tokens: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Transform temporal tokens and aggregate valid frames.

        Args:
            tokens: FloatTensor of shape (B, N_max, D_spatial).
            padding_mask: BoolTensor of shape (B, N_max) where True indicates valid frame.

        Returns:
            Projected aggregated representation of shape (B, 2 * hidden_dim).
        """
        # In PyTorch TransformerEncoder, src_key_padding_mask is True for masked positions
        src_key_padding_mask = ~padding_mask

        x = self.pos_encoder(tokens)
        transformed = self.encoder(x, src_key_padding_mask=src_key_padding_mask)

        mask = padding_mask.unsqueeze(-1)
        denom = mask.sum(dim=1).clamp(min=1e-6)

        z_mean = (transformed * mask.float()).sum(dim=1) / denom
        masked_trans = transformed.masked_fill(~mask, -1e9)
        z_max = masked_trans.max(dim=1).values

        pooled = torch.cat([z_mean, z_max], dim=-1)
        out: torch.Tensor = self.proj(pooled)
        return out


class TrackletClassifierNet(nn.Module):
    """End-to-end multi-epoch tracklet sequence classifier."""

    def __init__(self, config: TrackletModelConfig) -> None:
        super().__init__()
        self.config = config

        # 1. Spatial Backbone
        if config.spatial_backbone == "resnet_lite":
            self.spatial_backbone: nn.Module = ResNetLite(
                in_channels=3,
                spatial_embed_dim=config.spatial_embed_dim,
                norm_layer=config.norm_layer,
            )
        else:
            raise ValueError(f"Unsupported spatial backbone '{config.spatial_backbone}'.")

        # 2. Temporal Aggregator
        agg_type = config.temporal_aggregator
        if agg_type == "masked_pool":
            self.temporal_aggregator: nn.Module = MaskedPoolAggregator(
                spatial_embed_dim=config.spatial_embed_dim,
            )
        elif agg_type == "gru":
            self.temporal_aggregator = BiGRUAggregator(
                spatial_embed_dim=config.spatial_embed_dim,
                hidden_dim=config.temporal_hidden_dim,
                num_layers=config.temporal_layers,
                dropout=config.dropout_rate,
            )
        elif agg_type == "transformer":
            self.temporal_aggregator = TransformerAggregator(
                spatial_embed_dim=config.spatial_embed_dim,
                hidden_dim=config.temporal_hidden_dim,
                num_layers=config.temporal_layers,
                dropout=config.dropout_rate,
            )
        else:
            raise ValueError(f"Unsupported temporal aggregator '{agg_type}'.")

        # 3. Classification Head: MLP mapping aggregated features to 4 class logits
        agg_dim = getattr(self.temporal_aggregator, "output_dim", 256)
        head_dim = 128
        self.head = nn.Sequential(
            nn.Dropout(p=config.dropout_rate),
            nn.Linear(agg_dim, head_dim),
            nn.ReLU(inplace=True),
            nn.Linear(head_dim, config.num_classes),
        )

    def forward(
        self,
        tracklets: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass over variable-length sequence batch.

        Args:
            tracklets: FloatTensor of shape (B, N_max, 3, 128, 128).
            padding_mask: BoolTensor of shape (B, N_max) where True indicates valid frame.

        Returns:
            Logits FloatTensor of shape (B, 4).
        """
        if tracklets.ndim != 5:
            raise ValueError(
                f"Expected 5D tracklet batch (B, N_max, C, H, W), got {tuple(tracklets.shape)}."
            )
        b, n_max, c, h, w = tracklets.shape

        # Time-distributed spatial convolution pass
        flat_x = tracklets.reshape(b * n_max, c, h, w)

        # Robust median +- 20*MAD outlier clipping on difference frames (channel index 2)
        if c >= 3:
            diff = flat_x[:, 2:3, :, :]
            diff_flat = diff.flatten(start_dim=-2)
            med = diff_flat.median(dim=-1, keepdim=True).values.unsqueeze(-1)
            mad = (
                (diff_flat - med.squeeze(-1))
                .abs()
                .median(dim=-1, keepdim=True)
                .values.unsqueeze(-1)
            )
            mad = torch.clamp(mad, min=1e-5)
            diff_clipped = torch.clamp(diff, min=med - 20.0 * mad, max=med + 20.0 * mad)
            flat_x = torch.cat([flat_x[:, :2, :, :], diff_clipped, flat_x[:, 3:, :, :]], dim=1)

        flat_spatial = self.spatial_backbone(flat_x)
        tokens = flat_spatial.reshape(b, n_max, -1)

        # Explicit zero-masking of spatial tokens on padded frames
        tokens = tokens * padding_mask.unsqueeze(-1).float()

        # Temporal sequence aggregation across valid epochs
        agg_features = self.temporal_aggregator(tokens, padding_mask)

        # Classification MLP head
        logits: torch.Tensor = self.head(agg_features)
        return logits
