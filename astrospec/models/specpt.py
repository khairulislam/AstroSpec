"""SpecPT, a conv + transformer autoencoder for spectrum reconstruction, with a redshift head.

Pattnaik et al. 2025, ApJ 988:139. https://doi.org/10.3847/1538-4357/ade053
"""

import torch
import torch.nn.functional as F
from torch import nn

from ..layers import CrossAttentionBlock, LayerNorm, SelfAttention, TransformerBlock
from ..registry import register_model

__all__ = [
    "SpecPTEncoder",
    "SpecPTAutoencoder",
    "SpecPTRedshiftHead",
    "SpecPTRedshift",
    "specpt",
    "specpt_redshift",
]


class _ConvBlock(nn.Sequential):
    """Conv1d, batch norm, ReLU, and a stride-2 max pool."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )


class _ConvEncoder(nn.Module):
    """Three conv + pool stages, downsampling the spectrum 8x into 256 channels."""

    def __init__(self):
        super().__init__()
        self.stage1 = _ConvBlock(1, 64, kernel_size=41)
        self.stage2 = _ConvBlock(64, 128, kernel_size=21)
        self.stage3 = _ConvBlock(128, 256, kernel_size=11)

    def forward(self, flux):
        x = self.stage1(flux.unsqueeze(1))
        x = self.stage2(x)
        x = self.stage3(x)
        return x.transpose(1, 2)  # (B, T, 256)


class _ConvDecoder(nn.Module):
    """Inverts :class:`_ConvEncoder`, trimmed or padded to ``output_len``."""

    def __init__(self, output_len: int):
        super().__init__()
        self.output_len = output_len
        self.stage1 = nn.Sequential(
            nn.ConvTranspose1d(256, 128, kernel_size=11, stride=2, padding=5, output_padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )
        self.stage2 = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=21, stride=2, padding=10, output_padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )
        self.stage3 = nn.ConvTranspose1d(64, 1, kernel_size=41, stride=2, padding=20, output_padding=1)

    def forward(self, x):
        x = self.stage1(x.transpose(1, 2))
        x = self.stage2(x)
        x = self.stage3(x).squeeze(1)
        if x.shape[1] > self.output_len:
            x = x[:, : self.output_len]
        elif x.shape[1] < self.output_len:
            x = F.pad(x, (0, self.output_len - x.shape[1]))
        return x


class _ResidualMLPBlock(nn.Module):
    """Linear, SiLU, dropout, linear, then a post-norm residual add."""

    def __init__(self, dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        residual = x
        x = self.fc2(self.drop(F.silu(self.fc1(x))))
        return self.norm(x + residual)


class SpecPTEncoder(nn.Module):
    """Conv feature extractor, linear projection, and a transformer encoder.

    Consumes ``flux`` only, on the fixed grid the conv stages were sized for:
    the OmniSpectrum configuration resamples DESI and SDSS spectra onto a
    common 0.8 A/pixel grid from 3600 to 9824 A, 7780 pixels.

    Args:
        embed_dim: model width.
        num_layers: transformer encoder blocks.
        num_heads: attention heads per block.
        dropout: dropout in the projection and the blocks.

    Shape:
        ``flux`` ``(B, N)`` -> features ``(B, N // 8, embed_dim)``.
    """

    def __init__(
        self,
        embed_dim: int = 512,
        num_layers: int = 3,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.conv = _ConvEncoder()
        self.projection = nn.Linear(256, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            TransformerBlock(embedding_dim=embed_dim, num_heads=num_heads, dropout=dropout)
            for _ in range(num_layers)
        )
        self.norm = LayerNorm(embed_dim)

    def forward(self, flux):
        x = self.dropout(self.projection(self.conv(flux)))
        for block in self.blocks:
            x = block(x)
        return self.norm(x)


class SpecPTAutoencoder(nn.Module):
    """Encoder-decoder that reconstructs a spectrum from itself.

    A :class:`SpecPTEncoder` feeds a transformer decoder that cross-attends
    from a learned bank of positional queries into the encoded sequence, and
    a mirrored conv stack upsamples the result back to the input length.
    Pretraining minimizes the paper's normalized-MAD loss between input and
    reconstruction; that loss and the training loop are not implemented here,
    see the examples. The pretrained :attr:`encoder` is what
    :class:`SpecPTRedshift` builds on.

    Args:
        input_len: spectrum length, also sizing the decoder query bank.
        embed_dim: model width.
        num_enc_layers: transformer encoder blocks.
        num_dec_layers: transformer decoder blocks.
        num_heads: attention heads per block.
        dropout: dropout throughout the encoder and decoder.

    Shape:
        ``flux`` ``(B, input_len)`` -> reconstruction ``(B, input_len)``.
    """

    def __init__(
        self,
        input_len: int = 7780,
        embed_dim: int = 512,
        num_enc_layers: int = 3,
        num_dec_layers: int = 3,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_len = input_len
        self.encoder = SpecPTEncoder(
            embed_dim=embed_dim, num_layers=num_enc_layers, num_heads=num_heads, dropout=dropout
        )

        max_len = input_len // 8
        self.decoder_queries = nn.Parameter(torch.zeros(1, max_len, embed_dim))
        nn.init.trunc_normal_(self.decoder_queries, std=0.02)

        self.decoder_blocks = nn.ModuleList(
            CrossAttentionBlock(embedding_dim=embed_dim, num_heads=num_heads, dropout=dropout)
            for _ in range(num_dec_layers)
        )
        self.decoder_norm = LayerNorm(embed_dim)
        self.decoder_projection = nn.Linear(embed_dim, 256)
        self.conv_decoder = _ConvDecoder(output_len=input_len)

    def forward(self, flux):
        N = flux.shape[1]
        memory = self.encoder(flux)
        T = memory.shape[1]
        queries = self.decoder_queries[:, :T, :].expand(memory.shape[0], -1, -1)

        x = queries
        for block in self.decoder_blocks:
            x = block(x, memory)
        x = self.decoder_norm(x)

        recon = self.conv_decoder(self.decoder_projection(x))
        if recon.shape[1] > N:
            recon = recon[:, :N]
        elif recon.shape[1] < N:
            recon = F.pad(recon, (0, N - recon.shape[1]))
        return recon


class SpecPTRedshiftHead(nn.Module):
    """Redshift regression head over a :class:`SpecPTEncoder` sequence.

    Self-attention over the sequence with a residual skip, average-pooled
    over positions, then a stack of residual MLP blocks and a scalar head
    with a softplus so the predicted redshift is non-negative.

    Args:
        embed_dim: width of the encoder features this head consumes.
        num_heads: attention heads in the pooling self-attention.
        num_blocks: residual MLP blocks.
        hidden_dim: width of each residual MLP block's expansion.
        dropout: dropout in the attention and the MLP blocks.

    Shape:
        features ``(B, T, embed_dim)`` -> redshift ``(B,)``.
    """

    def __init__(
        self,
        embed_dim: int = 512,
        num_heads: int = 8,
        num_blocks: int = 5,
        hidden_dim: int = 412,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.attention = SelfAttention(embed_dim, num_heads, dropout=dropout)
        self.attention_norm = nn.LayerNorm(embed_dim)
        self.mlp_blocks = nn.Sequential(
            *[_ResidualMLPBlock(embed_dim, hidden_dim, dropout) for _ in range(num_blocks)]
        )
        self.output = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
            nn.Softplus(),
        )

    def forward(self, features):
        features = self.attention_norm(features + self.attention(features))
        pooled = self.mlp_blocks(features.mean(dim=1))
        return self.output(pooled).squeeze(-1)


class SpecPTRedshift(nn.Module):
    """A :class:`SpecPTEncoder` and :class:`SpecPTRedshiftHead`, for redshift regression.

    Pattnaik et al. pretrain the encoder inside a :class:`SpecPTAutoencoder`
    and load its weights here before optionally freezing them; loading a
    checkpoint into :attr:`encoder` is left to the caller, since it is a
    weight-transfer step rather than something this constructor should do.

    Args:
        embed_dim: model width, shared by the encoder and the head.
        num_enc_layers: transformer encoder blocks.
        num_heads: attention heads per block, in the encoder and the head.
        num_mlp_blocks: residual MLP blocks in the head.
        dropout: dropout in the encoder; the head always uses 0.2, as in the paper.
        freeze_encoder: if True, the encoder's parameters do not require grad.

    Shape:
        ``flux`` ``(B, N)`` -> redshift ``(B,)``.
    """

    def __init__(
        self,
        embed_dim: int = 512,
        num_enc_layers: int = 3,
        num_heads: int = 8,
        num_mlp_blocks: int = 5,
        dropout: float = 0.1,
        freeze_encoder: bool = True,
    ):
        super().__init__()

        self.encoder = SpecPTEncoder(
            embed_dim=embed_dim, num_layers=num_enc_layers, num_heads=num_heads, dropout=dropout
        )
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad_(False)

        self.head = SpecPTRedshiftHead(
            embed_dim=embed_dim, num_heads=num_heads, num_blocks=num_mlp_blocks, dropout=0.2
        )

    def forward(self, flux):
        return self.head(self.encoder(flux))


@register_model
def specpt(
    input_len: int = 7780,
    embed_dim: int = 512,
    num_enc_layers: int = 3,
    num_dec_layers: int = 3,
    num_heads: int = 8,
    **kwargs,
) -> SpecPTAutoencoder:
    """SpecPT autoencoder at the OmniSpectrum training configuration."""
    return SpecPTAutoencoder(
        input_len=input_len,
        embed_dim=embed_dim,
        num_enc_layers=num_enc_layers,
        num_dec_layers=num_dec_layers,
        num_heads=num_heads,
        **kwargs,
    )


@register_model
def specpt_redshift(
    embed_dim: int = 512,
    num_enc_layers: int = 3,
    num_heads: int = 8,
    num_mlp_blocks: int = 5,
    **kwargs,
) -> SpecPTRedshift:
    """SpecPT redshift head at the OmniSpectrum training configuration."""
    return SpecPTRedshift(
        embed_dim=embed_dim,
        num_enc_layers=num_enc_layers,
        num_heads=num_heads,
        num_mlp_blocks=num_mlp_blocks,
        **kwargs,
    )
