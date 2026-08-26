"""Cross-attention pooling head, the standard downstream head across OmniSpectrum.

``models/downstream_heads.py``'s cached-embedding regression/classification
heads and its fine-tuning operator all attach the same learned-query
cross-attention pool, followed by an MLP, to whichever encoder's token
sequence they are given. Every sequence encoder in this library
(:class:`astrospec.models.SpecFormer`, :class:`astrospec.models.AstroPT`,
:class:`astrospec.models.SpecPTEncoder`,
:class:`astrospec.models.ShenSpectralTokenizer`) is otherwise unusable for a
downstream task without something like it.
"""

from typing import Optional

import torch
from torch import nn

__all__ = ["CrossAttentionPool", "CrossAttentionHead"]


class CrossAttentionPool(nn.Module):
    """A single learned query token that cross-attends into a token sequence.

    Reduces a variable-length sequence of encoder tokens to one fixed-size
    vector, regardless of how many valid tokens it contains.

    Args:
        embed_dim: width of both the query and the sequence tokens.
        num_heads: attention heads.
        dropout: applied to the attention output.

    Shape:
        ``tokens`` ``(B, T, embed_dim)`` -> pooled ``(B, embed_dim)``.
    """

    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()

        self.query = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None):
        """
        Args:
            tokens: ``(B, T, embed_dim)``.
            key_padding_mask: optional ``(B, T)`` boolean, ``True`` on
                positions to ignore. This is ``nn.MultiheadAttention``'s own
                convention, the inverse of the ``valid`` argument the
                encoders in this library take.
        """
        query = self.query.expand(tokens.shape[0], -1, -1)
        pooled, _ = self.attention(query, tokens, tokens, key_padding_mask=key_padding_mask)
        return self.norm(self.dropout(pooled)).squeeze(1)


class CrossAttentionHead(nn.Module):
    """:class:`CrossAttentionPool` followed by an MLP.

    Args:
        embed_dim: width of the encoder tokens this head pools.
        num_outputs: size of the output vector: classes for classification,
            targets for regression.
        num_heads: attention heads in the pooling step.
        dropout: applied in the pooling step and the MLP.

    Shape:
        ``tokens`` ``(B, T, embed_dim)`` -> output ``(B, num_outputs)``.
    """

    def __init__(
        self,
        embed_dim: int,
        num_outputs: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.pool = CrossAttentionPool(embed_dim, num_heads=num_heads, dropout=dropout)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, num_outputs),
        )

    def forward(self, tokens: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None):
        return self.mlp(self.pool(tokens, key_padding_mask=key_padding_mask))
