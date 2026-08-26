"""SpecFormer, the transformer spectrum encoder of AstroCLIP.

Parker et al. 2024, MNRAS 531:4990. https://doi.org/10.1093/mnras/stae1450
"""

import math
from typing import Optional

import torch
from torch import nn

from ..layers import LayerNorm, TransformerBlock, init_by_depth
from ..registry import register_model

__all__ = ["SpecFormer", "specformer"]


class SpecFormer(nn.Module):
    """Transformer over flux patches, pretrained by masked reconstruction.

    Patches enter through a linear embedding, a learned embedding of the patch
    index supplies position, and a stack of pre-norm transformer blocks produces
    one token per patch. A linear head reconstructs the patch from its token,
    which is the masked-modelling objective the encoder is pretrained with; the
    tokens themselves are the representation used downstream.

    Consumes ``flux`` only, already patched (see :class:`astrospec.data.Patchify`);
    ``wavelength``, ``ivar``, ``mask``, and ``lsf_sigma`` are ignored. Position
    comes from the patch index, not from ``wavelength``, so patch every
    spectrum in a dataset on the same grid, or the position embedding means a
    different thing from one spectrum to the next.

    Args:
        input_dim: width of one patch. AstroCLIP uses 22: a 20-pixel patch plus
            the mean and standard deviation of that patch.
        embed_dim: model width.
        num_layers: number of transformer blocks.
        num_heads: attention heads per block.
        max_len: longest patch sequence, the size of the position embedding.
        dropout: dropout in the embeddings and the blocks.

    Shape:
        input ``(B, T, input_dim)`` -> reconstructions ``(B, T, input_dim)``,
        features ``(B, T, embed_dim)``.
    """

    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        num_layers: int,
        num_heads: int,
        max_len: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.max_len = max_len

        self.data_embed = nn.Linear(input_dim, embed_dim)
        self.position_embed = nn.Embedding(max_len, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            TransformerBlock(
                embedding_dim=embed_dim,
                num_heads=num_heads,
                causal=False,
                dropout=dropout,
                bias=True,
            )
            for _ in range(num_layers)
        )
        self.final_layernorm = LayerNorm(embed_dim, bias=True)
        self.head = nn.Linear(embed_dim, input_dim, bias=True)

        self._reset_parameters(num_layers)

    def _reset_parameters(self, num_layers: int) -> None:
        # embeddings are not scaled by depth
        std = 1 / math.sqrt(self.embed_dim)
        for embed in (self.data_embed, self.position_embed):
            nn.init.trunc_normal_(embed.weight, std=std, a=-3 * std, b=3 * std)

        self.blocks.apply(lambda m: init_by_depth(m, num_layers))
        self.head.apply(lambda m: init_by_depth(m, 1 / 2))

    def forward_features(self, flux, valid: Optional[torch.Tensor] = None):
        """Encode patches into one token each, without the reconstruction head.

        Args:
            flux: ``(B, T, input_dim)``, already patched.
            valid: optional ``(B, T)`` boolean, ``False`` on padded patches, as
                returned by :class:`astrospec.data.Patchify`. Attention skips
                padded patches so they cannot influence real tokens.
        """
        t = flux.shape[1]
        if t > self.max_len:
            raise ValueError(f"sequence of {t} patches exceeds max_len={self.max_len}")

        pos = torch.arange(t, dtype=torch.long, device=flux.device)
        x = self.dropout(self.data_embed(flux) + self.position_embed(pos))

        attn_mask = None
        if valid is not None:
            # (B, 1, 1, T): every query may attend to the valid keys only
            attn_mask = valid.bool()[:, None, None, :]

        for block in self.blocks:
            x = block(x, attn_mask=attn_mask)
        return self.final_layernorm(x)

    def forward(self, flux, valid: Optional[torch.Tensor] = None):
        """Reconstruct each patch from its token."""
        return self.head(self.forward_features(flux, valid=valid))


@register_model
def specformer(
    input_dim: int = 22,
    embed_dim: int = 768,
    num_layers: int = 6,
    num_heads: int = 6,
    max_len: int = 800,
    **kwargs,
) -> SpecFormer:
    """SpecFormer with the AstroCLIP configuration."""
    return SpecFormer(
        input_dim=input_dim,
        embed_dim=embed_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        max_len=max_len,
        **kwargs,
    )
