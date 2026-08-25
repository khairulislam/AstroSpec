"""AstroPT, a GPT-style causal transformer over spectral patches.

Smith et al. 2024. https://arxiv.org/abs/2405.14930
Reference implementation: https://github.com/Smith42/astroPT
"""

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from ..layers import LayerNorm, TransformerBlock, init_by_depth
from ..registry import register_model

__all__ = ["AstroPT", "AimTokenizer", "astropt"]


class AimTokenizer(nn.Module):
    """Linear, GELU, linear: AstroPT's default ``aim`` projection.

    Used for the patch embedding, the wavelength position embedding, and the
    output head alike.
    """

    def __init__(self, in_size: int, out_size: int, hidden_size: int, bias: bool = False):
        super().__init__()
        self.c_fc = nn.Linear(in_size, hidden_size, bias=bias)
        self.c_proj = nn.Linear(hidden_size, out_size, bias=bias)

    def forward(self, x):
        return self.c_proj(F.gelu(self.c_fc(x), approximate="tanh"))


class AstroPT(nn.Module):
    """Causal decoder-only transformer, pretrained by next-patch prediction.

    Patches are embedded with an :class:`AimTokenizer` and summed with a second
    tokenizer applied to the wavelengths of the same patch, so position is
    continuous and physical rather than a rank-indexed lookup. Causal blocks
    then let each patch see only its predecessors, and the head predicts the
    following patch from each hidden state.

    Consumes ``flux`` and ``wavelength``, both patched (see
    :class:`astrospec.data.Patchify`); ``ivar``, ``mask``, and ``lsf_sigma`` are
    ignored. Because position is read from wavelength, spectra on different
    grids are directly comparable.

    Training is not implemented here. The pretraining objective is a Huber loss
    between ``forward(...)[:, :-1]`` and ``patches[:, 1:]`` over patch pairs
    that are both valid; see the examples.

    Args:
        input_dim: pixels per patch.
        embed_dim: model width.
        num_layers: number of causal transformer blocks.
        num_heads: attention heads per block.
        dropout: AstroPT pretrains with none.
        bias: AstroPT uses no biases in its linear layers or layer norms.
        wavelength_range: wavelengths are min-max normalized onto [0, 1] with
            these bounds before the position tokenizer, as in AstroPT.

    Shape:
        ``patches`` and ``wavelength`` ``(B, T, input_dim)`` -> predictions
        ``(B, T, input_dim)``, features ``(B, T, embed_dim)``.
    """

    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.0,
        bias: bool = False,
        wavelength_range: Tuple[float, float] = (3000.0, 10000.0),
    ):
        super().__init__()

        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.wavelength_range = wavelength_range

        self.data_embed = AimTokenizer(input_dim, embed_dim, 4 * embed_dim, bias)
        self.pos_embed = AimTokenizer(input_dim, embed_dim, 4 * embed_dim, bias)
        self.blocks = nn.ModuleList(
            TransformerBlock(
                embedding_dim=embed_dim,
                num_heads=num_heads,
                causal=True,
                dropout=dropout,
                bias=bias,
            )
            for _ in range(num_layers)
        )
        self.norm = LayerNorm(embed_dim, bias=bias)
        self.head = AimTokenizer(embed_dim, input_dim, 4 * embed_dim, bias)

        self._reset_parameters(num_layers)

    def _reset_parameters(self, num_layers: int) -> None:
        for tokenizer in (self.data_embed, self.pos_embed, self.head):
            tokenizer.apply(lambda m: init_by_depth(m, 1 / 2))
        self.blocks.apply(lambda m: init_by_depth(m, num_layers))

    def forward_features(self, patches, wavelength, valid: Optional[torch.Tensor] = None):
        """Encode patches into one causal hidden state each.

        Args:
            patches: ``(B, T, input_dim)`` flux patches.
            wavelength: ``(B, T, input_dim)`` wavelengths of the same pixels, in A.
            valid: optional ``(B, T)`` boolean, ``False`` on padded patches.
        """
        low, high = self.wavelength_range
        x = self.data_embed(patches) + self.pos_embed((wavelength - low) / (high - low))

        # a query attends to valid keys only; the blocks add causality on top
        attn_mask = None if valid is None else valid.bool()[:, None, None, :]
        for block in self.blocks:
            x = block(x, attn_mask=attn_mask)
        return self.norm(x)

    def forward(self, patches, wavelength, valid: Optional[torch.Tensor] = None):
        """Predict the next patch from each position."""
        return self.head(self.forward_features(patches, wavelength, valid=valid))


@register_model
def astropt(
    input_dim: int = 32,
    embed_dim: int = 512,
    num_layers: int = 8,
    num_heads: int = 8,
    **kwargs,
) -> AstroPT:
    """AstroPT at the size of the OmniSpectrum spectra checkpoint."""
    return AstroPT(
        input_dim=input_dim,
        embed_dim=embed_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        **kwargs,
    )
