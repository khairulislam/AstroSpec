"""Universal Spectral Tokenizer, a native-grid transformer autoencoder.

Shen et al. 2025, arXiv:2510.17959, "Universal Spectral Tokenization via
Self-Supervised Panchromatic Representation Learning."
"""

import math
from typing import Optional

import torch
from torch import nn

from ..layers import LayerNorm, TransformerBlock, init_by_depth
from ..registry import register_model

__all__ = ["sinusoidal_wavelength_encoding", "ShenSpectralTokenizer", "shen_et_al"]


def sinusoidal_wavelength_encoding(wavelength: torch.Tensor, dim: int) -> torch.Tensor:
    """Log-spaced sinusoidal encoding of per-pixel wavelength, Shen et al. Eq. (1).

    Interleaves sine and cosine per frequency (``encoding[..., 0::2]`` is the
    sine terms, ``[..., 1::2]`` the cosine terms), with angular frequencies
    log-spaced between ``2*pi / 1e6`` and ``2*pi / 0.1`` per Angstrom, wide
    enough to span SDSS through APOGEE while resolving sub-pixel wavelength
    differences.

    Args:
        wavelength: wavelengths in Angstrom, any shape.
        dim: encoding width; must be even.

    Shape:
        ``wavelength (...,)`` -> encoding ``(..., dim)``.
    """
    if dim % 2:
        raise ValueError(f"dim must be even, got {dim}")

    max_period, min_period = 1e6, 0.1
    half = dim // 2
    omega = torch.exp(
        torch.linspace(
            math.log(2 * math.pi / max_period),
            math.log(2 * math.pi / min_period),
            half,
            device=wavelength.device,
            dtype=wavelength.dtype,
        )
    )

    angles = wavelength.unsqueeze(-1) * omega
    encoding = torch.zeros(*wavelength.shape, dim, device=wavelength.device, dtype=wavelength.dtype)
    encoding[..., 0::2] = torch.sin(angles)
    encoding[..., 1::2] = torch.cos(angles)
    return encoding


class ShenSpectralTokenizer(nn.Module):
    """Transformer autoencoder that tokenizes spectra patch by patch, on any native grid.

    Every patch carries its own wavelengths, so sequence position no longer
    means a fixed wavelength and patches from different surveys and pixel
    scales can sit in the same batch. A patch's flux and inverse-variance
    error are projected together, a per-pixel wavelength encoding (see
    :func:`sinusoidal_wavelength_encoding`) is averaged within the patch and
    added, and a transformer encoder produces one token per patch. The
    decoder adds a wavelength embedding to those tokens again and
    reconstructs flux at that grid; passing a different grid than the input's
    lets the same encoder output stand in for the spectrum resampled to
    another instrument's pixel scale, the paper's motivation for adding the
    embedding twice rather than once at the input.

    Consumes ``flux``, ``wavelength``, and optionally ``ivar``, all patched
    (see :class:`astrospec.data.Patchify`) with the same patch size; ``ivar``
    defaults to a uniform weight of one when omitted. Padded patches are
    excluded as attention keys wherever ``valid`` is given, the same
    key-only convention as :class:`astrospec.models.SpecFormer`.

    Pretraining minimizes a measurement-error-weighted Gaussian/Huber
    reconstruction loss over valid pixels; that loss and the training loop
    are not implemented here, see the examples.

    Args:
        patch_size: pixels per patch.
        embed_dim: model width.
        num_enc_layers: transformer encoder blocks.
        num_dec_layers: transformer decoder blocks.
        num_heads: attention heads per block.
        dropout: dropout in the input projection and the blocks.

    Shape:
        ``flux``, ``wavelength``, ``ivar`` all ``(B, T, patch_size)`` ->
        reconstruction ``(B, T, patch_size)``, features ``(B, T, embed_dim)``.
    """

    def __init__(
        self,
        patch_size: int = 32,
        embed_dim: int = 512,
        num_enc_layers: int = 6,
        num_dec_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.patch_size = patch_size
        self.embed_dim = embed_dim

        self.input_proj = nn.Linear(2 * patch_size, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.encoder_blocks = nn.ModuleList(
            TransformerBlock(embedding_dim=embed_dim, num_heads=num_heads, dropout=dropout)
            for _ in range(num_enc_layers)
        )
        self.encoder_norm = LayerNorm(embed_dim, bias=True)

        self.decoder_blocks = nn.ModuleList(
            TransformerBlock(embedding_dim=embed_dim, num_heads=num_heads, dropout=dropout)
            for _ in range(num_dec_layers)
        )
        self.decoder_norm = LayerNorm(embed_dim, bias=True)
        self.head = nn.Linear(embed_dim, patch_size)

        self._reset_parameters(num_enc_layers, num_dec_layers)

    def _reset_parameters(self, num_enc_layers: int, num_dec_layers: int) -> None:
        std = 1 / (self.embed_dim**0.5)
        nn.init.trunc_normal_(self.input_proj.weight, std=std, a=-3 * std, b=3 * std)
        nn.init.zeros_(self.input_proj.bias)

        self.encoder_blocks.apply(lambda m: init_by_depth(m, num_enc_layers))
        self.decoder_blocks.apply(lambda m: init_by_depth(m, num_dec_layers))
        self.head.apply(lambda m: init_by_depth(m, 1 / 2))

    def _wavelength_embedding(self, wavelength: torch.Tensor) -> torch.Tensor:
        return sinusoidal_wavelength_encoding(wavelength, self.embed_dim).mean(dim=2)

    def forward_features(
        self,
        flux: torch.Tensor,
        wavelength: torch.Tensor,
        ivar: Optional[torch.Tensor] = None,
        valid: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode patches into one token each.

        Args:
            flux: ``(B, T, patch_size)`` flux patches.
            wavelength: ``(B, T, patch_size)`` wavelengths of the same
                pixels, in Angstrom.
            ivar: optional ``(B, T, patch_size)`` inverse variance; a
                uniform weight of one is used where omitted.
            valid: optional ``(B, T)`` boolean, ``False`` on padded patches.
        """
        error = torch.ones_like(flux) if ivar is None else ivar.clamp(min=1e-8).rsqrt()
        x = self.dropout(self.input_proj(torch.cat([flux, error], dim=-1)))
        x = x + self._wavelength_embedding(wavelength)

        attn_mask = None if valid is None else valid.bool()[:, None, None, :]
        for block in self.encoder_blocks:
            x = block(x, attn_mask=attn_mask)
        return self.encoder_norm(x)

    def forward(
        self,
        flux: torch.Tensor,
        wavelength: torch.Tensor,
        ivar: Optional[torch.Tensor] = None,
        valid: Optional[torch.Tensor] = None,
        output_wavelength: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Reconstruct flux at ``output_wavelength``, defaulting to the input grid."""
        tokens = self.forward_features(flux, wavelength, ivar=ivar, valid=valid)
        target_wavelength = wavelength if output_wavelength is None else output_wavelength

        x = tokens + self._wavelength_embedding(target_wavelength)
        attn_mask = None if valid is None else valid.bool()[:, None, None, :]
        for block in self.decoder_blocks:
            x = block(x, attn_mask=attn_mask)
        x = self.decoder_norm(x)
        return self.head(x)


@register_model
def shen_et_al(
    patch_size: int = 32,
    embed_dim: int = 512,
    num_enc_layers: int = 6,
    num_dec_layers: int = 6,
    num_heads: int = 8,
    **kwargs,
) -> ShenSpectralTokenizer:
    """Universal Spectral Tokenizer at the paper's configuration."""
    return ShenSpectralTokenizer(
        patch_size=patch_size,
        embed_dim=embed_dim,
        num_enc_layers=num_enc_layers,
        num_dec_layers=num_dec_layers,
        num_heads=num_heads,
        **kwargs,
    )
