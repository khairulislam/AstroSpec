"""SpectrumEncoder, the CNN + dot-product attention encoder used by spender.

Melchior et al. 2023, AJ 166:74. https://doi.org/10.3847/1538-3881/ace0ff
after Serra et al. 2018. https://arxiv.org/abs/1805.03908
"""

from typing import Optional, Sequence

import torch
from torch import nn

from ..layers import MLP
from ..registry import register_model

__all__ = ["SpectrumEncoder", "spectrum_encoder"]


class SpectrumEncoder(nn.Module):
    """Convolutional encoder with a softmax attention pooling and an MLP.

    The final convolution splits its channels in half into attention values and
    keys; the keys are softmaxed over the pixel axis and used to pool the values
    into one vector per spectrum, which an MLP compresses to ``n_latent``.
    Attention pooling is what makes the encoder length-agnostic: spectra of
    different lengths give the same latent shape.

    Consumes ``flux`` only; ``wavelength``, ``ivar``, ``mask``, and
    ``lsf_sigma`` are ignored.

    Args:
        n_latent: latent dimension.
        filters: output channels of each convolution. The last must be even,
            since it is split into values and keys.
        sizes: kernel width of each convolution; also the pooling width between
            them. Must match ``filters`` in length.
        n_hidden: hidden widths of the MLP.
        act: MLP activations; defaults to ``PReLU`` per hidden layer.
        dropout: dropout rate in the convolutions and the MLP.

    The defaults are the lightweight variant used as a baseline in
    OmniSpectrum. spender's published encoder is ``filters=(128, 256, 512)``,
    ``sizes=(5, 11, 21)``, ``n_hidden=(128, 64, 32)``.

    Shape:
        input ``(B, L)`` or ``(B, 1, L)`` -> output ``(B, n_latent)``.
    """

    def __init__(
        self,
        n_latent: int,
        filters: Sequence[int] = (8, 16, 16, 32),
        sizes: Sequence[int] = (5, 10, 20, 40),
        n_hidden: Sequence[int] = (32, 32),
        act: Optional[Sequence[nn.Module]] = None,
        dropout: float = 0.0,
    ):
        super().__init__()

        if len(filters) != len(sizes):
            raise ValueError(
                f"filters and sizes must be the same length, got {len(filters)} and {len(sizes)}"
            )
        if filters[-1] % 2:
            raise ValueError(f"the last filter count must be even, got {filters[-1]}")

        self.n_latent = n_latent
        self.n_feature = filters[-1] // 2

        self.convs = nn.ModuleList(
            nn.Sequential(
                nn.Conv1d(1 if i == 0 else filters[i - 1], f, kernel_size=s, padding=s // 2),
                nn.InstanceNorm1d(f),
                nn.PReLU(f),
                nn.Dropout(dropout),
            )
            for i, (f, s) in enumerate(zip(filters, sizes))
        )
        # one pool between consecutive convolutions, none after the last
        self.pools = nn.ModuleList(
            nn.MaxPool1d(s, padding=s // 2) for s in sizes[:-1]
        )
        self.softmax = nn.Softmax(dim=-1)

        if act is None:
            # identity last so the latents stay centred on zero
            act = [nn.PReLU(n) for n in n_hidden] + [nn.Identity()]
        self.mlp = MLP(self.n_feature, n_latent, n_hidden=n_hidden, act=act, dropout=dropout)

    def forward(self, flux):
        if flux.ndim == 2:
            flux = flux.unsqueeze(1)

        x = flux
        for i, conv in enumerate(self.convs):
            x = conv(x)
            if i < len(self.pools):
                x = self.pools[i](x)

        values, keys = torch.split(x, self.n_feature, dim=1)
        attention = self.softmax(keys)
        return self.mlp((values * attention).sum(dim=2))


@register_model
def spectrum_encoder(**kwargs) -> SpectrumEncoder:
    return SpectrumEncoder(**kwargs)
