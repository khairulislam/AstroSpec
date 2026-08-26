"""GalSpecNet, a 1-D CNN classifier for optical spectra.

Wu et al. 2023, MNRAS 527:1163. https://doi.org/10.1093/mnras/stad2913
"""

from typing import Sequence

from torch import nn

from ..layers import MLP
from ..registry import register_model

__all__ = ["GalSpecNet", "galspecnet"]


class GalSpecNet(nn.Module):
    """Stacked 1-D convolutions and max-pools followed by an MLP head.

    Consumes ``flux`` only; ``wavelength``, ``ivar``, ``mask``, and
    ``lsf_sigma`` are ignored. Fixed-grid model: every spectrum must be
    resampled to ``input_length`` pixels, because the head is a flat MLP over
    the final convolutional activations.

    Args:
        input_length: number of pixels per spectrum.
        num_classes: output dimension; class logits, or regression targets.
        conv_channels: channel widths, starting at 1 for the raw flux channel.
        kernel_size: convolution width, no padding.
        mp_kernel_size: max-pool width, applied after every convolution but
            the last.
        dropout: dropout rate before the head.
        n_hidden: hidden widths of the MLP head.

    Shape:
        input ``(B, L)`` or ``(B, 1, L)`` -> output ``(B, num_classes)``.
    """

    def __init__(
        self,
        input_length: int,
        num_classes: int,
        conv_channels: Sequence[int] = (1, 64, 64, 32, 32),
        kernel_size: int = 3,
        mp_kernel_size: int = 4,
        dropout: float = 0.1,
        n_hidden: Sequence[int] = (256, 64, 16),
    ):
        super().__init__()

        layers = []
        length = input_length
        for i in range(len(conv_channels) - 1):
            layers += [
                nn.Conv1d(conv_channels[i], conv_channels[i + 1], kernel_size),
                nn.ReLU(),
            ]
            length -= kernel_size - 1
            if i < len(conv_channels) - 2:
                layers.append(nn.MaxPool1d(mp_kernel_size))
                length //= mp_kernel_size
        if length < 1:
            raise ValueError(
                f"input_length={input_length} is too short for {len(conv_channels) - 1} "
                f"convolutions of width {kernel_size}"
            )

        self.features = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)
        # dropout=0.0: GalSpecNet applies dropout once, above, before the head,
        # not between the head's own layers.
        self.head = MLP(conv_channels[-1] * length, num_classes, n_hidden=n_hidden, dropout=0.0)

    def forward(self, flux):
        if flux.ndim == 2:
            flux = flux.unsqueeze(1)
        x = self.features(flux)
        x = self.dropout(x.flatten(1))
        return self.head(x)


@register_model
def galspecnet(**kwargs) -> GalSpecNet:
    return GalSpecNet(**kwargs)
