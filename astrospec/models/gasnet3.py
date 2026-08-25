"""GaSNet-III, a CNN encoder over a learnable eigenvector basis.

Zhong et al. 2025, MNRAS 543:691. https://doi.org/10.1093/mnras/staf1482
Reference implementation: https://github.com/Fucheng-Zhong/GaSNet-III
"""

import torch
from torch import nn, Tensor

from ..registry import register_model

__all__ = ["GaSNet3", "gasnet3"]


class GaSNet3(nn.Module):
    """CNN encoder producing amplitude and eigenvector coefficients.

    Consumes ``flux`` only; ``wavelength``, ``ivar``, ``mask``, and
    ``lsf_sigma`` are ignored. Three conv+pool stages compress the input
    spectrum to ``n_eigenvectors + 1`` numbers: the last is a reconstruction
    amplitude, the rest are coefficients (squared and normalized to sum to 1)
    over a learnable rest-frame eigenvector basis. The reconstruction is the
    amplitude-scaled, coefficient-weighted sum of that basis, on the
    ``output_dim``-pixel rest-frame grid.

    Fixed-grid model: the observed-frame input has ``input_dim`` pixels, the
    rest-frame output has ``output_dim`` pixels; both are set at construction.
    Class-agnostic across STAR / GALAXY / QSO. The χ² curve between an
    input spectrum and this reconstruction over a grid of trial redshifts is
    how the source paper estimates redshift and flags anomalies; that search
    is a downstream task, not part of the model itself.

    Args:
        input_dim: pixels per observed-frame input spectrum.
        output_dim: pixels per rest-frame reconstruction.
        n_eigenvectors: size of the learnable eigenvector basis.

    Shape:
        input ``(B, L)`` or ``(B, 1, L)`` -> reconstruction
        ``(B, output_dim)``, coefficients ``(B, n_eigenvectors)``.
    """

    def __init__(self, input_dim: int, output_dim: int, n_eigenvectors: int = 20):
        super().__init__()
        self.n_eigenvectors = n_eigenvectors

        self.eigenvectors = nn.Parameter(torch.rand(n_eigenvectors, output_dim))

        conv_out = lambda n: (((n - 9 + 2) // 5 + 1) - 2) // 2 + 1
        flat_dim = 32 * conv_out(conv_out(conv_out(input_dim)))

        self.encoder = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=9, stride=5, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(8, 16, kernel_size=9, stride=5, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(16, 32, kernel_size=9, stride=5, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(flat_dim, n_eigenvectors + 1),
        )

    def forward(self, flux: Tensor):
        if flux.ndim == 2:
            flux = flux.unsqueeze(1)
        x = self.encoder(flux)  # (B, n_eigenvectors + 1)

        amplitude = x[:, self.n_eigenvectors :].abs()  # (B, 1)
        coef = x[:, : self.n_eigenvectors]
        coef = coef**2 / (coef**2).sum(dim=-1, keepdim=True).clamp(min=1e-8)

        reconstruction = amplitude * (coef @ self.eigenvectors)  # (B, output_dim)
        return reconstruction, coef


@register_model
def gasnet3(input_dim: int = 3522, output_dim: int = 4000, **kwargs) -> GaSNet3:
    return GaSNet3(input_dim=input_dim, output_dim=output_dim, **kwargs)
