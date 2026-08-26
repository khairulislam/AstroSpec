"""Example-only helpers built on astrospec.pretrained.aion.

AION's own spectrum codec resamples onto its own internal latent wavelength
grid, so it wants native per-pixel flux, inverse variance, bad-pixel mask,
and wavelength directly -- unlike this library's own fixed/patched-grid
models, which need `utils.desi`/`utils.sdss`'s COMMON_GRID resampling
instead.
"""

import numpy as np
import torch
from tqdm.auto import tqdm


@torch.no_grad()
def compute_spectrum_embeddings(
    model,
    codec_manager,
    flux: np.ndarray,
    ivar: np.ndarray,
    mask: np.ndarray,
    wavelength: np.ndarray,
    device,
    batch_size: int = 16,
) -> np.ndarray:
    """One mean-pooled AION embedding per spectrum, from the DESI spectrum modality."""
    from aion.modalities import DESISpectrum

    embeddings = []
    for i in tqdm(range(0, len(flux), batch_size), desc="embedding", unit="batch"):
        sl = slice(i, i + batch_size)
        spectrum = DESISpectrum(
            flux=torch.tensor(flux[sl], device=device),
            ivar=torch.tensor(ivar[sl], device=device),
            mask=torch.tensor(mask[sl], device=device),
            wavelength=torch.tensor(wavelength[sl], device=device),
        )
        tokens = codec_manager.encode(spectrum)
        embeddings.append(
            model.encode(tokens, num_encoder_tokens=spectrum.num_tokens).mean(dim=1).cpu()
        )
    return torch.cat(embeddings).numpy()
