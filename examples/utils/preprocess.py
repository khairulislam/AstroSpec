"""Shared per-spectrum normalization for the AstroSpec examples.

Absolute flux depends on distance, aperture and calibration rather than on the
physics these models read, so each spectrum is normalized against itself:

1. sigma-clip the narrow spikes cosmic rays, dead pixels and sky residuals
   leave behind, so instrumental artefacts do not set the scale factor;
2. divide by the spectrum's own mean flux, which puts the continuum near 1, so
   subtracting 1 centres it on zero with absorption negative and emission
   positive;
3. compress with ``arcsinh``, which is logarithmic for the emission lines that
   run orders of magnitude above the continuum, linear near zero, and unlike a
   logarithm defined for the negative values sky subtraction produces.

Matches OmniSpectrum's ``SpectrumPreprocessor``
(``/project/MLSys/OmniSpectrum/data/preprocessor.py``) under
``configs/data/omnicollator_sdss.yaml``, whose compression path takes the scale
factor from the clamped mean over valid pixels.

Every loader's ``standardize`` reuses this rather than reimplementing it.
"""

import numpy as np


def standardize(
    flux: np.ndarray,
    sigma_clip_threshold: float = 5.0,
    input_scaling: float = 1.0,
    norm_floor: float = 0.1,
) -> np.ndarray:
    """Sigma-clip, mean-normalize, and arcsinh-compress each spectrum independently.

    ``norm_floor`` keeps the scale factor away from zero for faint or
    sky-dominated spectra; it assumes survey flux units, so a release stored on
    a very different scale should pass its own value.
    """
    flux = np.asarray(flux, dtype=np.float32)
    valid = np.isfinite(flux)
    flux = np.where(valid, flux, 0.0)

    mean = np.zeros((len(flux), 1), dtype=np.float32)
    for _ in range(3):
        count = valid.sum(axis=-1, keepdims=True)
        mean = (flux * valid).sum(axis=-1, keepdims=True) / np.maximum(count, 1)
        variance = ((flux - mean) ** 2 * valid).sum(axis=-1, keepdims=True) / np.maximum(count - 1, 1)
        std = np.maximum(np.sqrt(variance), 1e-6)
        valid &= np.abs(flux - mean) <= sigma_clip_threshold * std
    flux = np.where(valid, flux, mean)

    norm = np.maximum(
        (flux * valid).sum(axis=-1, keepdims=True) / (valid.sum(axis=-1, keepdims=True) + 1.0),
        norm_floor,
    )
    flux = np.where(valid, flux / norm - 1.0, 0.0)
    return np.arcsinh(flux * input_scaling).astype(np.float32)
