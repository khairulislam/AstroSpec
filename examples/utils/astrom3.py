"""AstroM3 processed-spectrum loading for the AstroSpec examples."""

import numpy as np

from . import preprocess

HF_DATASET = "AstroMLCore/AstroM3Processed"
HF_CONFIG = "full_42"

#: AstroM3Processed ships flux already scaled to ~1e-3, orders of magnitude below
#: the survey-unit floor the shared preprocessor defaults to. Left at 0.1 the
#: floor would replace every scale factor with a constant, collapsing faint
#: spectra to a flat line instead of normalizing them.
NORM_FLOOR = 1e-8


def standardize(flux: np.ndarray) -> np.ndarray:
    """Shared preprocessing, floored for this release's flux scale."""
    return preprocess.standardize(flux, norm_floor=NORM_FLOOR)


def class_names(config: str = HF_CONFIG) -> list[str]:
    """Return the variable-star class names in the release's label order."""
    from datasets import load_dataset_builder

    return load_dataset_builder(HF_DATASET, config).info.features["label"].names


def load_split(split: str, n: int = None, config: str = HF_CONFIG) -> tuple:
    """Stream an AstroM3 split and return its flux channel and integer labels.

    AstroM3Processed stores each LAMOST spectrum as wavelength, flux, and flux
    error rows on a common 2,575-pixel grid.  The full publisher-provided split
    is the default; pass a smaller ``sub*`` configuration for a quick smoke run.
    """
    from datasets import load_dataset

    stream = load_dataset(HF_DATASET, config, split=split, streaming=True)
    if n is not None:
        stream = stream.take(n)
    rows = list(stream)
    if not rows:
        raise ValueError(f"AstroM3 {config!r} split {split!r} is empty")

    spectra = np.asarray([row["spectra"] for row in rows], dtype=np.float32)
    if spectra.ndim != 3 or spectra.shape[1] != 3:
        raise ValueError(f"expected (N, 3, L) AstroM3 spectra, got {spectra.shape}")
    return standardize(spectra[:, 1]), np.asarray([row["label"] for row in rows], dtype=np.int64)
