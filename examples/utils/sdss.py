"""Shared SDSS spectrum loading for the AstroSpec examples.

Both sources are the MultimodalUniverse SDSS release:

- a local HDF5 tree (``healpix=*/*.hdf5``), pointed at by ``ASTROSPEC_SDSS_ROOT``
  or the ``root`` argument;
- otherwise the Hub copy, streamed from
  `MultimodalUniverse/sdss <https://huggingface.co/datasets/MultimodalUniverse/sdss>`_.

The two differ: the Hub copy carries the spectra and ``Z`` but not the pipeline
``CLASS``, so the classification example needs the local tree. Use one
sub-survey at a time. The legacy ``sdss`` spectrograph and ``boss``/``eboss``
cover different wavelength ranges.

Individual SDSS spectra share a log-lambda pixel spacing but not a common
wavelength window, so their pixel counts differ from object to object. The
models in this library that take a fixed grid cannot see that variation, hence
:func:`resample` and the :data:`COMMON_GRID` these loaders interpolate onto.
AstroSpec ships no such resampling itself. The choice of grid, interpolation,
and normalization belongs to the dataset, not the model.
"""

import os
from glob import glob

import numpy as np

HF_DATASET = "MultimodalUniverse/sdss"
CLASS_NAMES = ["GALAXY", "QSO", "STAR"]

#: log-spaced wavelength grid in Angstrom, the SDSS 1e-4 dex pixel spacing over
#: the range the legacy spectrograph covers for nearly every object.
COMMON_GRID = 10 ** np.arange(np.log10(3800.0), np.log10(9200.0), 1e-4, dtype=np.float64)
COMMON_GRID = COMMON_GRID.astype(np.float32)


def resample(flux, wavelength, grid: np.ndarray = COMMON_GRID) -> np.ndarray:
    """Linearly interpolate each spectrum onto ``grid``; zero outside its coverage.

    Pixels carrying the ``-1`` wavelength sentinel this release uses for padding
    are dropped first, otherwise the wavelength axis is not increasing and the
    interpolation is meaningless.
    """
    out = np.empty((len(flux), len(grid)), dtype=np.float32)
    for i, (f, w) in enumerate(zip(flux, wavelength)):
        keep = w > 0
        out[i] = np.interp(grid, w[keep], np.nan_to_num(f)[keep], left=0.0, right=0.0)
    return out


def standardize(flux: np.ndarray) -> np.ndarray:
    """Replace non-finite pixels with zero and standardize each spectrum.

    Flux is in survey units and varies by orders of magnitude between objects,
    while the models expect a NaN-free, roughly unit-scale sequence.
    Per-spectrum standardization discards the absolute flux level, so it suits
    shape-driven tasks (classification, redshift) rather than photometric ones.
    """
    flux = np.nan_to_num(flux, nan=0.0, posinf=0.0, neginf=0.0)
    mean = flux.mean(axis=-1, keepdims=True)
    std = flux.std(axis=-1, keepdims=True)
    return (flux - mean) / np.maximum(std, 1e-6)


def local_files(root: str = None) -> list:
    """HDF5 shards of a local MMU-format sub-survey, or ``[]`` if there is no tree."""
    root = root or os.environ.get("ASTROSPEC_SDSS_ROOT", "")
    return sorted(glob(os.path.join(root, "healpix=*", "*.hdf5"))) if root else []


def load_classes(n_per_class: int, root: str = None, seed: int = 0) -> tuple:
    """Draw ``n_per_class`` GALAXY / QSO / STAR spectra from the local tree.

    Returns ``(flux, labels)``: ``(N, len(COMMON_GRID))`` float32 resampled flux
    and ``(N,)`` int64 indices into :data:`CLASS_NAMES`, shuffled. Labels are
    the SDSS pipeline ``CLASS``, i.e. what the survey's own template fit
    decided, not an independent ground truth.
    """
    import h5py

    files = local_files(root)
    if not files:
        raise FileNotFoundError(
            "no local SDSS HDF5 found; set ASTROSPEC_SDSS_ROOT to a directory of "
            "healpix=*/ shards (the Hub copy carries no CLASS labels)"
        )

    wanted = [name.encode().ljust(6) for name in CLASS_NAMES]
    chunks = [[] for _ in CLASS_NAMES]
    for path in files:
        if all(sum(len(f) for f, _ in c) >= n_per_class for c in chunks):
            break
        with h5py.File(path, "r") as file:
            classes = file["CLASS"][:]
            for label, key in enumerate(wanted):
                missing = n_per_class - sum(len(f) for f, _ in chunks[label])
                if missing <= 0:
                    continue
                rows = np.flatnonzero(classes == key)[:missing]
                if len(rows):
                    chunks[label].append(
                        (file["spectrum_flux"][rows], file["spectrum_lambda"][rows])
                    )

    counts = [sum(len(f) for f, _ in c) for c in chunks]
    if min(counts) < n_per_class:
        raise ValueError(
            f"only found {counts} spectra for {CLASS_NAMES}, wanted {n_per_class} each"
        )

    flux = np.concatenate([resample(f, w) for per_class in chunks for f, w in per_class])
    labels = np.concatenate([np.full(counts[i], i) for i in range(len(CLASS_NAMES))])

    order = np.random.default_rng(seed).permutation(len(labels))
    return flux[order], labels[order].astype(np.int64)


def load_spectra(n: int, root: str = None, redshift: bool = False) -> tuple:
    """Load ``n`` spectra, from the local tree if there is one, else the Hub.

    Returns ``(flux, z)``, both resampled onto :data:`COMMON_GRID`, with ``z``
    ``None`` unless ``redshift``. Read in file order rather than sampled, at
    most 200 per local shard so the subset spreads over the sky instead of
    sitting inside one healpix pixel.
    """
    files = local_files(root)
    flux, wavelength, z = (
        _read_local(files, n, redshift) if files else _read_hf(n, redshift)
    )
    return resample(flux, wavelength), z


def _read_local(files: list, n: int, redshift: bool) -> tuple:
    import h5py

    flux, wavelength, z = [], [], []
    taken = 0
    for path in files:
        if taken >= n:
            break
        with h5py.File(path, "r") as file:
            rows = slice(0, min(n - taken, 200))
            shard = file["spectrum_flux"][rows]
            flux.extend(shard)
            wavelength.extend(file["spectrum_lambda"][rows])
            if redshift:
                z.append(file["Z"][rows])
            taken += len(shard)

    return flux, wavelength, np.concatenate(z).astype(np.float32) if redshift else None


def _read_hf(n: int, redshift: bool) -> tuple:
    from datasets import load_dataset
    from tqdm.auto import tqdm

    columns = ["spectrum"] + (["Z"] if redshift else [])
    stream = load_dataset(HF_DATASET, split="train", streaming=True).select_columns(columns)
    rows = list(tqdm(stream.take(n), total=n, desc="streaming sdss", unit="spec"))

    flux = [np.asarray(row["spectrum"]["flux"], dtype=np.float32) for row in rows]
    wavelength = [np.asarray(row["spectrum"]["lambda"], dtype=np.float32) for row in rows]
    z = np.asarray([row["Z"] for row in rows], dtype=np.float32) if redshift else None
    return flux, wavelength, z
