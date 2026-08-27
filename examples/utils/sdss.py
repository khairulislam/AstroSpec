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

from .preprocess import standardize

HF_DATASET = "MultimodalUniverse/sdss"
CLASS_NAMES = ["GALAXY", "QSO", "STAR"]

#: dataset column holding the pipeline classification (``SPECTYPE`` in DESI)
CLASS_COLUMN = "CLASS"

#: dataset column holding the redshift warning bitmask (``ZWARN`` in DESI)
ZWARN_COLUMN = "ZWARNING"

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


def local_files(root: str = None) -> list:
    """HDF5 shards of a local MMU-format sub-survey, or ``[]`` if unset.

    A *set* root that matches no shards is almost always a typo, not an
    intentional request to stream, so that case raises rather than falling
    through to the Hub silently.
    """
    root = root or os.environ.get("ASTROSPEC_SDSS_ROOT", "")
    if not root:
        return []
    files = sorted(glob(os.path.join(root, "healpix=*", "*.hdf5")))
    if not files:
        raise FileNotFoundError(
            f"ASTROSPEC_SDSS_ROOT={root!r} has no healpix=*/*.hdf5 shards; check the path"
        )
    return files


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


def load_sample(n: int, root: str = None, seed: int = 0) -> tuple:
    """Draw ``n`` GALAXY / QSO / STAR spectra from the local tree, unbalanced.

    Unlike :func:`load_classes`, classes are not equalized: the returned
    labels carry SDSS's real class prevalence (mostly GALAXY, few QSO), which
    is what a model sees at inference time. Returns ``(flux, labels)`` as in
    :func:`load_classes`.
    """
    import h5py

    files = local_files(root)
    if not files:
        raise FileNotFoundError(
            "no local SDSS HDF5 found; set ASTROSPEC_SDSS_ROOT to a directory of "
            "healpix=*/ shards (the Hub copy carries no CLASS labels)"
        )

    wanted = {name.encode().ljust(6): i for i, name in enumerate(CLASS_NAMES)}
    flux, wavelength, labels = [], [], []
    taken = 0
    for path in files:
        if taken >= n:
            break
        with h5py.File(path, "r") as file:
            classes = file["CLASS"][:]
            keep = np.flatnonzero(np.isin(classes, list(wanted)))[: n - taken]
            if len(keep) == 0:
                continue
            flux.extend(file["spectrum_flux"][keep])
            wavelength.extend(file["spectrum_lambda"][keep])
            labels.extend(wanted[c] for c in classes[keep])
            taken += len(keep)

    if taken < n:
        raise ValueError(f"only found {taken} labeled spectra, wanted {n}")

    flux = resample(flux, wavelength)
    labels = np.asarray(labels, dtype=np.int64)
    order = np.random.default_rng(seed).permutation(taken)
    return flux[order], labels[order]


def load_spectra(
    n: int,
    root: str = None,
    redshift: bool = False,
    classes: tuple = None,
    max_zwarning: int = None,
) -> tuple:
    """Load ``n`` spectra, from the local tree if there is one, else the Hub.

    Returns ``(flux, z)``, both resampled onto :data:`COMMON_GRID`, with ``z``
    ``None`` unless ``redshift``. Read in file order rather than sampled, at
    most 200 per local shard so the subset spreads over the sky instead of
    sitting inside one healpix pixel.

    ``classes`` restricts the sample to pipeline :data:`CLASS_COLUMN` values,
    e.g. ``("GALAXY",)``. Redshift means something different for each class, so
    a regression on ``Z`` should pick one.

    ``max_zwarning`` keeps only rows whose :data:`ZWARN_COLUMN` bitmask is at
    most this; pass ``0`` to drop every redshift the pipeline flagged, which is
    what a supervised target should use.

    Both filters need the local tree, since the Hub copy carries neither column.
    """
    files = local_files(root)
    if (classes is not None or max_zwarning is not None) and not files:
        raise FileNotFoundError(
            f"filtering on {CLASS_COLUMN}/{ZWARN_COLUMN} needs a local SDSS tree; set "
            "ASTROSPEC_SDSS_ROOT to a directory of healpix=*/ shards (the Hub copy "
            "carries neither column)"
        )
    flux, wavelength, z = (
        _read_local(files, n, redshift, classes, max_zwarning)
        if files
        else _read_hf(n, redshift)
    )
    return resample(flux, wavelength), z


def _read_local(
    files: list,
    n: int,
    redshift: bool,
    classes: tuple = None,
    max_zwarning: int = None,
) -> tuple:
    import h5py

    flux, wavelength, z = [], [], []
    taken = 0
    for path in files:
        if taken >= n:
            break
        with h5py.File(path, "r") as file:
            keep = np.ones(len(file["Z"]), dtype=bool)
            if classes is not None:
                keep &= np.isin(np.char.strip(file[CLASS_COLUMN][:].astype("U")), list(classes))
            if max_zwarning is not None:
                keep &= file[ZWARN_COLUMN][:] <= max_zwarning
            rows = np.flatnonzero(keep)[: min(n - taken, 200)]
            if not len(rows):
                continue
            flux.extend(file["spectrum_flux"][rows])
            wavelength.extend(file["spectrum_lambda"][rows])
            if redshift:
                z.append(file["Z"][rows])
            taken += len(rows)

    return flux, wavelength, np.concatenate(z).astype(np.float32) if redshift else None


def _read_hf(n: int, redshift: bool) -> tuple:
    from datasets import load_dataset
    from tqdm import tqdm

    columns = ["spectrum"] + (["Z"] if redshift else [])
    stream = load_dataset(HF_DATASET, split="train", streaming=True).select_columns(columns)
    rows = list(tqdm(stream.take(n), total=n, desc="streaming sdss", unit="spec"))

    flux = [np.asarray(row["spectrum"]["flux"], dtype=np.float32) for row in rows]
    wavelength = [np.asarray(row["spectrum"]["lambda"], dtype=np.float32) for row in rows]
    z = np.asarray([row["Z"] for row in rows], dtype=np.float32) if redshift else None
    return flux, wavelength, z
