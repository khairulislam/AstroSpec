"""Shared DESI spectrum loading for the AstroSpec examples.

Both sources are the MultimodalUniverse DESI EDR SV3 release:

- a local HDF5 tree (``healpix=*/*.hdf5``), pointed at by ``ASTROSPEC_DESI_ROOT``
  or the ``root`` argument;
- otherwise the Hub copy, streamed from
  `MultimodalUniverse/desi <https://huggingface.co/datasets/MultimodalUniverse/desi>`_.

Unlike SDSS, every DESI EDR spectrum in this release already sits on the same
3,600-9,824 A, 0.80 A/pix grid (7,781 pixels), the grid AstroCLIP's spectrum
encoder was trained on, so this module does no resampling.
"""

import os
from glob import glob

import numpy as np

from .preprocess import standardize

HF_DATASET = "MultimodalUniverse/desi"
HF_CONFIG = "default"

#: dataset column holding the pipeline classification (``CLASS`` in SDSS)
CLASS_COLUMN = "SPECTYPE"

#: dataset column holding the redshift warning bitmask (``ZWARNING`` in SDSS)
ZWARN_COLUMN = "ZWARN"


def local_files(root: str = None) -> list:
    """HDF5 shards of a local MMU-format DESI tree, or ``[]`` if unset.

    A *set* root that matches no shards is almost always a typo, not an
    intentional request to stream, so that case raises rather than falling
    through to the Hub silently.
    """
    root = root or os.environ.get("ASTROSPEC_DESI_ROOT", "")
    if not root:
        return []
    files = sorted(glob(os.path.join(root, "healpix=*", "*.hdf5")))
    if not files:
        raise FileNotFoundError(
            f"ASTROSPEC_DESI_ROOT={root!r} has no healpix=*/*.hdf5 shards; check the path"
        )
    return files


def load_spectra(
    n: int,
    root: str = None,
    redshift: bool = False,
    classes: tuple = None,
    max_zwarning: int = None,
) -> tuple:
    """Load ``n`` spectra, from the local tree if there is one, else the Hub.

    Returns ``(flux, z)``, ``flux`` already on the common 7,781-pixel grid,
    with ``z`` ``None`` unless ``redshift``. Read in file order rather than
    sampled, at most 200 per local shard so the subset spreads over the sky
    instead of sitting inside one healpix pixel.

    ``classes`` restricts the sample to pipeline :data:`CLASS_COLUMN` values,
    e.g. ``("GALAXY",)``. Redshift means something different for each class, so
    a regression on ``Z`` should pick one.

    ``max_zwarning`` keeps only rows whose :data:`ZWARN_COLUMN` bitmask is at
    most this; pass ``0`` to drop every redshift the pipeline flagged.

    Both filters need the local tree.
    """
    files = local_files(root)
    if (classes is not None or max_zwarning is not None) and not files:
        raise FileNotFoundError(
            f"filtering on {CLASS_COLUMN}/{ZWARN_COLUMN} needs a local DESI tree; set "
            "ASTROSPEC_DESI_ROOT to a directory of healpix=*/ shards"
        )
    flux, z = (
        _read_local(files, n, redshift, classes, max_zwarning)
        if files
        else _read_hf(n, redshift)
    )
    return np.stack(flux).astype(np.float32), z


def _read_local(
    files: list,
    n: int,
    redshift: bool,
    classes: tuple = None,
    max_zwarning: int = None,
) -> tuple:
    import h5py

    flux, z = [], []
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
            if redshift:
                z.append(file["Z"][rows])
            taken += len(rows)

    return flux, np.concatenate(z).astype(np.float32) if redshift else None


def load_native(n: int, root: str = None, redshift: bool = False) -> tuple:
    """Load ``n`` spectra with their per-pixel inverse variance and bad-pixel mask.

    Returns ``(flux, ivar, mask, wavelength, z)``, from the local tree if there
    is one, else the Hub: its ``spectrum`` column carries ``ivar``, ``mask``,
    and ``lambda`` alongside ``flux``, so AION's spectrum codec (see
    ``utils/aion.py``), which wants all four, works either way.
    """
    files = local_files(root)
    return (
        _read_local_native(files, n, redshift)
        if files
        else _read_hf_native(n, redshift)
    )


def _read_local_native(files: list, n: int, redshift: bool) -> tuple:
    import h5py

    flux, ivar, mask, wavelength, z = [], [], [], [], []
    taken = 0
    for path in files:
        if taken >= n:
            break
        with h5py.File(path, "r") as file:
            rows = np.arange(min(n - taken, 200, len(file["Z"])))
            if not len(rows):
                continue
            flux.extend(file["spectrum_flux"][rows])
            ivar.extend(file["spectrum_ivar"][rows])
            mask.extend(file["spectrum_mask"][rows])
            wavelength.extend(file["spectrum_lambda"][rows])
            if redshift:
                z.append(file["Z"][rows])
            taken += len(rows)

    return (
        np.stack(flux).astype(np.float32),
        np.stack(ivar).astype(np.float32),
        np.stack(mask).astype(bool),
        np.stack(wavelength).astype(np.float32),
        np.concatenate(z).astype(np.float32) if redshift else None,
    )


def _stream(n: int, redshift: bool) -> list:
    from datasets import load_dataset
    from tqdm import tqdm

    columns = ["spectrum"] + (["Z"] if redshift else [])
    stream = (
        load_dataset(HF_DATASET, HF_CONFIG, split="train", streaming=True)
        .select_columns(columns)
    )
    return list(tqdm(stream.take(n), total=n, desc="streaming desi", unit="spec"))


def _read_hf(n: int, redshift: bool) -> tuple:
    rows = _stream(n, redshift)
    flux = [np.asarray(row["spectrum"]["flux"], dtype=np.float32) for row in rows]
    z = np.asarray([row["Z"] for row in rows], dtype=np.float32) if redshift else None
    return flux, z


def _read_hf_native(n: int, redshift: bool) -> tuple:
    rows = _stream(n, redshift)
    flux = np.stack([np.asarray(r["spectrum"]["flux"], dtype=np.float32) for r in rows])
    ivar = np.stack([np.asarray(r["spectrum"]["ivar"], dtype=np.float32) for r in rows])
    mask = np.stack([np.asarray(r["spectrum"]["mask"], dtype=bool) for r in rows])
    wavelength = np.stack([np.asarray(r["spectrum"]["lambda"], dtype=np.float32) for r in rows])
    z = np.asarray([r["Z"] for r in rows], dtype=np.float32) if redshift else None
    return flux, ivar, mask, wavelength, z
