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
HF_CONFIG = "edr_sv3"

#: dataset column holding the pipeline classification (``CLASS`` in SDSS)
CLASS_COLUMN = "SPECTYPE"

#: dataset column holding the redshift warning bitmask (``ZWARNING`` in SDSS)
ZWARN_COLUMN = "ZWARN"


def local_files(root: str = None) -> list:
    """HDF5 shards of a local MMU-format DESI tree, or ``[]`` if there is none."""
    root = root or os.environ.get("ASTROSPEC_DESI_ROOT", "")
    return sorted(glob(os.path.join(root, "healpix=*", "*.hdf5"))) if root else []


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

    Returns ``(flux, ivar, mask, wavelength, z)``, all local-tree only: AION's
    spectrum codec (see ``utils/aion.py``) wants ``ivar`` and ``mask``
    directly, which the Hub release does not carry. ``wavelength`` is the
    file's own ``spectrum_lambda``, identical for every DESI EDR row, but read
    per row rather than assumed so a differently-processed tree is still
    handled correctly.
    """
    import h5py

    files = local_files(root)
    if not files:
        raise FileNotFoundError(
            "no local DESI HDF5 found; set ASTROSPEC_DESI_ROOT to a directory of "
            "healpix=*/ shards (ivar/mask are not in the Hub release)"
        )

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


def _read_hf(n: int, redshift: bool) -> tuple:
    from datasets import load_dataset
    from tqdm.auto import tqdm

    columns = ["spectrum"] + (["Z"] if redshift else [])
    stream = (
        load_dataset(HF_DATASET, HF_CONFIG, split="train", streaming=True)
        .select_columns(columns)
    )
    rows = list(tqdm(stream.take(n), total=n, desc="streaming desi", unit="spec"))

    flux = [np.asarray(row["spectrum"]["flux"], dtype=np.float32) for row in rows]
    z = np.asarray([row["Z"] for row in rows], dtype=np.float32) if redshift else None
    return flux, z
