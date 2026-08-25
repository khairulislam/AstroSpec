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

HF_DATASET = "MultimodalUniverse/desi"
HF_CONFIG = "edr_sv3"


def standardize(flux: np.ndarray) -> np.ndarray:
    """Replace non-finite pixels with zero and standardize each spectrum."""
    flux = np.nan_to_num(flux, nan=0.0, posinf=0.0, neginf=0.0)
    mean = flux.mean(axis=-1, keepdims=True)
    std = flux.std(axis=-1, keepdims=True)
    return (flux - mean) / np.maximum(std, 1e-6)


def local_files(root: str = None) -> list:
    """HDF5 shards of a local MMU-format DESI tree, or ``[]`` if there is none."""
    root = root or os.environ.get("ASTROSPEC_DESI_ROOT", "")
    return sorted(glob(os.path.join(root, "healpix=*", "*.hdf5"))) if root else []


def load_spectra(n: int, root: str = None, redshift: bool = False) -> tuple:
    """Load ``n`` spectra, from the local tree if there is one, else the Hub.

    Returns ``(flux, z)``, ``flux`` already on the common 7,781-pixel grid,
    with ``z`` ``None`` unless ``redshift``. Read in file order rather than
    sampled, at most 200 per local shard so the subset spreads over the sky
    instead of sitting inside one healpix pixel.
    """
    files = local_files(root)
    flux, z = _read_local(files, n, redshift) if files else _read_hf(n, redshift)
    return np.stack(flux).astype(np.float32), z


def _read_local(files: list, n: int, redshift: bool) -> tuple:
    import h5py

    flux, z = [], []
    taken = 0
    for path in files:
        if taken >= n:
            break
        with h5py.File(path, "r") as file:
            rows = slice(0, min(n - taken, 200))
            shard = file["spectrum_flux"][rows]
            flux.extend(shard)
            if redshift:
                z.append(file["Z"][rows])
            taken += len(shard)

    return flux, np.concatenate(z).astype(np.float32) if redshift else None


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
