# Examples

Optional, self-contained notebooks. Each installs its own extra dependencies in
its first cell; none of this is required to use `astrospec` itself.

## Data

Both notebooks read the [MultimodalUniverse](https://github.com/MultimodalUniverse/MultimodalUniverse)
SDSS release through `utils/sdss.py`, which takes it from a local HDF5 tree when
`ASTROSPEC_SDSS_ROOT` points at one and streams
[`MultimodalUniverse/sdss`](https://huggingface.co/datasets/MultimodalUniverse/sdss)
from the Hub otherwise:

```bash
export ASTROSPEC_SDSS_ROOT=/path/to/sdss   # a directory of healpix=*/ shards
```

The two copies are not identical. The Hub copy carries the spectra and `Z` but
not the pipeline `CLASS`, so `sdss_classification.ipynb` needs a local tree.

`utils/sdss.py` also resamples: SDSS spectra share a log-lambda pixel spacing but
cover slightly different wavelength windows, so their pixel counts differ from
object to object, and the fixed-grid models cannot see that variation. AstroSpec
ships no resampling of its own. The grid, the interpolation, and the
normalization are properties of a dataset, not of a model.

## `sdss_classification.ipynb`

Trains `GalSpecNet` to separate the SDSS pipeline classes GALAXY / QSO / STAR on
6,000 spectra, split 70/10/20, and reports a per-class report and confusion
matrix. The shortest end-to-end path through the library: load, resample,
`create_model`, train, evaluate.

## `specformer_pretraining.ipynb`

Pretrains `SpecFormer` on 16,000 unlabelled spectra by reconstructing masked
patches, which is AstroCLIP's spectrum-tower objective. It then mean-pools
`forward_features` into one vector per spectrum and searches those for cosine
nearest neighbours. Redshift, which the model never sees, only serves as a check
that retrieved neighbours are physically similar. Uses `astrospec.data.Patchify`
and the 22-number patch encoding (patch standardized by its own mean and
standard deviation, with both appended) that SpecFormer expects.
