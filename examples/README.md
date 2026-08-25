# Examples

Optional, self-contained notebooks. Each installs its own extra dependencies in
its first cell; none of this is required to use `astrospec` itself.

## Data

Four notebooks read SDSS spectra through `utils/sdss.py`, from a local HDF5
tree when `ASTROSPEC_SDSS_ROOT` points at one and streaming
[`MultimodalUniverse/sdss`](https://huggingface.co/datasets/MultimodalUniverse/sdss)
otherwise:

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

`astroclip_similarity_search.ipynb` reads [MultimodalUniverse](https://github.com/MultimodalUniverse/MultimodalUniverse)
DESI EDR SV3 spectra instead, through `utils/desi.py`, the grid AstroCLIP's
spectrum encoder was actually trained on:

```bash
export ASTROSPEC_DESI_ROOT=/path/to/desi/edr_sv3   # a directory of healpix=*/ shards
```

Every spectrum in this release already sits on the same 7,781-pixel grid, so
`utils/desi.py` does no resampling.

## `sdss_classification.ipynb`

Trains `GalSpecNet` to separate the SDSS pipeline classes GALAXY / QSO / STAR on
6,000 spectra, split 70/10/20, and reports a per-class report and confusion
matrix. The shortest end-to-end path through the library: load, resample,
`create_model`, train, evaluate.

## `specformer_pretraining.ipynb`

Pretrains `SpecFormer` at its published size, 6 layers of width 768, on 16,000
unlabelled spectra by reconstructing masked patches, which is AstroCLIP's
spectrum-tower objective. It then mean-pools
`forward_features` into one vector per spectrum and searches those for cosine
nearest neighbours. Redshift, which the model never sees, only serves as a check
that retrieved neighbours are physically similar. Uses `astrospec.data.Patchify`
and the 22-number patch encoding (patch standardized by its own mean and
standard deviation, with both appended) that SpecFormer expects.

45 epochs, about half an hour on an A100. Two settings are load-bearing at this
width. Without a warmup plus cosine learning-rate schedule the loss turns back
up around epoch 10 and the encoder collapses, and without centering the pooled
embeddings before the cosine every pair scores above 0.999.

## `specformer_pretrained_similarity_search.ipynb`

The pretrained counterpart to `specformer_pretraining.ipynb`: instead of
training `SpecFormer` from scratch, it loads the released spectrum tower of
AstroCLIP's checkpoint via `astrospec.pretrained.load_pretrained_specformer`
into `SpecFormer` directly (AstroCLIP's cross-modal model itself is not built
or used) and goes straight to the same nearest-neighbour retrieval, on 2,000
held-out spectra. No training happens, and the released weights retrieve
neighbours at a noticeably higher cosine similarity and lower median `|dz|`
than the from-scratch run, having seen far more than 16,000 spectra and 45
epochs. Needs the `pretrained` extra (`pip install astrospec[pretrained]`)
for `huggingface_hub`.

## `astroclip_similarity_search.ipynb`

Loads the full released `AstroClipModel` (image encoder + SpecFormer +
cross-attention projection heads, 370M parameters) via
`astrospec.pretrained.load_pretrained_astroclip`, the spectrum-side
counterpart to AstroLens's own `astroclip_similarity_search.ipynb`, which
does the image side of the same checkpoint. `model(spectrum, input_type="spectrum")`
gives the aligned 1024-d embedding directly, with no `Patchify` or pooling, and
the same nearest-neighbour retrieval as the other two SpecFormer notebooks
follows. Unlike them, needs the `astroclip` package and its `dinov2`
dependency (both `--no-deps` git installs, per the notebook's install cell),
so it cannot run from `astrospec[pretrained]` alone.
