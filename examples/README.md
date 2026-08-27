# Examples

Optional, self-contained notebooks. Each installs its own extra dependencies in
its first cell; none of this is required to use `astrospec` itself.

## Data

Every notebook imports `utils`, which loads a local `.env` in this directory on
import (`utils/__init__.py`), so `ASTROSPEC_SDSS_ROOT` and `ASTROSPEC_DESI_ROOT`
only need setting once, in one file, rather than exporting them in every shell
that launches a kernel:

```bash
# examples/.env, gitignored
ASTROSPEC_SDSS_ROOT=/path/to/sdss   # a directory of healpix=*/ shards
ASTROSPEC_DESI_ROOT=/path/to/desi/edr_sv3
```

An exported shell variable still works too; `.env` only fills in what the
environment does not already have. A path that is set but wrong -- a typo, a
moved directory -- raises `FileNotFoundError` rather than silently falling
back to streaming, since that is almost never what a set path means.

Four notebooks read SDSS spectra through `utils/sdss.py`, from the local tree
when `ASTROSPEC_SDSS_ROOT` points at one and streaming
[`MultimodalUniverse/sdss`](https://huggingface.co/datasets/MultimodalUniverse/sdss)
otherwise.

The two copies are not identical. The Hub copy carries the spectra and `Z` but
not the pipeline `CLASS`, so `galspecnet_source_classification_on_sdss.ipynb`
needs a local tree.

`utils/sdss.py` also resamples: SDSS spectra share a log-lambda pixel spacing but
cover slightly different wavelength windows, so their pixel counts differ from
object to object, and the fixed-grid models cannot see that variation. AstroSpec
ships no resampling of its own. The grid, the interpolation, and the
normalization are properties of a dataset, not of a model.

Every loader normalizes through one shared routine, `utils/preprocess.py`, which
reproduces OmniSpectrum's `SpectrumPreprocessor` as configured for SDSS:
sigma-clip the outlier pixels so cosmic rays and sky residuals do not set the
scale, divide each spectrum by its own mean flux so the continuum sits near
zero once 1 is subtracted, then compress with `arcsinh`, which handles emission
lines orders of magnitude above the continuum while staying defined for the
negative values sky subtraction leaves. Its `norm_floor` guards the scale factor
against faint and sky-dominated spectra and assumes survey flux units, so
`utils/astrom3.py` overrides it for a release that ships flux pre-scaled to
~1e-3.

`astroclip_similarity_search.ipynb` and `astropt_pretraining.ipynb` read
[MultimodalUniverse](https://github.com/MultimodalUniverse/MultimodalUniverse)
DESI EDR SV3 spectra instead, through `utils/desi.py`'s `load_spectra`, the
grid AstroCLIP's spectrum encoder was actually trained on. Every spectrum in
this release already sits on the same 7,781-pixel grid, so `utils/desi.py`
does no resampling. `aion_spectrum_embeddings.ipynb` instead uses
`desi.load_native`, which also reads per-pixel inverse variance, the
bad-pixel mask, and per-row wavelength, since AION's own spectrum codec wants
those directly; the Hub release's `spectrum` column carries `ivar`, `mask`,
and `lambda` alongside `flux`, so this loader streams from the Hub exactly
like `load_spectra` when there is no local tree.

`spender_redshift_regression_on_sdss.ipynb` uses the same SDSS/DESI sources and
their pipeline `Z` values as regression targets. Like the class-label example it
needs a local tree, because it selects a single pipeline class and only the local
shards carry the class column (`CLASS` in SDSS, `SPECTYPE` in DESI).

`galspecnet_classification_on_astrom3.ipynb` streams
[`AstroMLCore/AstroM3Processed`](https://huggingface.co/datasets/AstroMLCore/AstroM3Processed).
It uses only the processed LAMOST spectrum, not the accompanying light curve or
metadata, and honors AstroM3's published train/validation/test split. The
processed release needs no `trust_remote_code`.

## `galspecnet_source_classification_on_sdss.ipynb`

Trains `GalSpecNet` to separate the SDSS pipeline classes GALAXY / QSO / STAR on
6,000 spectra drawn in the survey's natural (imbalanced) class proportions,
split 70/10/20, with training-split inverse-frequency class weights in the
loss. Reports a per-class report and confusion matrix. The shortest end-to-end
path through the library: load, resample, `create_model`, train, evaluate.

## `spender_redshift_regression_on_sdss.ipynb`

Trains `SpectrumEncoder`, spender's CNN + attention-pooling encoder, with
`n_latent=1` to regress redshift from 50,000 SDSS spectra by default; set
`survey = "desi"` for DESI EDR SV3 on its native 7,781-pixel grid. The sample is
restricted to the `GALAXY` class, since redshift is read from different features
in stars, galaxies and quasars, to redshifts the pipeline did not flag
(`max_zwarning=0`), and to `0.01 < Z <= 0.5` (the upper bound is spender's own
`z_max` for SDSS, 0.8 for DESI; the lower one keeps log-space targets from
carrying runaway outliers). A deliberate 20-epoch draft run plots train and
validation learning curves alongside held-out predictions and residuals. It is
still not a redshift benchmark: the target distribution is the survey's own, and
the per-object `Z_ERR` is unused.

## `galspecnet_classification_on_astrom3.ipynb`

Trains GalSpecNet on the ten AstroM3 variable-star classes from the LAMOST flux
channel. It uses the publisher's full `full_42` train/validation/test split and
training-split inverse-frequency weights to handle class imbalance, without
requiring the local multi-TB survey tree. Its wider GalSpecNet baseline trains
for 30 epochs with cosine learning-rate decay, then reports per-class results
and a confusion matrix.

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

## `aion_spectrum_embeddings.ipynb`

Loads the released `aion-base` checkpoint via
`astrospec.pretrained.load_pretrained_aion`, the spectrum-side counterpart to
AstroLens's `aion_embeddings.ipynb`, which does the image side of the same
model. Builds an `aion.modalities.DESISpectrum` from native
`(flux, ivar, mask, wavelength)`, encodes it through AION's `CodecManager`,
mean-pools the 273 resulting tokens, and runs the same nearest-neighbour
retrieval as the SpecFormer and AstroCLIP notebooks, on 2,000 spectra. Unlike
`astroclip_similarity_search.ipynb`, the `aion` extra
(`pip install astrospec[aion]`) is a normal package install, no `--no-deps`
git installs needed.

## `astropt_pretraining.ipynb`

Pretrains `AstroPT` on 16,000 unlabelled DESI spectra with its own
self-supervised objective: causal next-patch prediction, a Huber loss between
each patch's prediction and the next patch, over positions where both are
valid. There is no public spectra-pretrained AstroPT checkpoint to load
instead (`Smith42/astroPT`'s release is image-only), so this notebook is the
model's pretraining step, not a loader. `forward_features` gives one causal
hidden state per patch; since each has only seen its own position and earlier
ones, the last valid patch's state, not the mean, is the read-out used for
nearest-neighbour retrieval against held-out redshift, as in
`specformer_pretraining.ipynb`.

## `astropt_classification_on_sdss.ipynb`

Trains `AstroPT` and a `CrossAttentionHead` fully supervised, end to end from
a random init, to separate the same SDSS GALAXY / QSO / STAR classes as
`galspecnet_source_classification_on_sdss.ipynb`, on the same 6,000-spectrum
sample and split. It exists because there is no pretrained AstroPT checkpoint
to attach a head to; training the full encoder from scratch on 4,200 labelled
spectra is a less data-efficient use of it than pretraining plus fine-tuning
would be, but it gives a direct comparison between a fixed-grid CNN and a
wavelength-positioned transformer on identical data.
