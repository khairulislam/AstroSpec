# AstroSpec - Unified Spectral Library for Astronomy

A unified library of spectroscopic models for astronomy: spectral
classification, redshift estimation, stellar property estimation, and
self-supervised representation learning on native instrument grids, each a
self-contained implementation behind one registry and a common PyTorch
interface, with pretrained weights where available.

## Table of Contents

- [Usage](#usage)
- [Models](#models)
- [Pretrained weights](#pretrained-weights)
- [Examples](#examples)
- [Resources](#resources)
- [Citations](#citations)

## Usage

```bash
pip install -e .
```

```python
import astrospec

astrospec.list_models()
model = astrospec.create_model("galspecnet", input_length=3522, num_classes=3)
```

Spectra carry more than flux. Models declare which of `flux`, `wavelength`,
`ivar`, `mask`, and `lsf_sigma` they consume; fixed-grid models say so in their
section below. AstroSpec standardizes those names, not the preprocessing —
surveys differ in grid, calibration, masking, and resolution, and the library
does not hide that.

## Models

### GalSpecNet

A 1-D convolutional classifier for optical spectra
([Wu et al. 2023, MNRAS 527:1163](https://doi.org/10.1093/mnras/stad2913)).
Stacked convolutions with interleaved max-pooling compress the flux sequence,
and a flat MLP maps the final activations to class logits. Introduced for
spectral classification of SDSS galaxies, it serves as the supervised baseline
against which the self-supervised encoders here are measured.

Consumes `flux` only. Fixed-grid: every spectrum must be resampled to
`input_length` pixels, since the head is a flat MLP over the convolutional
output.

| Argument | Default | Meaning |
|---|---|---|
| `input_length` | — | pixels per spectrum |
| `num_classes` | — | output dimension (logits, or regression targets) |
| `conv_channels` | `(1, 64, 64, 32, 32)` | channel widths, starting at the raw flux channel |
| `kernel_size` | `3` | convolution width, no padding |
| `mp_kernel_size` | `4` | max-pool width, after every convolution but the last |
| `dropout` | `0.1` | dropout before the head |
| `n_hidden` | `(256, 64, 16)` | hidden widths of the MLP head |

```python
from astrospec.models import GalSpecNet

model = GalSpecNet(input_length=3522, num_classes=3)
logits = model(flux)  # (B, L) or (B, 1, L) -> (B, num_classes)
```

### SpectrumEncoder (spender)

The spectrum encoder from spender
([Melchior et al. 2023, AJ 166:74](https://doi.org/10.3847/1538-3881/ace0ff)),
itself a modification of [Serra et al. 2018](https://arxiv.org/abs/1805.03908).
A stack of convolutions splits its final channels into attention values and
keys; the keys are softmaxed over the pixel axis and pool the values into one
vector per spectrum, which an MLP compresses to `n_latent`. That pooling is
what makes the encoder length-agnostic — spectra on different grids yield the
same latent shape — so it is the natural counterpart to GalSpecNet's flat head.

Consumes `flux` only.

| Argument | Default | Meaning |
|---|---|---|
| `n_latent` | — | latent dimension |
| `filters` | `(8, 16, 16, 32)` | convolution output channels; the last must be even |
| `sizes` | `(5, 10, 20, 40)` | kernel width per convolution, and the pooling width between them |
| `n_hidden` | `(32, 32)` | hidden widths of the MLP |
| `act` | `PReLU` per hidden layer | MLP activations |
| `dropout` | `0.0` | dropout in the convolutions and the MLP |

```python
from astrospec.models import SpectrumEncoder

model = SpectrumEncoder(n_latent=6)
latent = model(flux)  # (B, L) or (B, 1, L) -> (B, n_latent)
```

The defaults are the lightweight variant used as a baseline in OmniSpectrum;
spender's published encoder is `filters=(128, 256, 512)`, `sizes=(5, 11, 21)`,
`n_hidden=(128, 64, 32)`.

## Pretrained weights

`astrospec.pretrained` loads released checkpoints into the matching model,
with the optional `huggingface_hub` dependency (`pip install astrospec[pretrained]`).

| Model | Source | Params | Modality |
|---|---|---|---|

## Examples

*Plain PyTorch scripts and notebooks, added alongside the models they use.*

## Resources

* [MultimodalUniverse](https://github.com/MultimodalUniverse/MultimodalUniverse): 100 TB of astronomical data in a single ML-ready format, including the spectroscopic surveys used here.

## Citations

If you find this library useful in your research, please consider citing it
(see `CITATION.cff`), along with the original works behind the included models.

```bibtex
@article{wu2024galaxy,
  title     = {Galaxy spectral classification and feature analysis based on convolutional neural network},
  volume    = {527},
  number    = {1},
  pages     = {1163--1176},
  journal   = {Monthly Notices of the Royal Astronomical Society},
  author    = {Wu, Ying and Tao, Yihan and Fan, Dongwei and Cui, Chenzhou and Zhang, Yanxia},
  year      = {2023},
  doi       = {10.1093/mnras/stad2913}
}

@article{melchior2023autoencoding,
  title     = {Autoencoding Galaxy Spectra. I. Architecture},
  volume    = {166},
  number    = {2},
  pages     = {74},
  journal   = {The Astronomical Journal},
  author    = {Melchior, Peter and Liang, Yan and Hahn, ChangHoon and Goulding, Andy},
  year      = {2023},
  doi       = {10.3847/1538-3881/ace0ff}
}
```
