# AstroSpec

Spectroscopic models for astronomy under one registry and a common PyTorch
interface. The models cover spectral classification, redshift estimation,
stellar property estimation, and self-supervised representation learning on
native instrument grids. Each is a self-contained implementation, with
pretrained weights where available.

## Table of contents

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
`ivar`, `mask`, and `lsf_sigma` they consume. AstroSpec standardizes those
names, not the preprocessing. Surveys differ in grid, calibration, masking,
and resolution, and the library does not hide that.

Transformer models here consume patches rather than pixels, so the library
ships `astrospec.data.Patchify`. Those models will not run without it. The
caller handles everything else about loading and preprocessing. Apply
`Patchify` with the same settings to every per-pixel quantity and they stay
aligned; it also reports which patches are padding, so a model can mask
them.

```python
from astrospec.data import Patchify

patches, valid = Patchify(patch_size=20)(flux)
```

## Models

Each model is a plain `nn.Module` in `astrospec/models/`, one paper per module.
Each class docstring documents its constructor arguments and shapes.

### GalSpecNet

A 1-D convolutional classifier for optical spectra
([Wu et al. 2023, MNRAS 527:1163](https://doi.org/10.1093/mnras/stad2913)).
Stacked convolutions with interleaved max-pooling compress the flux sequence,
and a flat MLP maps the final activations to class logits. The head fixes the
grid, so resample spectra to a common length first. Consumes `flux`.

### SpectrumEncoder (spender)

The spectrum encoder of spender
([Melchior et al. 2023, AJ 166:74](https://doi.org/10.3847/1538-3881/ace0ff)),
after [Serra et al. 2018](https://arxiv.org/abs/1805.03908). A convolutional
stack splits its final channels into attention values and keys; the keys are
softmaxed over the pixel axis and pool the values into a single vector, which
an MLP compresses to a latent. That pooling makes the encoder
length-agnostic: spectra on different grids yield the same latent shape.
Consumes `flux`.

### SpecFormer

The spectrum tower of AstroCLIP
([Parker et al. 2024, MNRAS 531:4990](https://doi.org/10.1093/mnras/stae1450)).
Flux patches enter through a linear embedding, a learned embedding of the patch
index supplies position, and pre-norm transformer blocks produce one token per
patch. Pretraining reconstructs masked patches; `forward_features` returns the
tokens used downstream. Position is the patch index rather than `wavelength`,
so patch the dataset on a consistent grid. Consumes `flux`, patched.

### AstroPT

A GPT-style causal transformer over spectral patches
([Smith et al. 2024, arXiv:2405.14930](https://arxiv.org/abs/2405.14930),
[Smith42/astroPT](https://github.com/Smith42/astroPT)). Two tokenizers embed
the patches and the wavelengths of the same pixels, and the model sums the two,
so position is continuous and physical rather than a rank-indexed lookup, and
spectra on different grids stay comparable. Causal blocks let each patch attend
only to its predecessors; pretraining predicts the next patch. The multimodal
chaining of spectra onto this backbone is the
[Euclid Q1 follow-up](https://arxiv.org/abs/2503.15312). Consumes `flux` and
`wavelength`, both patched.

## Pretrained weights

`astrospec.pretrained` loads released checkpoints into the matching model,
with the optional `huggingface_hub` dependency (`pip install astrospec[pretrained]`).

| Model | Source | Params | Modality |
|---|---|---|---|

## Examples

Plain PyTorch notebooks in [`examples/`](examples/), added alongside the models
they use. Both read SDSS spectra from a local MultimodalUniverse HDF5 tree or,
failing that, stream them from the Hub.

| Notebook | What it does |
|---|---|
| [`sdss_classification.ipynb`](examples/sdss_classification.ipynb) | Trains GalSpecNet to separate GALAXY / QSO / STAR |
| [`specformer_pretraining.ipynb`](examples/specformer_pretraining.ipynb) | Pretrains SpecFormer on masked patches, then retrieves nearest neighbours in its embedding space |

## Resources

* [MultimodalUniverse](https://github.com/MultimodalUniverse/MultimodalUniverse): 100 TB of astronomical data in a single ML-ready format, including the spectroscopic surveys used here.

## Citations

Please cite this library (see `CITATION.cff`) along with the original works
behind the models you use.

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

@article{parker2024astroclip,
  title     = {AstroCLIP: a cross-modal foundation model for galaxies},
  volume    = {531},
  number    = {4},
  pages     = {4990--5011},
  journal   = {Monthly Notices of the Royal Astronomical Society},
  author    = {Parker, Liam and Lanusse, Francois and Golkar, Siavash and Sarra, Leopoldo and Cranmer, Miles and Bietti, Alberto and Eickenberg, Michael and Krawezik, Geraud and McCabe, Michael and Morel, Rudy and others},
  year      = {2024},
  doi       = {10.1093/mnras/stae1450}
}

@article{smith2024astropt,
  title         = {AstroPT: Scaling Large Observation Models for Astronomy},
  author        = {Smith, Michael J. and Roberts, Ryan J. and Angeloudi, Eirini and Huertas-Company, Marc},
  year          = {2024},
  eprint        = {2405.14930},
  archivePrefix = {arXiv},
  primaryClass  = {astro-ph.IM}
}
```
