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
model = astrospec.create_model("galspecnet", num_classes=3)
```

Spectra carry more than flux. Models declare which of `flux`, `wavelength`,
`ivar`, `mask`, and `lsf_sigma` they consume; fixed-grid models say so in their
section below. AstroSpec standardizes those names, not the preprocessing —
surveys differ in grid, calibration, masking, and resolution, and the library
does not hide that.

## Models

*One section per model, added as each is implemented: source paper, input
shape, configurable arguments, and the use case that motivated its inclusion.*

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
```
