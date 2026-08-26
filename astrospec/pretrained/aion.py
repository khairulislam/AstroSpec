"""Loader for a released AION-1 checkpoint (Polymathic AI), via the
`aion` package.

Source: https://huggingface.co/polymathic-ai/aion-base (Parker et al. 2025,
https://arxiv.org/abs/2510.17960), license MIT (Polymathic AI). AION-1 is a
two-stage model -- 39 per-modality tokenizers feeding a shared
encoder-decoder transformer over an astronomical masked-modeling objective --
too large and dependency-heavy to reimplement as a single native module the
way this library's other models are. This is instead a thin wrapper around
the `aion` package (`pip install astrospec[aion]`), imported lazily so it is
not a core runtime dependency, mirroring `astrolens.pretrained.aion` on the
image side.
"""

__all__ = ["load_pretrained_aion"]

VARIANTS = ("aion-base", "aion-large", "aion-xlarge")


def load_pretrained_aion(variant: str = "aion-base", device: str = "cpu"):
    """Load a released AION-1 checkpoint and its codec manager.

    Returns `(model, codec_manager)`. `model` is an `aion.model.AION`
    instance (`.encode()` for embeddings, `.forward()` for generative
    predictions); `codec_manager` turns typed `aion.modalities` objects into
    the token dict AION expects. The spectrum modalities,
    `aion.modalities.SDSSSpectrum` and `DESISpectrum`, both take
    `(flux, ivar, mask, wavelength)`, this library's own forward-contract
    keys, on the survey's native (un-patched, un-resampled) pixel grid. See
    https://github.com/PolymathicAI/AION and
    `examples/aion_spectrum_embeddings.ipynb` for a worked example.
    """
    from aion.codecs import CodecManager
    from aion.model import AION

    model = AION.from_pretrained(f"polymathic-ai/{variant}").to(device).eval()
    codec_manager = CodecManager(device=device)
    return model, codec_manager
