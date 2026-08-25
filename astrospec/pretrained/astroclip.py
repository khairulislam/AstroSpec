"""Loader for the released AstroCLIP checkpoint (Polymathic AI), via the
`astroclip` package.

Source: https://huggingface.co/polymathic-ai/astroclip (Parker et al. 2024,
https://doi.org/10.1093/mnras/stae1450), license MIT (Polymathic AI).
AstroCLIP CLIP-aligns a 302M-parameter DINOv2 image encoder (astrodino) with
a 43M-parameter masked-modeling spectrum transformer (specformer, the
released weights :func:`astrospec.pretrained.specformer.load_pretrained_specformer`
loads on its own) via cross-attention projection heads. The full cross-modal
model is too large and dependency-heavy to reimplement as a native module the
way this library's other models are. This is instead a thin wrapper around
the `astroclip` package, imported lazily so it is not a core runtime
dependency. Install it per the AstroCLIP README
(https://github.com/PolymathicAI/AstroCLIP#installation). Its dinov2 and
astroclip installs both require `--no-deps` git installs, so it cannot be
expressed as a single `pip install astrospec[...]` extra.
"""

__all__ = ["load_pretrained_astroclip"]

REPO_ID = "polymathic-ai/astroclip"
CKPT_FILENAME = "astroclip.ckpt"


def load_pretrained_astroclip(device: str = "cpu"):
    """Load the released AstroCLIP checkpoint.

    Returns an `astroclip.models.AstroClipModel` instance. Call
    `model(spectrum, input_type="spectrum")` for the aligned spectrum
    embedding. `spectrum` is a raw, un-patched `(B, L, 1)` flux sequence on
    AstroCLIP's native DESI EDR grid (7,781 pixels, 3,600-9,824 A, 0.80
    A/pix); `SpecFormer.preprocess` handles standardization and slicing into
    patches internally. Call `model(image, input_type="image")` for the
    image branch; see https://github.com/PolymathicAI/AstroCLIP for image
    preprocessing (144x144 center crop, `decals_to_rgb`).
    """
    from astroclip.models import AstroClipModel
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo_id=REPO_ID, filename=CKPT_FILENAME)
    return AstroClipModel.load_from_checkpoint(checkpoint_path=path).to(device).eval()
