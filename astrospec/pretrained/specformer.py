"""Loader for SpecFormer's released weights, published as part of AstroCLIP.

Source: https://huggingface.co/polymathic-ai/astroclip (Parker et al. 2024,
https://doi.org/10.1093/mnras/stae1450), license MIT (Polymathic AI).
AstroCLIP CLIP-aligns an image encoder with a SpecFormer spectrum encoder;
this loader takes only the ``spectrum_encoder.backbone.*`` weights out of the
released Lightning checkpoint and loads them into
:func:`astrospec.models.specformer.specformer` directly. The AstroCLIP model
itself (the image tower, the CLIP alignment) is not built here, so using the
pretrained spectrum tower needs no Lightning, dinov2, or the ``astroclip``
package. Requires the optional ``huggingface_hub`` dependency
(``pip install astrospec[pretrained]``), imported lazily so it is not a core
runtime dependency.
"""

import astrospec

__all__ = ["load_pretrained_specformer"]

REPO_ID = "polymathic-ai/astroclip"
CKPT_FILENAME = "astroclip.ckpt"
STATE_DICT_PREFIX = "spectrum_encoder.backbone."


def load_pretrained_specformer(
    device: str = "cpu",
    repo_id: str = REPO_ID,
    filename: str = CKPT_FILENAME,
):
    """Build a :class:`~astrospec.models.specformer.SpecFormer` with its released trained weights.

    The published configuration (6 layers, width 768, 22-number patches,
    ``max_len=800``) is used, matching the checkpoint exactly.
    """
    import torch
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo_id=repo_id, filename=filename)
    state_dict = torch.load(path, map_location="cpu", weights_only=False)["state_dict"]
    backbone = {
        k[len(STATE_DICT_PREFIX) :]: v
        for k, v in state_dict.items()
        if k.startswith(STATE_DICT_PREFIX)
    }

    model = astrospec.create_model("specformer")
    model.load_state_dict(backbone, strict=True)
    return model.to(device).eval()
