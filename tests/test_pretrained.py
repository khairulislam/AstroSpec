import torch

import astrospec
from astrospec.pretrained.specformer import STATE_DICT_PREFIX, load_pretrained_specformer


def _fake_checkpoint():
    """A synthetic checkpoint matching the released layout, small enough to
    build in-memory instead of downloading a real one."""
    reference = astrospec.create_model("specformer")
    state_dict = {STATE_DICT_PREFIX + k: v for k, v in reference.state_dict().items()}
    # extra keys from the image tower and cross-attention heads, ignored by the loader
    state_dict["image_encoder.backbone.norm.weight"] = torch.randn(4)
    return {"state_dict": state_dict}


def test_load_pretrained_specformer_maps_checkpoint_keys(tmp_path, monkeypatch):
    checkpoint = _fake_checkpoint()
    path = tmp_path / "astroclip.ckpt"
    torch.save(checkpoint, path)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", lambda repo_id, filename: str(path))

    model = load_pretrained_specformer(repo_id="fake/repo", filename="fake.ckpt")

    reference_state = {
        k[len(STATE_DICT_PREFIX) :]: v
        for k, v in checkpoint["state_dict"].items()
        if k.startswith(STATE_DICT_PREFIX)
    }
    for name, param in model.state_dict().items():
        torch.testing.assert_close(param, reference_state[name])

    patches = torch.randn(2, 5, model.input_dim)
    model(patches)  # loaded weights run without shape errors
