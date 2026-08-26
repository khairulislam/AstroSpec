"""Registry-wide contract: every sequence encoder exposes forward_features()
returning one token per position, (B, T, embed_dim). Models without a
meaningful token sequence (GalSpecNet, SpectrumEncoder, GaSNet-III -- see
each model's docstring) are outside this contract and excluded below.
"""

import pytest
import torch

import astrospec

EMBED_DIM, NUM_HEADS, SEQ_LEN = 16, 2, 6


def _patches(seq_len, patch_size):
    return torch.randn(2, seq_len, patch_size)


def _wavelength(seq_len, patch_size):
    return (
        torch.linspace(3600.0, 9800.0, seq_len * patch_size)
        .view(1, seq_len, patch_size)
        .expand(2, -1, -1)
    )


CASES = {
    "specformer": dict(
        kwargs=dict(input_dim=22, embed_dim=EMBED_DIM, num_layers=1, num_heads=NUM_HEADS, max_len=64),
        inputs=lambda: (_patches(SEQ_LEN, 22),),
    ),
    "astropt": dict(
        kwargs=dict(input_dim=16, embed_dim=EMBED_DIM, num_layers=1, num_heads=NUM_HEADS),
        inputs=lambda: (_patches(SEQ_LEN, 16), _wavelength(SEQ_LEN, 16)),
    ),
    "specpt": dict(
        kwargs=dict(
            input_len=256, embed_dim=EMBED_DIM, num_enc_layers=1, num_dec_layers=1, num_heads=NUM_HEADS
        ),
        inputs=lambda: (torch.randn(2, 256),),
    ),
    "specpt_redshift": dict(
        kwargs=dict(embed_dim=EMBED_DIM, num_enc_layers=1, num_heads=NUM_HEADS, num_mlp_blocks=1),
        inputs=lambda: (torch.randn(2, 256),),
    ),
    "shen_et_al": dict(
        kwargs=dict(
            patch_size=16, embed_dim=EMBED_DIM, num_enc_layers=1, num_dec_layers=1, num_heads=NUM_HEADS
        ),
        inputs=lambda: (_patches(SEQ_LEN, 16), _wavelength(SEQ_LEN, 16)),
    ),
}


@pytest.mark.parametrize("model_name", sorted(CASES))
def test_forward_features_returns_one_token_per_position(model_name):
    case = CASES[model_name]
    model = astrospec.create_model(model_name, **case["kwargs"]).eval()

    assert hasattr(model, "forward_features")
    tokens = model.forward_features(*case["inputs"]())

    assert tokens.ndim == 3
    assert tokens.shape[0] == 2
    assert tokens.shape[-1] == EMBED_DIM


def test_every_registered_model_is_accounted_for():
    """Models not in CASES must document why forward_features isn't meaningful."""
    excluded = {"galspecnet", "spectrum_encoder", "gasnet3"}
    assert set(astrospec.list_models()) == set(CASES) | excluded
