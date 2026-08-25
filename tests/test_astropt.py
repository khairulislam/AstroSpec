import pytest
import torch

import astrospec
from astrospec.models import AstroPT

INPUT_DIM, EMBED_DIM, SEQ_LEN = 20, 64, 16


@pytest.fixture
def model():
    return AstroPT(
        input_dim=INPUT_DIM, embed_dim=EMBED_DIM, num_layers=2, num_heads=8
    ).eval()


@pytest.fixture
def batch():
    patches = torch.randn(2, SEQ_LEN, INPUT_DIM)
    wavelength = (
        torch.linspace(3600.0, 9800.0, SEQ_LEN * INPUT_DIM)
        .view(1, SEQ_LEN, INPUT_DIM)
        .expand(2, -1, -1)
    )
    return patches, wavelength


def test_create_model_from_registry():
    model = astrospec.create_model("astropt", embed_dim=EMBED_DIM, num_layers=2)
    assert isinstance(model, AstroPT)
    assert model.input_dim == 32


def test_forward_shape(model, batch):
    patches, wavelength = batch
    assert model(patches, wavelength).shape == (2, SEQ_LEN, INPUT_DIM)
    assert model.forward_features(patches, wavelength).shape == (2, SEQ_LEN, EMBED_DIM)


def test_backward(model, batch):
    model.train()
    model(*batch).sum().backward()
    assert all(p.grad is not None for p in model.parameters())


def test_state_dict_round_trip(model, batch):
    other = AstroPT(
        input_dim=INPUT_DIM, embed_dim=EMBED_DIM, num_layers=2, num_heads=8
    ).eval()
    other.load_state_dict(model.state_dict())
    torch.testing.assert_close(model(*batch), other(*batch))


def test_attention_is_causal(model, batch):
    patches, wavelength = batch
    perturbed = patches.clone()
    perturbed[:, 8] = torch.randn(INPUT_DIM) * 50

    before = model.forward_features(patches, wavelength)
    after = model.forward_features(perturbed, wavelength)

    torch.testing.assert_close(before[:, :8], after[:, :8])
    assert not torch.allclose(before[:, 9:], after[:, 9:])


def test_padded_patches_do_not_change_valid_tokens(model, batch):
    patches, wavelength = batch
    valid = torch.ones(2, SEQ_LEN, dtype=torch.bool)
    valid[:, -3:] = False

    perturbed = patches.clone()
    perturbed[:, -3:] = torch.randn(2, 3, INPUT_DIM) * 100

    torch.testing.assert_close(
        model.forward_features(patches, wavelength, valid=valid)[:, :-3],
        model.forward_features(perturbed, wavelength, valid=valid)[:, :-3],
    )


def test_wavelength_changes_the_representation(model, batch):
    patches, wavelength = batch
    shifted = wavelength + 500.0
    assert not torch.allclose(
        model.forward_features(patches, wavelength),
        model.forward_features(patches, shifted),
    )
