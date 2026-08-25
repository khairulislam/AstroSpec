import pytest
import torch

import astrospec
from astrospec.models import SpectrumEncoder

N_LATENT = 6


@pytest.fixture
def model():
    return SpectrumEncoder(n_latent=N_LATENT)


def test_create_model_from_registry():
    model = astrospec.create_model("spectrum_encoder", n_latent=N_LATENT)
    assert isinstance(model, SpectrumEncoder)


@pytest.mark.parametrize("shape", [(2, 3522), (2, 1, 3522)])
def test_forward_shape(model, shape):
    assert model(torch.randn(*shape)).shape == (2, N_LATENT)


def test_latent_shape_is_length_agnostic(model):
    # attention pooling collapses the pixel axis, so any grid gives one latent
    assert model(torch.randn(2, 3522)).shape == model(torch.randn(2, 7781)).shape


def test_backward(model):
    model(torch.randn(2, 3522)).sum().backward()
    assert all(p.grad is not None for p in model.parameters())


def test_state_dict_round_trip(model):
    other = SpectrumEncoder(n_latent=N_LATENT)
    other.load_state_dict(model.state_dict())

    model.eval()
    other.eval()
    x = torch.randn(2, 3522)
    torch.testing.assert_close(model(x), other(x))


def test_odd_final_filter_count_rejected():
    with pytest.raises(ValueError, match="even"):
        SpectrumEncoder(n_latent=N_LATENT, filters=(8, 15), sizes=(5, 10))
