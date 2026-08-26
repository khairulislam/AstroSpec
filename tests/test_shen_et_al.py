import pytest
import torch

import astrospec
from astrospec.models import ShenSpectralTokenizer
from astrospec.models.shen_et_al import sinusoidal_wavelength_encoding

PATCH_SIZE, EMBED_DIM, SEQ_LEN = 16, 32, 12


@pytest.fixture
def model():
    return ShenSpectralTokenizer(
        patch_size=PATCH_SIZE,
        embed_dim=EMBED_DIM,
        num_enc_layers=2,
        num_dec_layers=2,
        num_heads=4,
        dropout=0.0,
    ).eval()


@pytest.fixture
def batch():
    flux = torch.randn(2, SEQ_LEN, PATCH_SIZE)
    wavelength = (
        torch.linspace(3600.0, 9800.0, SEQ_LEN * PATCH_SIZE)
        .view(1, SEQ_LEN, PATCH_SIZE)
        .expand(2, -1, -1)
    )
    ivar = torch.rand(2, SEQ_LEN, PATCH_SIZE) + 0.5
    return flux, wavelength, ivar


def test_sinusoidal_wavelength_encoding_shape_and_unit_norm():
    wavelength = torch.linspace(3600.0, 9800.0, 40)
    enc = sinusoidal_wavelength_encoding(wavelength, dim=16)
    assert enc.shape == (40, 16)

    # each (sin, cos) pair from the same frequency lies on the unit circle
    sin, cos = enc[..., 0::2], enc[..., 1::2]
    torch.testing.assert_close(sin.pow(2) + cos.pow(2), torch.ones_like(sin))

    with pytest.raises(ValueError, match="even"):
        sinusoidal_wavelength_encoding(wavelength, dim=15)


def test_create_model_from_registry():
    model = astrospec.create_model("shen_et_al", embed_dim=EMBED_DIM, num_heads=4)
    assert isinstance(model, ShenSpectralTokenizer)
    assert model.patch_size == 32


def test_forward_shape(model, batch):
    flux, wavelength, ivar = batch
    assert model(flux, wavelength, ivar=ivar).shape == (2, SEQ_LEN, PATCH_SIZE)
    assert model.forward_features(flux, wavelength, ivar=ivar).shape == (2, SEQ_LEN, EMBED_DIM)


def test_ivar_is_optional(model, batch):
    flux, wavelength, _ = batch
    assert model(flux, wavelength).shape == (2, SEQ_LEN, PATCH_SIZE)


def test_backward(model, batch):
    model.train()
    model(*batch).sum().backward()
    assert all(p.grad is not None for p in model.parameters())


def test_state_dict_round_trip(model, batch):
    other = ShenSpectralTokenizer(
        patch_size=PATCH_SIZE,
        embed_dim=EMBED_DIM,
        num_enc_layers=2,
        num_dec_layers=2,
        num_heads=4,
        dropout=0.0,
    ).eval()
    other.load_state_dict(model.state_dict())
    torch.testing.assert_close(model(*batch), other(*batch))


def test_padded_patches_do_not_change_valid_tokens(model, batch):
    flux, wavelength, ivar = batch
    valid = torch.ones(2, SEQ_LEN, dtype=torch.bool)
    valid[:, -3:] = False

    perturbed_flux = flux.clone()
    perturbed_flux[:, -3:] = torch.randn(2, 3, PATCH_SIZE) * 100

    torch.testing.assert_close(
        model.forward_features(flux, wavelength, ivar=ivar, valid=valid)[:, :-3],
        model.forward_features(perturbed_flux, wavelength, ivar=ivar, valid=valid)[:, :-3],
    )


def test_wavelength_changes_the_representation(model, batch):
    flux, wavelength, ivar = batch
    shifted = wavelength + 500.0
    assert not torch.allclose(
        model.forward_features(flux, wavelength, ivar=ivar),
        model.forward_features(flux, shifted, ivar=ivar),
    )


def test_output_wavelength_reconstructs_a_different_grid(model, batch):
    flux, wavelength, ivar = batch
    shifted = wavelength + 500.0
    assert not torch.allclose(
        model(flux, wavelength, ivar=ivar, output_wavelength=wavelength),
        model(flux, wavelength, ivar=ivar, output_wavelength=shifted),
    )
