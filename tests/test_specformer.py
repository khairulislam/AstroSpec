import pytest
import torch

import astrospec
from astrospec.models import SpecFormer

INPUT_DIM, EMBED_DIM, MAX_LEN = 22, 96, 800
SEQ_LEN = 32


@pytest.fixture
def model():
    return SpecFormer(
        input_dim=INPUT_DIM,
        embed_dim=EMBED_DIM,
        num_layers=2,
        num_heads=6,
        max_len=MAX_LEN,
        dropout=0.0,
    ).eval()


def test_create_model_from_registry():
    model = astrospec.create_model("specformer", embed_dim=EMBED_DIM, num_layers=2)
    assert isinstance(model, SpecFormer)
    assert model.input_dim == 22 and model.max_len == 800


def test_forward_shape(model):
    x = torch.randn(2, SEQ_LEN, INPUT_DIM)
    assert model(x).shape == (2, SEQ_LEN, INPUT_DIM)
    assert model.forward_features(x).shape == (2, SEQ_LEN, EMBED_DIM)


def test_backward(model):
    model.train()
    model(torch.randn(2, SEQ_LEN, INPUT_DIM)).sum().backward()
    assert all(p.grad is not None for p in model.parameters())


def test_state_dict_round_trip(model):
    other = SpecFormer(
        input_dim=INPUT_DIM,
        embed_dim=EMBED_DIM,
        num_layers=2,
        num_heads=6,
        max_len=MAX_LEN,
        dropout=0.0,
    ).eval()
    other.load_state_dict(model.state_dict())

    x = torch.randn(2, SEQ_LEN, INPUT_DIM)
    torch.testing.assert_close(model(x), other(x))


def test_padded_patches_do_not_change_valid_tokens(model):
    x = torch.randn(2, SEQ_LEN, INPUT_DIM)
    valid = torch.ones(2, SEQ_LEN, dtype=torch.bool)
    valid[:, -3:] = False

    perturbed = x.clone()
    perturbed[:, -3:] = torch.randn(2, 3, INPUT_DIM) * 100

    torch.testing.assert_close(
        model.forward_features(x, valid=valid)[:, :-3],
        model.forward_features(perturbed, valid=valid)[:, :-3],
    )


def test_sequence_longer_than_max_len_rejected(model):
    with pytest.raises(ValueError, match="max_len"):
        model(torch.randn(1, MAX_LEN + 1, INPUT_DIM))
