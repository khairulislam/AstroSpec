import pytest
import torch

import astrospec
from astrospec.models import GalSpecNet

INPUT_LENGTH = 3522
NUM_CLASSES = 3


@pytest.fixture
def model():
    return GalSpecNet(input_length=INPUT_LENGTH, num_classes=NUM_CLASSES)


def test_create_model_from_registry():
    model = astrospec.create_model(
        "galspecnet", input_length=INPUT_LENGTH, num_classes=NUM_CLASSES
    )
    assert isinstance(model, GalSpecNet)


@pytest.mark.parametrize("shape", [(2, INPUT_LENGTH), (2, 1, INPUT_LENGTH)])
def test_forward_shape(model, shape):
    assert model(torch.randn(*shape)).shape == (2, NUM_CLASSES)


def test_backward(model):
    model(torch.randn(2, INPUT_LENGTH)).sum().backward()
    assert all(p.grad is not None for p in model.parameters())


def test_state_dict_round_trip(model):
    other = GalSpecNet(input_length=INPUT_LENGTH, num_classes=NUM_CLASSES)
    other.load_state_dict(model.state_dict())

    model.eval()
    other.eval()
    x = torch.randn(2, INPUT_LENGTH)
    torch.testing.assert_close(model(x), other(x))


def test_input_length_too_short():
    with pytest.raises(ValueError, match="too short"):
        GalSpecNet(input_length=16, num_classes=NUM_CLASSES)
