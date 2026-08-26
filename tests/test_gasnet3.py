import pytest
import torch

import astrospec
from astrospec.models import GaSNet3

INPUT_DIM = 3522
OUTPUT_DIM = 4000
N_EIGENVECTORS = 10


@pytest.fixture
def model():
    return GaSNet3(input_dim=INPUT_DIM, output_dim=OUTPUT_DIM, n_eigenvectors=N_EIGENVECTORS)


def test_create_model_from_registry():
    model = astrospec.create_model(
        "gasnet3", input_dim=INPUT_DIM, output_dim=OUTPUT_DIM, n_eigenvectors=N_EIGENVECTORS
    )
    assert isinstance(model, GaSNet3)


def test_registry_defaults_match_omnispectrum_training_config():
    model = astrospec.create_model("gasnet3")
    assert model.eigenvectors.shape == (20, 9413)  # (n_eigenvectors, output_dim)


@pytest.mark.parametrize("shape", [(2, INPUT_DIM), (2, 1, INPUT_DIM)])
def test_forward_shape(model, shape):
    recon, coef = model(torch.randn(*shape))
    assert recon.shape == (2, OUTPUT_DIM)
    assert coef.shape == (2, N_EIGENVECTORS)


def test_coefficients_are_normalized(model):
    _, coef = model(torch.randn(2, INPUT_DIM))
    torch.testing.assert_close(coef.sum(dim=-1), torch.ones(2))


def test_backward(model):
    recon, _ = model(torch.randn(2, INPUT_DIM))
    recon.sum().backward()
    assert all(p.grad is not None for p in model.parameters())


def test_state_dict_round_trip(model):
    other = GaSNet3(input_dim=INPUT_DIM, output_dim=OUTPUT_DIM, n_eigenvectors=N_EIGENVECTORS)
    other.load_state_dict(model.state_dict())

    model.eval()
    other.eval()
    x = torch.randn(2, INPUT_DIM)
    torch.testing.assert_close(model(x)[0], other(x)[0])
