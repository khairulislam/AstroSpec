import pytest
import torch

import astrospec
from astrospec.models import SpecPTAutoencoder, SpecPTEncoder, SpecPTRedshift, SpecPTRedshiftHead

INPUT_LEN, EMBED_DIM, NUM_HEADS = 256, 32, 4
T = INPUT_LEN // 8


class TestSpecPTEncoder:
    @pytest.fixture
    def model(self):
        return SpecPTEncoder(embed_dim=EMBED_DIM, num_layers=2, num_heads=NUM_HEADS, dropout=0.0).eval()

    def test_forward_shape(self, model):
        flux = torch.randn(2, INPUT_LEN)
        assert model(flux).shape == (2, T, EMBED_DIM)

    def test_backward(self, model):
        model.train()
        model(torch.randn(2, INPUT_LEN)).sum().backward()
        assert all(p.grad is not None for p in model.parameters())

    def test_state_dict_round_trip(self, model):
        other = SpecPTEncoder(embed_dim=EMBED_DIM, num_layers=2, num_heads=NUM_HEADS, dropout=0.0).eval()
        other.load_state_dict(model.state_dict())
        flux = torch.randn(2, INPUT_LEN)
        torch.testing.assert_close(model(flux), other(flux))


class TestSpecPTAutoencoder:
    @pytest.fixture
    def model(self):
        return SpecPTAutoencoder(
            input_len=INPUT_LEN,
            embed_dim=EMBED_DIM,
            num_enc_layers=2,
            num_dec_layers=2,
            num_heads=NUM_HEADS,
            dropout=0.0,
        ).eval()

    def test_create_model_from_registry(self):
        model = astrospec.create_model("specpt", embed_dim=EMBED_DIM, num_heads=NUM_HEADS)
        assert isinstance(model, SpecPTAutoencoder)
        assert model.input_len == 7780

    def test_forward_shape(self, model):
        assert model(torch.randn(2, INPUT_LEN)).shape == (2, INPUT_LEN)

    def test_backward(self, model):
        model.train()
        model(torch.randn(2, INPUT_LEN)).sum().backward()
        assert all(p.grad is not None for p in model.parameters())

    def test_state_dict_round_trip(self, model):
        other = SpecPTAutoencoder(
            input_len=INPUT_LEN,
            embed_dim=EMBED_DIM,
            num_enc_layers=2,
            num_dec_layers=2,
            num_heads=NUM_HEADS,
            dropout=0.0,
        ).eval()
        other.load_state_dict(model.state_dict())
        flux = torch.randn(2, INPUT_LEN)
        torch.testing.assert_close(model(flux), other(flux))

    def test_odd_input_length_round_trips_to_the_same_length(self, model):
        flux = torch.randn(2, INPUT_LEN - 3)
        assert model(flux).shape == (2, INPUT_LEN - 3)


class TestSpecPTRedshiftHead:
    @pytest.fixture
    def model(self):
        return SpecPTRedshiftHead(
            embed_dim=EMBED_DIM, num_heads=NUM_HEADS, num_blocks=2, hidden_dim=16, dropout=0.0
        ).eval()

    def test_forward_shape(self, model):
        features = torch.randn(2, T, EMBED_DIM)
        assert model(features).shape == (2,)

    def test_output_is_non_negative(self, model):
        features = torch.randn(2, T, EMBED_DIM) * 10
        assert (model(features) >= 0).all()

    def test_backward(self, model):
        model.train()
        model(torch.randn(2, T, EMBED_DIM)).sum().backward()
        assert all(p.grad is not None for p in model.parameters())


class TestSpecPTRedshift:
    def test_create_model_from_registry(self):
        model = astrospec.create_model("specpt_redshift", embed_dim=EMBED_DIM, num_heads=NUM_HEADS)
        assert isinstance(model, SpecPTRedshift)

    def test_forward_shape(self):
        model = SpecPTRedshift(
            embed_dim=EMBED_DIM, num_enc_layers=2, num_heads=NUM_HEADS, num_mlp_blocks=2
        ).eval()
        assert model(torch.randn(2, INPUT_LEN)).shape == (2,)

    def test_freeze_encoder_excludes_it_from_backward(self):
        model = SpecPTRedshift(
            embed_dim=EMBED_DIM,
            num_enc_layers=2,
            num_heads=NUM_HEADS,
            num_mlp_blocks=2,
            freeze_encoder=True,
        )
        model.train()
        model(torch.randn(2, INPUT_LEN)).sum().backward()

        assert all(p.grad is None for p in model.encoder.parameters())
        assert all(p.grad is not None for p in model.head.parameters())
