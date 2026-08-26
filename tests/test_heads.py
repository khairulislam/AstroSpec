import pytest
import torch

from astrospec.heads import CrossAttentionHead, CrossAttentionPool
from astrospec.models import AstroPT, ShenSpectralTokenizer, SpecFormer, SpecPTEncoder

EMBED_DIM, NUM_HEADS, SEQ_LEN, NUM_OUTPUTS = 32, 4, 10, 3


@pytest.fixture
def pool():
    return CrossAttentionPool(EMBED_DIM, num_heads=NUM_HEADS, dropout=0.0).eval()


@pytest.fixture
def head():
    return CrossAttentionHead(EMBED_DIM, NUM_OUTPUTS, num_heads=NUM_HEADS, dropout=0.0).eval()


def test_pool_shape(pool):
    tokens = torch.randn(2, SEQ_LEN, EMBED_DIM)
    assert pool(tokens).shape == (2, EMBED_DIM)


def test_pool_ignores_masked_positions(pool):
    tokens = torch.randn(2, SEQ_LEN, EMBED_DIM)
    key_padding_mask = torch.zeros(2, SEQ_LEN, dtype=torch.bool)
    key_padding_mask[:, -3:] = True

    perturbed = tokens.clone()
    perturbed[:, -3:] = torch.randn(2, 3, EMBED_DIM) * 100

    torch.testing.assert_close(
        pool(tokens, key_padding_mask=key_padding_mask),
        pool(perturbed, key_padding_mask=key_padding_mask),
    )


def test_head_shape(head):
    tokens = torch.randn(2, SEQ_LEN, EMBED_DIM)
    assert head(tokens).shape == (2, NUM_OUTPUTS)


def test_head_backward(head):
    head.train()
    tokens = torch.randn(2, SEQ_LEN, EMBED_DIM)
    head(tokens).sum().backward()
    assert all(p.grad is not None for p in head.parameters())


class TestComposesWithEveryEncoder:
    """CrossAttentionHead is otherwise-useless encoders' path to a task output."""

    def test_specformer(self, head):
        encoder = SpecFormer(
            input_dim=22, embed_dim=EMBED_DIM, num_layers=1, num_heads=NUM_HEADS, max_len=64
        ).eval()
        patches = torch.randn(2, SEQ_LEN, 22)
        tokens = encoder.forward_features(patches)
        assert head(tokens).shape == (2, NUM_OUTPUTS)

    def test_astropt(self, head):
        encoder = AstroPT(input_dim=16, embed_dim=EMBED_DIM, num_layers=1, num_heads=NUM_HEADS).eval()
        patches = torch.randn(2, SEQ_LEN, 16)
        wavelength = torch.linspace(3600.0, 9800.0, SEQ_LEN * 16).view(1, SEQ_LEN, 16).expand(2, -1, -1)
        tokens = encoder.forward_features(patches, wavelength)
        assert head(tokens).shape == (2, NUM_OUTPUTS)

    def test_specpt_encoder(self, head):
        encoder = SpecPTEncoder(embed_dim=EMBED_DIM, num_layers=1, num_heads=NUM_HEADS).eval()
        tokens = encoder(torch.randn(2, 256))
        assert head(tokens).shape == (2, NUM_OUTPUTS)

    def test_shen_et_al(self, head):
        encoder = ShenSpectralTokenizer(
            patch_size=16, embed_dim=EMBED_DIM, num_enc_layers=1, num_dec_layers=1, num_heads=NUM_HEADS
        ).eval()
        flux = torch.randn(2, SEQ_LEN, 16)
        wavelength = torch.linspace(3600.0, 9800.0, SEQ_LEN * 16).view(1, SEQ_LEN, 16).expand(2, -1, -1)
        tokens = encoder.forward_features(flux, wavelength)
        assert head(tokens).shape == (2, NUM_OUTPUTS)
