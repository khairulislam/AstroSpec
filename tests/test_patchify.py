import pytest
import torch

from astrospec.data import Patchify


def test_exact_multiple_needs_no_padding():
    patches, valid = Patchify(patch_size=10)(torch.randn(100))
    assert patches.shape == (10, 10)
    assert valid.all()


def test_padded_tail_patch_is_marked_invalid():
    patches, valid = Patchify(patch_size=10)(torch.randn(105))
    assert patches.shape == (11, 10)
    assert valid[:-1].all() and not valid[-1]


def test_without_pad_the_tail_is_dropped():
    patches, valid = Patchify(patch_size=10, pad=False)(torch.randn(105))
    assert patches.shape == (10, 10)
    assert valid.all()


def test_overlap_shortens_the_stride():
    x = torch.arange(100, dtype=torch.float32)
    patches, _ = Patchify(patch_size=10, overlap=4)(x)
    torch.testing.assert_close(patches[0][6:], patches[1][:4])


def test_batched_matches_single():
    patchify = Patchify(patch_size=10, overlap=4)
    x = torch.randn(3, 105)
    patches, valid = patchify(x)
    assert patches.shape[0] == 3
    for i in range(3):
        torch.testing.assert_close(patches[i], patchify(x[i])[0])
        assert torch.equal(valid[i], patchify(x[i])[1])


def test_wavelength_and_mask_stay_aligned_with_flux():
    patchify = Patchify(patch_size=10)
    flux, wavelength = torch.randn(100), torch.linspace(3600.0, 9800.0, 100)
    mask = torch.zeros(100, dtype=torch.bool)

    assert patchify(flux)[0].shape == patchify(wavelength)[0].shape
    assert patchify(mask)[0].dtype == torch.float32


def test_invalid_arguments_rejected():
    with pytest.raises(ValueError, match="overlap"):
        Patchify(patch_size=10, overlap=10)
    with pytest.raises(ValueError, match="shorter than"):
        Patchify(patch_size=10)(torch.randn(5))
    with pytest.raises(ValueError, match=r"\(L,\) or \(B, L\)"):
        Patchify(patch_size=10)(torch.randn(2, 3, 100))
