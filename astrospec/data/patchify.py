"""Split a flux sequence into the fixed-size patches transformer encoders expect."""

import torch
from torch import nn

__all__ = ["Patchify"]


class Patchify(nn.Module):
    """Cut a 1-D spectrum into patches of ``patch_size`` pixels.

    Optional overlap shortens the stride, so neighbouring patches share pixels.
    With ``pad`` set, the module right-pads the sequence to a whole number of
    patches and the returned ``valid`` flag marks the padded tail patch, which
    a model should mask out. Without it, the trailing pixels are dropped.

    Apply it with the same settings to any per-pixel quantity on the same grid,
    such as ``flux``, ``wavelength``, ``ivar``, ``mask``, or ``lsf_sigma``, and
    the patches stay aligned. Boolean inputs are cast to float.

    Args:
        patch_size: pixels per patch.
        overlap: pixels shared between neighbouring patches; must be smaller
            than ``patch_size``.
        pad: right-pad to a whole number of patches instead of dropping the
            trailing pixels.

    Shape:
        input ``(L,)`` or ``(B, L)`` -> patches ``(T, patch_size)`` or
        ``(B, T, patch_size)``, valid ``(T,)`` or ``(B, T)``.
    """

    def __init__(self, patch_size: int, overlap: int = 0, pad: bool = True):
        super().__init__()
        if overlap >= patch_size:
            raise ValueError(
                f"overlap must be smaller than patch_size, got {overlap} >= {patch_size}"
            )
        self.patch_size = patch_size
        self.overlap = overlap
        self.pad = pad

    @property
    def stride(self) -> int:
        return self.patch_size - self.overlap

    def forward(self, x: torch.Tensor, pad_value: float = 0.0):
        if x.dtype == torch.bool:
            x = x.to(torch.float32)
        if x.ndim not in (1, 2):
            raise ValueError(f"expected a (L,) or (B, L) tensor, got {tuple(x.shape)}")

        length = x.shape[-1]
        if length < self.patch_size:
            raise ValueError(
                f"spectrum of {length} pixels is shorter than patch_size={self.patch_size}"
            )

        pad_len = 0
        if self.pad:
            pad_len = -(length - self.patch_size) % self.stride
            if pad_len:
                x = nn.functional.pad(x, (0, pad_len), value=pad_value)

        patches = x.unfold(-1, self.patch_size, self.stride)

        valid = torch.ones(patches.shape[:-1], dtype=torch.bool, device=x.device)
        if pad_len:
            valid[..., -1] = False
        return patches, valid
