"""Multi-layer perceptron shared by several spectral encoders."""

from typing import Optional, Sequence

from torch import nn

__all__ = ["MLP"]


class MLP(nn.Sequential):
    """Linear layers with an activation and dropout between them.

    Args:
        n_in: input width.
        n_out: output width.
        n_hidden: hidden widths.
        act: activations, one per hidden layer. Defaults to ``LeakyReLU``.
            ``len(n_hidden) + 1`` entries are required for compatibility with
            spender's ``MLP``, but the last is unused: the output layer is
            linear.
        dropout: dropout rate after each activation.
    """

    def __init__(
        self,
        n_in: int,
        n_out: int,
        n_hidden: Sequence[int] = (16, 16, 16),
        act: Optional[Sequence[nn.Module]] = None,
        dropout: float = 0.0,
    ):
        if act is None:
            act = [nn.LeakyReLU() for _ in range(len(n_hidden) + 1)]
        if len(act) != len(n_hidden) + 1:
            raise ValueError(
                f"act must have len(n_hidden) + 1 = {len(n_hidden) + 1} entries, got {len(act)}"
            )

        widths = [n_in, *n_hidden, n_out]
        layers = []
        for i in range(len(widths) - 2):
            layers += [nn.Linear(widths[i], widths[i + 1]), act[i], nn.Dropout(dropout)]
        layers.append(nn.Linear(widths[-2], widths[-1]))
        super().__init__(*layers)
