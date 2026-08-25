"""Pre-norm transformer blocks shared by the transformer encoders."""

import math
import numbers
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

__all__ = ["LayerNorm", "FeedForward", "SelfAttention", "TransformerBlock", "init_by_depth"]


class LayerNorm(nn.Module):
    """Layer norm with an optional bias, which ``torch.nn.LayerNorm`` cannot disable.

    Args:
        shape: normalized shape, trailing the batch dimensions.
        eps: added to the denominator for numerical stability.
        bias: whether to learn a bias term.
    """

    def __init__(
        self,
        shape: Union[int, Tuple[int, ...], torch.Size],
        eps: float = 1e-5,
        bias: bool = True,
    ):
        super().__init__()

        self.eps = eps
        self.normalized_shape = (shape,) if isinstance(shape, numbers.Integral) else tuple(shape)

        self.weight = nn.Parameter(torch.ones(self.normalized_shape))
        self.bias = nn.Parameter(torch.zeros(self.normalized_shape)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)


class FeedForward(nn.Module):
    """The two-layer expansion MLP of a transformer block.

    Named ``encoder`` and ``decoder`` after the OmniSpectrum implementation, so
    that checkpoints trained there load without renaming.

    Args:
        in_features: input and output width.
        hidden_features: width of the expansion.
        activation: applied after the expansion; defaults to GELU.
        dropout: applied to the output.
        bias: whether the linear layers carry biases.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        activation: Optional[nn.Module] = None,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()

        self.encoder = nn.Linear(in_features, hidden_features, bias=bias)
        self.activation = activation if activation is not None else nn.GELU()
        self.decoder = nn.Linear(hidden_features, in_features, bias=bias)
        self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, x):
        x = self.decoder(self.activation(self.encoder(x)))
        return x if self.dropout_layer is None else self.dropout_layer(x)


class SelfAttention(nn.Module):
    """Multi-head self-attention over a token sequence.

    Args:
        embedding_dim: model width; must divide evenly by ``num_heads``.
        num_heads: number of attention heads.
        causal: restrict each token to attend to itself and its predecessors.
        dropout: applied to the attention weights and to the residual pathway.
        bias: whether the projections carry biases.

    Shape:
        input ``(B, T, embedding_dim)`` -> output of the same shape.
        ``attn_mask`` follows ``torch.nn.functional.scaled_dot_product_attention``:
        a boolean tensor broadcastable to ``(B, num_heads, T, T)`` where ``True``
        marks positions that may be attended to.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        causal: bool = False,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        if embedding_dim % num_heads:
            raise ValueError(
                f"embedding_dim={embedding_dim} is not divisible by num_heads={num_heads}"
            )

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.causal = causal
        self.dropout = dropout

        self.attention = nn.Linear(embedding_dim, 3 * embedding_dim, bias=bias)
        self.projection = nn.Linear(embedding_dim, embedding_dim, bias=bias)
        self.residual_dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask: Optional[torch.Tensor] = None):
        B, T, C = x.shape
        q, k, v = self.attention(x).split(self.embedding_dim, dim=2)
        q, k, v = (t.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2) for t in (q, k, v))

        is_causal = self.causal
        if is_causal and attn_mask is not None:
            # combine explicitly rather than relying on both being honoured at once
            causal = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
            attn_mask = attn_mask & causal
            is_causal = False

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )
        y = y.transpose(1, 2).reshape(B, T, C)
        return self.residual_dropout(self.projection(y))


class TransformerBlock(nn.Module):
    """Pre-norm self-attention and feed-forward, each on a residual branch.

    Args:
        embedding_dim: model width.
        num_heads: number of attention heads.
        causal: restrict attention to preceding tokens.
        dropout: used in attention, the residual pathway, and the feed-forward.
        bias: whether the layer norms and linear layers carry biases.
        mlp_expansion: hidden width of the feed-forward, as a multiple of
            ``embedding_dim``.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        causal: bool = False,
        dropout: float = 0.0,
        bias: bool = True,
        mlp_expansion: int = 4,
    ):
        super().__init__()

        self.layernorm1 = LayerNorm(embedding_dim, bias=bias)
        self.attention = SelfAttention(
            embedding_dim, num_heads, causal=causal, dropout=dropout, bias=bias
        )
        self.layernorm2 = LayerNorm(embedding_dim, bias=bias)
        self.mlp = FeedForward(
            embedding_dim, mlp_expansion * embedding_dim, dropout=dropout, bias=bias
        )

    def forward(self, x, attn_mask: Optional[torch.Tensor] = None):
        x = x + self.attention(self.layernorm1(x), attn_mask=attn_mask)
        return x + self.mlp(self.layernorm2(x))


def init_by_depth(module: nn.Module, depth: float) -> None:
    """Scale linear-layer initialization down with the depth of the stack.

    Keeps the variance of the residual stream roughly constant as blocks are
    added. Apply with ``module.apply(lambda m: init_by_depth(m, num_layers))``.
    """
    if isinstance(module, nn.Linear):
        std = 1 / math.sqrt(2 * module.weight.size(-1) * depth)
        nn.init.trunc_normal_(module.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
