"""Functional ops with hand-derived backward passes. No autograd shortcuts."""

from rmk.backend import xp
from rmk.tensor import Tensor


def gelu(x):
    """GELU activation, tanh approximation (GPT-2's form)."""
    c = xp.sqrt(2.0 / xp.pi)
    a = 0.044715
    xd = x.data
    t = xp.tanh(c * (xd + a * xd ** 3))
    out = Tensor(0.5 * xd * (1.0 + t), (x,), "gelu")

    def _backward():
        dinner = c * (1.0 + 3.0 * a * xd ** 2)
        dgelu = 0.5 * (1.0 + t) + 0.5 * xd * (1.0 - t ** 2) * dinner
        x.grad += dgelu * out.grad

    out._backward = _backward
    return out


def layer_norm(x, gamma, beta, eps=1e-5):
    """LayerNorm over the last axis. x, gamma, beta are Tensors; gamma/beta shape (D,)."""
    xd = x.data
    mu = xd.mean(axis=-1, keepdims=True)
    xc = xd - mu
    var = (xc ** 2).mean(axis=-1, keepdims=True)
    std = xp.sqrt(var + eps)
    xhat = xc / std
    out = Tensor(gamma.data * xhat + beta.data, (x, gamma, beta), "layer_norm")

    def _backward():
        g = out.grad
        axes = tuple(range(g.ndim - 1))  # reduce over all but the feature axis
        beta.grad += g.sum(axis=axes)
        gamma.grad += (g * xhat).sum(axis=axes)
        dxhat = g * gamma.data
        mean_dxhat = dxhat.mean(axis=-1, keepdims=True)
        mean_dxhat_xhat = (dxhat * xhat).mean(axis=-1, keepdims=True)
        x.grad += (dxhat - mean_dxhat - xhat * mean_dxhat_xhat) / std

    out._backward = _backward
    return out
