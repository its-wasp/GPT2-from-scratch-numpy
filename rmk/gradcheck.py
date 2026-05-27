"""
Numerical gradient check for ndarray Tensors.

Compares analytical gradients (from Tensor.backward()) against finite-difference
estimates of the form (f(x+eps) - f(x-eps)) / (2*eps), one element at a time.
Use this to verify every new op before using it in real code.

If `f` returns a non-scalar Tensor, the output is summed to a scalar before
calling backward. This lets us gradcheck any op without needing to construct
a scalar loss by hand.
"""

from rmk.backend import xp
from rmk.tensor import Tensor


def gradcheck(f, inputs, eps=1e-6, tol=1e-5):
    """
    f: callable taking Tensor args, returning a Tensor (any shape).
    inputs: list of Tensor leaves whose gradients should match the analytical ones.

    Returns the maximum absolute error. Raises AssertionError if it exceeds `tol`.
    """

    def f_scalar(*args):
        out = f(*args)
        return out.sum() if out.data.ndim > 0 else out

    # 1. Analytical gradients via our autograd.
    for inp in inputs:
        inp.zero_grad()
    f_scalar(*inputs).backward()
    analytical = [inp.grad.copy() for inp in inputs]

    # 2. Numerical gradients via central differences, one element at a time.
    #    reshape(-1) on a contiguous ndarray returns a view, so mutating
    #    flat_data[i] perturbs inp.data in place.
    numerical = []
    for inp in inputs:
        # reshape(-1) is a view only for contiguous arrays; otherwise we'd
        # perturb a copy and the numerical gradient would be silently wrong.
        assert inp.data.flags["C_CONTIGUOUS"], (
            "gradcheck input must be contiguous so reshape(-1) is a view"
        )
        num = xp.zeros_like(inp.data)
        flat_data = inp.data.reshape(-1)
        flat_num = num.reshape(-1)
        for i in range(flat_data.size):
            orig = float(flat_data[i])
            flat_data[i] = orig + eps
            L_plus = float(f_scalar(*inputs).data)
            flat_data[i] = orig - eps
            L_minus = float(f_scalar(*inputs).data)
            flat_data[i] = orig
            flat_num[i] = (L_plus - L_minus) / (2 * eps)
        numerical.append(num)

    # 3. Compare.
    max_err = 0.0
    for i, (a, n) in enumerate(zip(analytical, numerical)):
        err = float(xp.abs(a - n).max())
        max_err = max(max_err, err)
        assert err < tol, f"gradcheck input[{i}]: max_err={err}"
    return max_err
