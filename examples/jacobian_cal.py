"""
Jacobian calculation demo (refactored):
- Move building block to `tyxonq.libs.circuits_library.example_block`.
- Use numeric backend for RNG/array ops in forward where applicable.
"""

from __future__ import annotations

import numpy as np
import tyxonq as tq


def _numerical_jacobian(f, x, eps: float = 1e-6):
    # Keep finite-diff in NumPy for clarity; this is demo-only and backend-agnostic
    x = np.asarray(x, dtype=float)
    y0 = np.asarray(f(x), dtype=float).reshape([-1])
    cols = []
    x_flat = x.reshape([-1])
    for i in range(x_flat.size):
        x_plus = x.copy()
        x_plus.reshape([-1])[i] = x_flat[i] + eps
        y_plus = np.asarray(f(x_plus), dtype=float).reshape([-1])
        cols.append((y_plus - y0) / eps)
    return np.stack(cols, axis=-1)


def forward_statevector(params, n: int, nlayers: int):
    from tyxonq.libs.circuits_library import example_block
    from tyxonq.devices.simulators.statevector.engine import StatevectorEngine

    c = tq.Circuit(n)
    c = example_block(c, params, nlayers=nlayers)
    eng = StatevectorEngine("numpy")
    state = eng.state(c)
    return np.asarray(state, dtype=np.complex128)


ess = __name__

def get_jac(n: int, nlayers: int):
    def f_state(params):
        s = forward_statevector(params, n=n, nlayers=nlayers)
        return np.real(s)

    params = np.ones([2 * nlayers * n], dtype=float)
    n1 = _numerical_jacobian(f_state, params)
    n2 = _numerical_jacobian(f_state, params)
    params64 = params.astype(np.float64)
    n3 = _numerical_jacobian(f_state, params64)
    n4 = np.real(n3)
    return n1, n2, n3, n4


if __name__ == "__main__":
    n1, n2, n3, n4 = get_jac(3, 1)
    print(n1.shape, n2.shape, n3.shape, n4.shape)
    np.testing.assert_allclose(n3.real, n4, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(n1, n2, rtol=1e-6, atol=1e-6)
