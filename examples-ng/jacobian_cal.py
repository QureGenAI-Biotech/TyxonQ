"""
jacobian calculation on different backend
"""

import numpy as np
import torch
import tyxonq as tq


def _numerical_jacobian(f, x, eps=1e-6):
    y0 = f(x)
    y0f = y0.reshape([-1])
    cols = []
    x_flat = x.reshape([-1])
    for i in range(x_flat.numel()):
        x_plus = x.clone()
        x_plus.reshape([-1])[i] = x_flat[i] + (x_flat.new_tensor(eps))
        y_plus = f(x_plus).reshape([-1])
        cols.append((y_plus - y0f) / eps)
    return tq.backend.stack(cols, axis=-1)


def get_jac(n, nlayers):
    def state(params):
        params = K.reshape(params, [2 * nlayers, n])
        c = tq.Circuit(n)
        c = tq.templates.blocks.example_block(c, params, nlayers=nlayers)
        return c.state()

    params = K.ones([2 * nlayers * n])
    n1 = _numerical_jacobian(state, params)
    n2 = _numerical_jacobian(state, params)
    # pytorch backend, jaxrev is upto conjugate with real jacobian
    params = K.cast(params, "float64")
    n3 = _numerical_jacobian(state, params)
    n4 = tq.backend.real(n3)
    # n4 is the real part of n3
    return n1, n2, n3, n4


for b in ["pytorch"]:
    with tq.runtime_backend(b) as K:
        with tq.runtime_dtype("complex128"):
            n1, n2, n3, n4 = get_jac(3, 1)

            print(n1)
            print(n2)
            print(n3)
            print(n4)

            np.testing.assert_allclose(
                K.real(n3).resolve_conj().detach().cpu().numpy(),
                n4.resolve_conj().detach().cpu().numpy(),
                rtol=1e-6,
                atol=1e-6,
            )
            np.testing.assert_allclose(
                n1.resolve_conj().detach().cpu().numpy(),
                n2.resolve_conj().detach().cpu().numpy(),
                rtol=1e-6,
                atol=1e-6,
            )
