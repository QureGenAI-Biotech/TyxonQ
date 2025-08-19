"""
jacobian calculation on different backend
"""

import numpy as np
import tyxonq as tq


def get_jac(n, nlayers):
    def state(params):
        params = K.reshape(params, [2 * nlayers, n])
        c = tq.Circuit(n)
        c = tq.templates.blocks.example_block(c, params, nlayers=nlayers)
        return c.state()

    params = K.ones([2 * nlayers * n])
    n1 = K.jacfwd(state)(params)
    n2 = K.jacrev(state)(params)
    # pytorch backend, jaxrev is upto conjugate with real jacobian
    params = K.cast(params, "float64")
    n3 = K.jacfwd(state)(params)
    n4 = K.jacrev(state)(params)
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

            np.testing.assert_allclose(K.real(n3).detach().cpu().numpy(), n4.detach().cpu().numpy())
            if K.name == "pytorch":
                n2 = K.conj(n2)
            np.testing.assert_allclose(n1.detach().cpu().numpy(), n2.detach().cpu().numpy())
