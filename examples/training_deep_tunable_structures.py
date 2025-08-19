"""
An integrated script demonstrating:
1. shortcut setup of cotengra contractor (with correct interplay with multiprocessing);
2. jit scan acceleration for deep structured circuit with multiple variables;
3. tensor controlled tunable circuit structures all in one jit;
4. batched trainable parameters via vmap/vvag;
and yet anonther demonstration of infras for training with incremental random activation
"""

import time
import numpy as np
import torch
import tyxonq as tq


def main():
    tq.set_contractor("cotengra-40-64")
    K = tq.set_backend("pytorch")
    K.set_dtype("complex128")

    ii = tq.gates._ii_matrix
    xx = tq.gates._xx_matrix
    yy = tq.gates._yy_matrix
    zz = tq.gates._zz_matrix

    n = 12
    nlayers = 7
    g = tq.templates.graphs.Line1D(n)
    ncircuits = 10
    heih = tq.quantum.heisenberg_hamiltonian(
        g, hzz=1.0, hyy=1.0, hxx=1.0, hx=0, hy=0, hz=0
    )

    def energy(params, structures, n, nlayers):
        def one_layer(state, others):
            params, structures = others
            # print(state.shape, params.shape, structures.shape)
            l = 0
            c = tq.Circuit(n, inputs=state)
            for i in range(1, n, 2):
                matrix = structures[3 * l, i] * ii + (1.0 - structures[3 * l, i]) * (
                    K.cos(params[3 * l, i]) * ii + 1.0j * K.sin(params[3 * l, i]) * zz
                )
                c.any(
                    i,
                    (i + 1) % n,
                    unitary=matrix,
                )

            ### YY
            for i in range(1, n, 2):
                matrix = structures[3 * l + 1, i] * ii + (
                    1.0 - structures[3 * l + 1, i]
                ) * (
                    K.cos(params[3 * l + 1, i]) * ii
                    + 1.0j * K.sin(params[3 * l + 1, i]) * yy
                )
                c.any(
                    i,
                    (i + 1) % n,
                    unitary=matrix,
                )

            ### XX
            for i in range(1, n, 2):
                matrix = structures[3 * l + 2, i] * ii + (
                    1.0 - structures[3 * l + 2, i]
                ) * (
                    K.cos(params[3 * l + 2, i]) * ii
                    + 1.0j * K.sin(params[3 * l + 2, i]) * xx
                )
                c.any(
                    i,
                    (i + 1) % n,
                    unitary=matrix,
                )

            ### Even layer
            ### ZZ
            for i in range(0, n, 2):
                matrix = structures[3 * l, i] * ii + (1.0 - structures[3 * l, i]) * (
                    K.cos(params[3 * l, i]) * ii + 1.0j * K.sin(params[3 * l, i]) * zz
                )
                c.any(
                    i,
                    (i + 1) % n,
                    unitary=matrix,
                )
            ### YY

            for i in range(0, n, 2):
                matrix = structures[3 * l + 1, i] * ii + (
                    1.0 - structures[3 * l + 1, i]
                ) * (
                    K.cos(params[3 * l + 1, i]) * ii
                    + 1.0j * K.sin(params[3 * l + 1, i]) * yy
                )
                c.any(
                    i,
                    (i + 1) % n,
                    unitary=matrix,
                )

            ### XX
            for i in range(0, n, 2):
                matrix = structures[3 * l + 2, i] * ii + (
                    1.0 - structures[3 * l + 2, i]
                ) * (
                    K.cos(params[3 * l + 2, i]) * ii
                    + 1.0j * K.sin(params[3 * l + 2, i]) * xx
                )
                c.any(
                    i,
                    (i + 1) % n,
                    unitary=matrix,
                )
            s = c.state()
            return s, s

        params = K.cast(K.real(params), dtype="complex128")
        structures = (K.sign(structures) + 1) / 2  # 0 or 1
        structures = K.cast(structures, dtype="complex128")

        c = tq.Circuit(n)

        for i in range(n):
            c.x(i)
        for i in range(0, n, 2):
            c.h(i)
        for i in range(0, n, 2):
            c.cnot(i, i + 1)
        s = c.state()
        # warning pytorch might be unable to do this exactly
        s, _ = K.scan(
            one_layer,
            s,
            (
                K.reshape(params, [nlayers, 3, n]),
                K.reshape(structures, [nlayers, 3, n]),
            ),
        )
        c = tq.Circuit(n, inputs=s)
        # e = tq.templates.measurements.heisenberg_measurements(
        #     c, g, hzz=1, hxx=1, hyy=1, hx=0, hy=0, hz=0
        # )
        e = tq.templates.measurements.operator_expectation(c, heih)
        return K.real(e)

    # warning pytorch might be unable to do this exactly
    vagf = K.jit(K.vvag(energy, argnums=0, vectorized_argnums=0), static_argnums=(2, 3))

    structures = tq.array_to_tensor(
        np.random.uniform(low=0.0, high=1.0, size=[3 * nlayers, n]), dtype="complex128"
    )
    structures -= 1.0 * K.ones([3 * nlayers, n])
    params = tq.array_to_tensor(
        np.random.uniform(low=-0.1, high=0.1, size=[ncircuits, 3 * nlayers, n]),
        dtype="float64",
    )

    # opt = K.optimizer(tf.keras.optimizers.Adam(1e-2))
    optimizer = torch.optim.Adam([torch.nn.Parameter(params)], lr=1e-2)

    for _ in range(50):
        time0 = time.time()
        e, grads = vagf(params, structures, n, nlayers)
        time1 = time.time()
        # warning: pytorch optimizer usage is different
        optimizer.zero_grad()
        params.grad = grads
        optimizer.step()
        print(K.numpy(e), time1 - time0)


if __name__ == "__main__":
    main()
