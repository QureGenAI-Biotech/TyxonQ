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

    ii = tq.array_to_tensor(tq.gates._ii_matrix, dtype=tq.dtypestr)
    xx = tq.array_to_tensor(tq.gates._xx_matrix, dtype=tq.dtypestr)
    yy = tq.array_to_tensor(tq.gates._yy_matrix, dtype=tq.dtypestr)
    zz = tq.array_to_tensor(tq.gates._zz_matrix, dtype=tq.dtypestr)

    n = 6
    nlayers = 3
    g = tq.templates.graphs.Line1D(n)
    ncircuits = 3
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
                w = structures[3 * l, i]
                angle = params[3 * l, i]
                matrix = w * ii + (1.0 - w) * (
                    K.cos(angle) * ii + 1.0j * K.sin(angle) * zz
                )
                c.any(
                    i,
                    (i + 1) % n,
                    unitary=matrix,
                )

            ### YY
            for i in range(1, n, 2):
                w = structures[3 * l + 1, i]
                angle = params[3 * l + 1, i]
                matrix = w * ii + (1.0 - w) * (
                    K.cos(angle) * ii + 1.0j * K.sin(angle) * yy
                )
                c.any(
                    i,
                    (i + 1) % n,
                    unitary=matrix,
                )

            ### XX
            for i in range(1, n, 2):
                w = structures[3 * l + 2, i]
                angle = params[3 * l + 2, i]
                matrix = w * ii + (1.0 - w) * (
                    K.cos(angle) * ii + 1.0j * K.sin(angle) * xx
                )
                c.any(
                    i,
                    (i + 1) % n,
                    unitary=matrix,
                )

            ### Even layer
            ### ZZ
            for i in range(0, n, 2):
                w = structures[3 * l, i]
                angle = params[3 * l, i]
                matrix = w * ii + (1.0 - w) * (
                    K.cos(angle) * ii + 1.0j * K.sin(angle) * zz
                )
                c.any(
                    i,
                    (i + 1) % n,
                    unitary=matrix,
                )
            ### YY

            for i in range(0, n, 2):
                w = structures[3 * l + 1, i]
                angle = params[3 * l + 1, i]
                matrix = w * ii + (1.0 - w) * (
                    K.cos(angle) * ii + 1.0j * K.sin(angle) * yy
                )
                c.any(
                    i,
                    (i + 1) % n,
                    unitary=matrix,
                )

            ### XX
            for i in range(0, n, 2):
                w = structures[3 * l + 2, i]
                angle = params[3 * l + 2, i]
                matrix = w * ii + (1.0 - w) * (
                    K.cos(angle) * ii + 1.0j * K.sin(angle) * xx
                )
                c.any(
                    i,
                    (i + 1) % n,
                    unitary=matrix,
                )
            s = c.state()
            return s, s

        params = K.cast(K.real(params), dtype="complex128")
        structures = (K.sign(structures) + 1) / 2  # complex-safe via backend
        structures = K.cast(structures, dtype="complex128")

        c = tq.Circuit(n)

        for i in range(n):
            c.x(i)
        for i in range(0, n, 2):
            c.h(i)
        for i in range(0, n, 2):
            c.cnot(i, i + 1)
        s = c.state()
        # manual loop to avoid functorch scan overhead
        p3 = K.reshape(params, [nlayers, 3, n])
        s3 = K.reshape(structures, [nlayers, 3, n])
        for lidx in range(nlayers):
            s, _ = one_layer(s, (p3[lidx], s3[lidx]))
        c = tq.Circuit(n, inputs=s)
        # e = tq.templates.measurements.heisenberg_measurements(
        #     c, g, hzz=1, hxx=1, hyy=1, hx=0, hy=0, hz=0
        # )
        e = tq.templates.measurements.operator_expectation(c, heih)
        return K.real(e)

    # warning pytorch might be unable to do this exactly
    vagf = K.jit(K.value_and_grad(energy, argnums=0), static_argnums=(2, 3))

    structures = tq.array_to_tensor(
        np.random.uniform(low=0.0, high=1.0, size=[3 * nlayers, n]), dtype="complex128"
    )
    structures -= 1.0 * K.ones([3 * nlayers, n])
    params = tq.array_to_tensor(
        np.random.uniform(low=-0.1, high=0.1, size=[3 * nlayers, n]),
        dtype="float64",
    )

    # opt = K.optimizer(tf.keras.optimizers.Adam(1e-2))
    params = torch.nn.Parameter(params)
    optimizer = torch.optim.Adam([params], lr=1e-2)

    for _ in range(10):
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
