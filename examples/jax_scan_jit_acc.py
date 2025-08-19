"""
reducing pytorch jit compiling time by some magic:
for backend agnostic but similar approach,
see `hea_scan_jit_acc.py`
"""

import numpy as np
import torch
import tyxonq as tq

K = tq.set_backend("pytorch")
K.set_dtype("complex128")


def energy_reference(param, n, nlayers):
    c = tq.Circuit(n)
    for i in range(n):
        c.h(i)
    for i in range(nlayers):
        for j in range(n - 1):
            c.rzz(j, j + 1, theta=param[i, j, 0])
        for j in range(n):
            c.rx(j, theta=param[i, j, 1])
    return K.real(c.expectation_ps(z=[0, 1]))


# warning pytorch might be unable to do this exactly
vg_reference = K.jit(
    K.value_and_grad(energy_reference, argnums=0), static_argnums=(1, 2)
)

with tq.runtime_backend("pytorch") as tfk:

    def energy_reference_tf(param, n, nlayers):
        c = tq.Circuit(n)
        for i in range(n):
            c.h(i)
        for i in range(nlayers):
            for j in range(n - 1):
                c.rzz(j, j + 1, theta=param[i, j, 0])
            for j in range(n):
                c.rx(j, theta=param[i, j, 1])
        return tfk.real(c.expectation_ps(z=[0, 1]))

    # warning pytorch might be unable to do this exactly
    vg_reference_tf = tfk.jit(
        tfk.value_and_grad(energy_reference_tf, argnums=0), static_argnums=(1, 2)
    )

# a jit efficient way to utilize pytorch scan


def energy(param, n, nlayers, each):
    def loop_f(s_, param_):
        c_ = tq.Circuit(n, inputs=s_)
        for i in range(each):
            for j in range(n - 1):
                c_.rzz(j, j + 1, theta=param_[i, j, 0])
            for j in range(n):
                c_.rx(j, theta=param_[i, j, 1])
        s_ = c_.state()
        return s_, s_

    c = tq.Circuit(n)
    for i in range(n):
        c.h(i)
    s = c.state()
    # warning pytorch might be unable to do this exactly
    s1, _ = K.scan(loop_f, s, K.reshape(param, [nlayers // each, each, n, 2]))
    c1 = tq.Circuit(n, inputs=s1)
    return K.real(c1.expectation_ps(z=[0, 1]))


# warning pytorch might be unable to do this exactly
vg = K.jit(K.value_and_grad(energy, argnums=0), static_argnums=(1, 2, 3))

if __name__ == "__main__":
    n = 10
    nlayers = 32
    param = K.implicit_randn([nlayers, n, 2])

    r1 = tq.utils.benchmark(vg, param, n, nlayers, 1)
    print(r1[0][0])
    r1 = tq.utils.benchmark(vg, param, n, nlayers, 2)
    print(r1[0][0])
    r1 = tq.utils.benchmark(vg, param, n, nlayers, 4)
    print(r1[0][0])

    with tq.runtime_backend("pytorch"):
        print("pytorch plain impl")
        param_tf = tq.array_to_tensor(param, dtype="float32")
        r0 = tq.utils.benchmark(vg_reference_tf, param_tf, n, nlayers)

    np.testing.assert_allclose(r0[0][0].detach().cpu().numpy(), r1[0][0].detach().cpu().numpy(), atol=1e-5)
    np.testing.assert_allclose(r0[0][1].detach().cpu().numpy(), r1[0][1].detach().cpu().numpy(), atol=1e-5)
    # correctness check

    print("pytorch plain impl (may be super slow for deeper system)")
    r0 = tq.utils.benchmark(vg_reference, param, n, nlayers)  # too slow
