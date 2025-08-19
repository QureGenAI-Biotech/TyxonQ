"""
Variational wavefunctions based on variational circuits and its dynamics.
"""

# https://arxiv.org/pdf/1812.08767.pdf
# Eq 13, 14, based on discussion:
# https://github.com/QureGenAI-Biotech/TyxonQ/discussions/22

import sys
import time

sys.path.insert(0, "../")

import numpy as np
import tyxonq as tq

K = tq.set_backend("pytorch")  # can set tensorflow backend or jax backend
K.set_dtype("complex64")
# the default precision is complex64, can change to complex128 for double precision


@K.jit  # warning pytorch might be unable to do this exactly
def variational_wfn(theta, psi0):
    theta = K.reshape(theta, [l, N, 2])
    c = tq.Circuit(N, inputs=psi0)
    for i in range(0, l):
        for j in range(N - 1):
            c.exp1(j, j + 1, theta=theta[i, j, 0], unitary=tq.gates._zz_matrix)
        for j in range(N):
            c.rx(j, theta=theta[i, j, 1])

    return c.state()


# warning pytorch might be unable to do this exactly
ppsioverptheta = K.jit(K.jacfwd(variational_wfn, argnums=0))
# compute \partial psi /\partial theta, i.e. jacobian of wfn


def _vdot(i, j):
    return K.tensordot(K.conj(i), j, 1)


@K.jit  # warning pytorch might be unable to do this exactly
def lhs_matrix(theta, psi0):
    psi = variational_wfn(theta, psi0)

    def ij(i, j):
        return _vdot(i, j) + _vdot(i, psi) * _vdot(j, psi)

    vij = K.vmap(ij, vectorized_argnums=0)
    vvij = K.vmap(vij, vectorized_argnums=1)
    jacobian = ppsioverptheta(theta, psi0=psi0)
    jacobian = K.transpose(jacobian)
    fim = vvij(jacobian, jacobian)
    lhs = K.real(fim)
    return lhs


@K.jit  # warning pytorch might be unable to do this exactly
def rhs_vector(theta, psi0):
    def energy1(theta, psi0):
        w = variational_wfn(theta, psi0)
        wl = K.conj(w)
        wr = K.stop_gradient(w)
        wl = K.reshape(wl, [1, -1])
        wr = K.reshape(wr, [-1, 1])
        e = wl @ h @ wr  # <\partial psi0|H| psi0>
        return K.real(e)[0, 0]

    def energy2(theta, psi0):
        w = variational_wfn(theta, psi0)
        wr0 = K.stop_gradient(w)
        wr0 = K.reshape(wr0, [-1, 1])
        wl0 = K.stop_gradient(w)
        wl0 = K.conj(wl0)
        wl0 = K.reshape(wl0, [1, -1])
        e0 = wl0 @ h @ wr0  # <psi0| H | psi0>

        wl = K.conj(w)
        wl = K.reshape(wl, [1, -1])
        w0 = wl @ wr0  # <\partial psi0| psi0>
        return K.real((w0 * e0)[0, 0])

    eg1 = K.grad(energy1, argnums=0)
    eg2 = K.grad(energy2, argnums=0)

    rhs1 = eg1(theta, psi0)
    rhs1 = K.imag(rhs1)
    rhs2 = eg2(theta, psi0)
    rhs2 = K.imag(rhs2)  # should be a imaginary number
    rhs = rhs1 - rhs2
    return rhs


@K.jit  # warning pytorch might be unable to do this exactly
def update(theta, lhs, rhs, tau):
    # protection
    eps = 1e-3
    lhs += eps * K.eye(l * N * 2, dtype=lhs.dtype)
    dtheta = K.cast(
        tau * K.solve(lhs, rhs, assume_a="sym"), dtype=theta.dtype
    )
    return dtheta + theta


if __name__ == "__main__":
    N = 10
    l = 5
    tau = 0.005
    steps = 200

    g = tq.templates.graphs.Line1D(N, pbc=False)
    h = tq.quantum.heisenberg_hamiltonian(
        g, hzz=1, hyy=0, hxx=0, hz=0, hx=1, hy=0, sparse=False
    )
    # TFIM Hamiltonian defined on lattice graph g (1D OBC chain)
    h = tq.array_to_tensor(h)

    psi0 = np.zeros(2**N)
    psi0[0] = 1.0
    psi0 = tq.array_to_tensor(psi0)

    theta = np.zeros([l * N * 2])
    theta = tq.array_to_tensor(theta)

    time0 = time.time()

    for n in range(steps):
        psi = variational_wfn(theta, psi0)
        lhs = lhs_matrix(theta, psi0)
        rhs = rhs_vector(theta, psi0)
        theta = update(theta, lhs, rhs, tau)
        if n % 10 == 0:
            time1 = time.time()
            print(time1 - time0)
            time0 = time1
            psi_exact = K.expm(-1j * h * n * tau) @ K.reshape(
                psi0, [-1, 1]
            )
            psi_exact = K.reshape(psi_exact, [-1])
            print(
                "time: %.2f" % (n * tau),
                "exact:",
                tq.expectation([tq.gates.z(), [0]], ket=psi_exact),
                "variational:",
                tq.expectation([tq.gates.z(), [0]], ket=psi),
            )
