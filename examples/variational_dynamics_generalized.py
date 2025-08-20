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


@K.jit
def variational_wfn(theta, psi0):
    theta = K.reshape(theta, [l, N, 2])
    c = tq.Circuit(N, inputs=psi0)
    for i in range(0, l):
        for j in range(N - 1):
            c.exp1(j, j + 1, theta=theta[i, j, 0], unitary=tq.gates._zz_matrix)
        for j in range(N):
            c.rx(j, theta=theta[i, j, 1])

    return c.state()

def numerical_jacobian(theta, psi0, eps=1e-3):
    """Numerical Jacobian J (state_dim, P) with central differences."""
    theta = K.reshape(theta, [-1])
    num_params = theta.shape[0]
    cols = []
    for k in range(num_params):
        tp = K.copy(theta)
        tm = K.copy(theta)
        tp[k] = tp[k] + eps
        tm[k] = tm[k] - eps
        wp = variational_wfn(tp, psi0)
        wm = variational_wfn(tm, psi0)
        cols.append((wp - wm) / (2 * eps))
    J = K.transpose(K.stack(cols))
    return J


def _vdot(i, j):
    return K.tensordot(K.conj(i), j, 1)


def lhs_matrix(theta, psi0):
    psi = variational_wfn(theta, psi0)
    J = numerical_jacobian(theta, psi0)
    JH = K.conj(K.transpose(J))  # (P, state)
    fim = JH @ J  # (P, P)
    v = JH @ K.reshape(psi, [-1, 1])  # (P,1), components <dpsi|psi>
    outer = v @ K.transpose(v)  # (P,P), v v^T
    lhs = K.real(fim + outer)
    return lhs


def rhs_vector(theta, psi0):
    psi = variational_wfn(theta, psi0)
    J = numerical_jacobian(theta, psi0)
    JH = K.conj(K.transpose(J))  # (P, state)
    Hpsi = K.reshape(h @ K.reshape(psi, [-1, 1]), [-1])
    e0 = K.tensordot(K.conj(psi), Hpsi, 1)  # scalar <psi|H|psi>
    term1 = JH @ K.reshape(Hpsi, [-1, 1])  # (P,1)
    v = JH @ K.reshape(psi, [-1, 1])  # (P,1)
    term2 = v * K.reshape(e0, [1, 1])  # (P,1)
    rhs = K.imag(K.reshape(term1 - term2, [-1]))
    return rhs


@K.jit
def update(theta, lhs, rhs, tau):
    # protection
    eps = 1e-3
    lhs += eps * K.eye(l * N * 2, dtype=lhs.dtype)
    dtheta = K.cast(
        tau * K.solve(lhs, rhs, assume_a="sym"), dtype=theta.dtype
    )
    return dtheta + theta


if __name__ == "__main__":
    N = 6
    l = 3
    tau = 0.01
    steps = 20

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
