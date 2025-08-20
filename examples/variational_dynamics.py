"""
Variational wavefunctions based on variational circuits and its dynamics.
"""

# Variational Quantum Algorithm for Quantum Dynamics
# Ref: PRL 125, 010501 (2020)

import sys
import time

sys.path.insert(0, "../")

import numpy as np
import tyxonq as tq

K = tq.set_backend("pytorch")  # can set tensorflow backend or jax backend
K.set_dtype("complex64")
# the default precision is complex64, can change to complex128 for double precision


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
    """Compute numerical Jacobian J_{:,k} = d psi / d theta_k via central differences.
    Returns a tensor of shape (2^N, P) where P = l*N*2.
    """
    theta = K.reshape(theta, [-1])
    num_params = theta.shape[0]
    # preallocate jacobian as list of columns
    cols = []
    for k in range(num_params):
        tp = K.copy(theta)
        tm = K.copy(theta)
        tp[k] = tp[k] + eps
        tm[k] = tm[k] - eps
        wp = variational_wfn(tp, psi0)
        wm = variational_wfn(tm, psi0)
        cols.append((wp - wm) / (2 * eps))
    # stack columns: shape (P, 2^N) -> then transpose
    J = K.transpose(K.stack(cols))
    return J


def ij(i, j):
    """
    Inner product
    """
    return K.tensordot(K.conj(i), j, 1)


def lhs_matrix_from_jacobian(J):
    # FIM = Re( J^H J )
    JH = K.conj(K.transpose(J))
    fim = K.real(JH @ J)
    return fim


def rhs_vector_from_jacobian(theta, psi, J):
    # RHS_k = Im( <d psi / d theta_k | H | psi> )
    Hpsi = K.reshape(h @ K.reshape(psi, [-1, 1]), [-1])
    # J shape: (state_dim, P) -> conj(J)^T shape: (P, state_dim)
    inner = K.transpose(K.conj(J)) @ K.reshape(Hpsi, [-1, 1])
    inner = K.reshape(inner, [-1])
    rhs = K.imag(inner)
    return rhs
    # for ITE, imag is replace with real
    # a simpler way to get rhs in ITE case is to directly evaluate
    # 0.5*\nabla <H>


@K.jit
def update(theta, lhs, rhs, tau):
    # protection
    eps = 1e-4
    lhs += eps * K.eye(l * N * 2, dtype=lhs.dtype)
    return (
        K.cast(
            tau * K.solve(lhs, rhs, assume_a="sym"), dtype=theta.dtype
        )
        + theta
    )


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
        J = numerical_jacobian(theta, psi0, eps=1e-3)
        lhs = lhs_matrix_from_jacobian(J)
        rhs = rhs_vector_from_jacobian(theta, psi, J)
        theta = update(theta, lhs, rhs, tau)
        if n % 5 == 0:
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
