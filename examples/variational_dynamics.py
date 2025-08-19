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


# warning pytorch might be unable to do this exactly
ppsioverptheta = K.jit(K.jacfwd(variational_wfn, argnums=0))
# compute \partial psi /\partial theta, i.e. jacobian of wfn


def ij(i, j):
    """
    Inner product
    """
    return K.tensordot(K.conj(i), j, 1)


@K.jit  # warning pytorch might be unable to do this exactly
def lhs_matrix(theta, psi0):
    vij = K.vmap(ij, vectorized_argnums=0)
    vvij = K.vmap(vij, vectorized_argnums=1)
    jacobian = ppsioverptheta(theta, psi0=psi0)
    # fim = K.adjoint(jacobian)@jacobian is also ok
    # speed comparison?
    jacobian = K.transpose(jacobian)
    fim = vvij(jacobian, jacobian)
    fim = K.real(fim)
    return fim


@K.jit  # warning pytorch might be unable to do this exactly
def rhs_vector(theta, psi0):
    def energy(theta, psi0):
        w = variational_wfn(theta, psi0)
        wl = K.stop_gradient(w)
        wl = K.conj(wl)
        wr = w
        wl = K.reshape(wl, [1, -1])
        wr = K.reshape(wr, [-1, 1])
        e = wl @ h @ wr
        # use sparse matrix if required
        return K.real(e)[0, 0]

    eg = K.grad(energy, argnums=0)
    rhs = eg(theta, psi0)
    rhs = K.imag(rhs)
    return rhs
    # for ITE, imag is replace with real
    # a simpler way to get rhs in ITE case is to directly evaluate
    # 0.5*\nabla <H>


@K.jit  # warning pytorch might be unable to do this exactly
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
