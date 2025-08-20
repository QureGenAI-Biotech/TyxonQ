"""
MERA VQE example with Hamiltonian expectation in MPO representation
"""

import time
import logging
import numpy as np
import tensornetwork as tn
import torch
import cotengra
import tyxonq as tq

logger = logging.getLogger("tyxonq")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
logger.addHandler(ch)


K = tq.set_backend("pytorch")
K.set_dtype("complex128")


optc = cotengra.ReusableHyperOptimizer(
    methods=["greedy", "kahypar"],
    parallel=False,
    minimize="combo",
    max_time=15,
    max_repeats=64,
    progbar=False,
)

tq.set_contractor("custom", optimizer=optc, preprocessing=True)


def MERA_circuit(params, n, d):
    c = tq.Circuit(n)

    idx = 0

    for i in range(n):
        c.rx(i, theta=params[2 * i])
        c.rz(i, theta=params[2 * i + 1])
    idx += 2 * n

    for n_layer in range(1, int(np.log2(n)) + 1):
        n_qubit = 2**n_layer
        step = int(n / n_qubit)

        for _ in range(d):
            # even
            for i in range(step, n - step, 2 * step):
                c.exp1(i, i + step, theta=params[idx], unitary=tq.gates._xx_matrix)
                c.exp1(i, i + step, theta=params[idx + 1], unitary=tq.gates._zz_matrix)
                idx += 2

            # odd
            for i in range(0, n, 2 * step):
                c.exp1(i, i + step, theta=params[idx], unitary=tq.gates._xx_matrix)
                c.exp1(i, i + step, theta=params[idx + 1], unitary=tq.gates._zz_matrix)
                idx += 2

        for i in range(0, n, step):
            c.rx(i, theta=params[idx])
            c.rz(i, theta=params[idx + 1])
            idx += 2

    return c, idx


def MERA(params, n, d, hamiltonian_mpo):
    c, _ = MERA_circuit(params, n, d)
    return tq.templates.measurements.mpo_expectation(c, hamiltonian_mpo)


# Removed warning comment
MERA_vvag = K.jit(K.vectorized_value_and_grad(MERA), static_argnums=(1, 2, 3))


def train(opt, j, b, n, d, batch, maxiter):
    Jx = j * np.ones([n - 1])  # strength of xx interaction (OBC)
    Bz = -b * np.ones([n])  # strength of transverse field
    hamiltonian_mpo = tn.matrixproductstates.mpo.FiniteTFI(Jx, Bz, dtype=np.complex128)
    # matrix product operator
    hamiltonian_mpo = tq.quantum.tn2qop(hamiltonian_mpo)
    _, idx = MERA_circuit(K.ones([int(1e6)]), n, d)
    params = torch.nn.Parameter(torch.randn(batch, idx) * 0.05)
    optimizer = torch.optim.Adam([params], lr=1e-2)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    
    times = []
    times.append(time.time())
    for i in range(maxiter):
        # Forward pass - handle batched parameters
        if params.dim() > 1:
            e = MERA_vvag(params, n, d, hamiltonian_mpo)
            e = K.mean(e[0])  # Take mean of batch
        else:
            e = MERA(params, n, d, hamiltonian_mpo)
        
        # Backward pass
        optimizer.zero_grad()
        e.backward()
        optimizer.step()
        
        if i % 500 == 499:
            scheduler.step()
            
        times.append(time.time())
        if i % 100 == 99:
            print("energy: ", e.detach().cpu().item())
            print(
                "time analysis: ",
                times[1] - times[0],
                (times[-1] - times[1]) / (len(times) - 2),
            )
    return torch.min(e)


if __name__ == "__main__":
    # Reduced problem size and iterations for CI speed
    e = train(None, 1, -1, 8, 1, 2, 50)
    print("optimized energy:", e.detach().cpu().item())

# backend: n, d, batch: compiling time, running time
# pytorch: 16, 2, 8: 730s, 0.033s
# pytorch: 32, 2, 2: 251s, 0.9s
