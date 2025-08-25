"""
A simple script to benchmark the approximation power of the MPS simulator.
"""

import sys

sys.path.insert(0, "../")

import tyxonq as tq

K = tq.set_backend("pytorch")
K.set_dtype("complex128")


def tfi_energy(c, n, j=1.0, h=-1.0):
    e = 0.0
    for i in range(n):
        e += h * c.expectation((tq.gates.x(), [i]))
    for i in range(n - 1):
        e += j * c.expectation((tq.gates.z(), [i]), (tq.gates.z(), [(i + 1) % n]))
    return e


def energy(param, mpsd=None):
    if mpsd is None:
        c = tq.Circuit(n)
    else:
        c = tq.MPSCircuit(n)
        c.set_split_rules({"max_singular_values": mpsd})

    for i in range(n):
        c.h(i)
    for j in range(nlayers):
        for i in range(n - 1):
            c.exp1(
                i,
                (i + 1) % n,
                theta=param[2 * j, i],
                unitary=tq.gates._zz_matrix,
            )
        for i in range(n):
            c.rx(i, theta=param[2 * j + 1, i])

    e = tfi_energy(c, n)
    e = K.real(e)
    if mpsd is not None:
        fidelity = c._fidelity
    else:
        fidelity = None
    return e, c.state(), fidelity


n, nlayers = 15, 20
print("number of qubits: ", n)
print("number of layers: ", nlayers)

param = K.implicit_randu([2 * nlayers, n])
# param = K.ones([2 * nlayers, n])
# it turns out that the mps approximation power highly depends on the
# parameters, if we use ``param = K.ones``, the apprixmation ratio decays very fast
# At least, the estimated fidelity is a very good proxy metric for real fidelity
# as long as it is larger than 50%
e0, s0, _ = energy(param)
print(
    "entanglement: ",
    K.numpy(
        tq.quantum.entropy(tq.quantum.reduced_density_matrix(s0, cut=n // 2))
    ),
)

for mpsd in [2, 5, 10, 20, 50, 100]:
    e1, s1, f1 = energy(param, mpsd=mpsd)
    print("------------------------")
    print("bond dimension: ", mpsd)
    print(
        "exact energy: ",
        K.numpy(e0),
        "mps simulator energy: ",
        K.numpy(e1),
    )
    print(
        "energy relative error(%): ",
        K.numpy(K.abs((e1 - e0) / e0)) * 100,
    )
    print("estimated fidelity:", K.numpy(f1))
    print(
        "real fidelity:",
        K.numpy(
            K.abs(K.tensordot(K.conj(s1), s0, 1))
        ),
    )
