"""
Compute the circuit gradient with single qubit random Haar averaged to
demonstrate the gradient vanishing, aka, barren plateau.
"""

import numpy as np
from tqdm import tqdm
import tyxonq as tq

xx = tq.gates._xx_matrix
yy = tq.gates._yy_matrix
zz = tq.gates._zz_matrix

nqubits, nlayers = 8, 6
# number of qubits and number of even-odd brick layers in the circuit
K = tq.set_backend("pytorch")

xx_t = K.cast(K.convert_to_tensor(xx), K.dtypestr)
yy_t = K.cast(K.convert_to_tensor(yy), K.dtypestr)
zz_t = K.cast(K.convert_to_tensor(zz), K.dtypestr)

def ansatz(param, gate_param):
    # gate_param here is a 2D-vector for theta and phi in the XXZ gate
    c = tq.Circuit(nqubits)
    # customize the circuit structure as you like
    for j in range(nlayers):
        for i in range(nqubits):
            c.ry(i, theta=param[j, 0, i, 0])
            c.rz(i, theta=param[j, 0, i, 1])
            c.ry(i, theta=param[j, 0, i, 2])
        for i in range(0, nqubits - 1, 2):  # even brick
            c.exp(
                i,
                i + 1,
                theta=1.0,
                unitary=gate_param[0] * (xx_t + yy_t) + gate_param[1] * zz_t,
            )
        for i in range(nqubits):
            c.ry(i, theta=param[j, 1, i, 0])
            c.rz(i, theta=param[j, 1, i, 1])
            c.ry(i, theta=param[j, 1, i, 2])
        for i in range(1, nqubits - 1, 2):  # odd brick with OBC
            c.exp(
                i,
                i + 1,
                theta=1.0,
                unitary=gate_param[0] * (xx_t + yy_t) + gate_param[1] * zz_t,
            )
    return c


def measure_z(param, gate_param, i):
    c = ansatz(param, gate_param)
    # K.real(c.expectation_ps(x=[i, i+1])) for measure on <X_iX_i+1>
    return K.real(c.expectation_ps(z=[i]))


# Manual batching to avoid functorch vmap issues
_single_value_and_grad = K.value_and_grad(measure_z, argnums=0)

def batched_grads(param_batch, gate_param, measure_on):
    grads = []
    for b in range(param_batch.shape[0]):
        _, g = _single_value_and_grad(param_batch[b], gate_param, measure_on)
        grads.append(g)
    return K.stack(grads)

if __name__ == "__main__":
    # Reduced sizes to keep CI runtime within ~5-10s
    batch = 8
    reps = 3
    measure_on = nqubits // 2
    # which qubit the sigma_z observable is on

    # we will average `reps*batch` different unitaries
    rlist = []
    gate_param = np.array([0, np.pi / 4])
    gate_param = tq.array_to_tensor(gate_param)
    for _ in tqdm(range(reps)):
        param = np.random.uniform(0, 2 * np.pi, size=[batch, nlayers, 2, nqubits, 3])
        param = tq.array_to_tensor(param, dtype="float32")
        gs = batched_grads(param, gate_param, measure_on)
        # gs.shape = [batch, nlayers, 2, nqubits, 3]
        gs = K.abs(gs) ** 2
        rlist.append(gs)
    gs2 = K.stack(rlist)
    # gs2.shape = [reps, batch, nlayers, 2, nqubits, 3]
    gs2 = K.reshape(gs2, [-1, nlayers, 2, nqubits, 3])
    gs2_mean = K.numpy(
        K.mean(gs2, axis=0)
    )  # numpy array for the averaged abs square for gradient of each element
    print(gs2_mean)
