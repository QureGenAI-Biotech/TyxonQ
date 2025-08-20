"""
Time evolution of Heisenberg model realized by Trotter decomposition
"""

import numpy as np
import tyxonq as tq

K = tq.set_backend("pytorch")
K.set_dtype("complex128")

xx = tq.gates._xx_matrix
yy = tq.gates._yy_matrix
zz = tq.gates._zz_matrix

nqubit = 4
t = 0.5
tau = 0.1


def Trotter_step_unitary(input_state, tau, nqubit):
    c = tq.Circuit(nqubit, inputs=input_state)
    for i in range(nqubit - 1):  ### U_zz
        c.exp1(i, i + 1, theta=tau, unitary=zz)
    for i in range(nqubit - 1):  ### U_yy
        c.exp1(i, i + 1, theta=tau, unitary=yy)
    for i in range(nqubit - 1):  ### U_xx
        c.exp1(i, i + 1, theta=tau, unitary=xx)
    TSUstate = c.state()  ### return state U(τ)|ψ_i>
    z0 = c.expectation_ps(z=[0])
    return TSUstate, z0


TSU_vmap = K.jit(
    K.vmap(
        Trotter_step_unitary,
        vectorized_argnums=0,
    )
)

ninput = 2
input_state = np.zeros((ninput, 2**nqubit))
input_state[0, 0] = 1.0
input_state[1, -1] = 1.0

for _ in range(int(t / tau)):
    input_state, z0 = TSU_vmap(input_state, tau, nqubit)
    print("z: ", z0)
