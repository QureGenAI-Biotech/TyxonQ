"""
Some chaotic properties calculations from the circuit state.
"""

from functools import partial
import sys

sys.path.insert(0, "../")
import torch
import tyxonq as tq

K = tq.set_backend("pytorch")


# build the circuit


# warning pytorch might be unable to do this
def get_state(params, n, nlayers, inputs=None):
    c = tq.Circuit(n, inputs=inputs)  # inputs is for input state of the circuit
    for i in range(nlayers):
        for j in range(n):
            c.ry(j, theta=params[i, j])
        for j in range(n):
            c.cnot(j, (j + 1) % n)
    # one can further customize the layout and gate type above
    return c.state()  # output wavefunction


params = torch.randn([5, 10], requires_grad=True)

s = get_state(params, n=10, nlayers=5)

print(s)  # output state in vector form
rm = tq.quantum.reduced_density_matrix(s, cut=10 // 2)
print(tq.quantum.entropy(rm))
# entanglement
# for more quantum quantities functions, please refer to tq.quantum module


# warning pytorch might be unable to do this
def frame_potential(param1, param2, t, n, nlayers):
    s1 = get_state(param1, n, nlayers)
    s2 = get_state(param2, n, nlayers)
    inner = K.tensordot(K.conj(s1), s2, 1)
    return K.abs(inner) ** (2 * t)


# calculate several samples together using vmap

frame_potential_vmap = K.vmap(
    partial(frame_potential, t=1, n=10, nlayers=5), vectorized_argnums=(0, 1)
)

for _ in range(5):
    print(frame_potential_vmap(torch.randn([3, 5, 10]), torch.randn([3, 5, 10])))
    # the first dimension is the batch

# get \partial \psi_i/ \partial \params_j (Jacobian)

jac_func = K.jacfwd(partial(get_state, n=10, nlayers=5))
print(jac_func(params))

# correlation


def get_zz(params, n, nlayers, inputs=None):
    s = get_state(params, n, nlayers, inputs)
    c = tq.Circuit(n, inputs=s)
    z1z2 = c.expectation([tq.gates.z(), [1]], [tq.gates.z(), [2]])
    return K.real(z1z2)


# hessian matrix

# warning pytorch might be unable to do this
h_func = K.hessian(partial(get_zz, n=10, nlayers=5))
# suggest jax backend for hessian and directly use `jax.hessian` may be better
print(h_func(params))

# optimization, suppose the energy we want to minimize is just z1z2 as above

vg_func = K.value_and_grad(get_zz)
opt = torch.optim.Adam([params], lr=1e-2)

for i in range(200):  # gradient descent
    opt.zero_grad()
    energy, grads = vg_func(params, 10, 5)
    energy.backward()
    opt.step()
    if i % 20 == 0:
        print(energy)  # see energy optimization dynamics
