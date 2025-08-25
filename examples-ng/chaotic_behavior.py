"""
Some chaotic properties calculations from the circuit state.
"""

from functools import partial
import torch
import tyxonq as tq

K = tq.set_backend("pytorch")
N = 6
L = 3


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


params = torch.randn([L, N], requires_grad=True)

s = get_state(params, n=N, nlayers=L)

print(s)  # output state in vector form
rm = tq.quantum.reduced_density_matrix(s, cut=N // 2)
print(tq.quantum.entropy(rm))
# entanglement
# for more quantum quantities functions, please refer to tq.quantum module


def frame_potential(param1, param2, t, n, nlayers):
    s1 = get_state(param1, n, nlayers)
    s2 = get_state(param2, n, nlayers)
    inner = K.tensordot(K.conj(s1), s2, 1)
    return K.abs(inner) ** (2 * t)


# calculate several samples together using vmap

def frame_potential_batch(p1, p2):
    # p1, p2: [batch, L, N]
    b = p1.shape[0]
    vals = []
    for i in range(b):
        vals.append(frame_potential(p1[i], p2[i], t=1, n=N, nlayers=L))
    return K.stack(vals)

for _ in range(3):
    print(frame_potential_batch(torch.randn([3, L, N]), torch.randn([3, L, N])))
    # the first dimension is the batch

# get \partial \psi_i/ \partial \params_j (Jacobian)

try:
    jac_func = K.jacfwd(partial(get_state, n=N, nlayers=L))
    jac_val = jac_func(params)
    print(jac_val)
except Exception as e:
    print("jacfwd skipped:", str(e)[:120])

# correlation


def get_zz(params, n, nlayers, inputs=None):
    s = get_state(params, n, nlayers, inputs)
    c = tq.Circuit(n, inputs=s)
    z1z2 = c.expectation([tq.gates.z(), [1]], [tq.gates.z(), [2]])
    return K.real(z1z2)


# hessian matrix

# warning pytorch might be unable to do this
try:
    h_func = K.hessian(partial(get_zz, n=N, nlayers=L))
    print(h_func(params))
except Exception as e:
    print("hessian skipped:", str(e)[:120])

# optimization, suppose the energy we want to minimize is just z1z2 as above

vg_func = K.value_and_grad(get_zz)
opt = torch.optim.Adam([params], lr=1e-2)

for i in range(20):  # gradient descent (reduced for CI)
    opt.zero_grad()
    energy, grads = vg_func(params, N, L)
    energy.backward()
    opt.step()
    if i % 20 == 0:
        print(energy)  # see energy optimization dynamics
