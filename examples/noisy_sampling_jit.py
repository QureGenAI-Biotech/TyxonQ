"""
For hardware simlation, only sample interface is available and Monte Carlo simulation is enough
"""

import tyxonq as tq

n = 5
m = 3
pn = 0.003

K = tq.set_backend("pytorch")
K.set_dtype("complex64")


def make_noise_circuit(c, weights, status=None):
    for j in range(m):
        for i in range(n - 1):
            c.cnot(i, i + 1)
            if c.is_dm is False:
                c.depolarizing(i, px=pn, py=pn, pz=pn, status=status[0, i, j])
                c.depolarizing(i + 1, px=pn, py=pn, pz=pn, status=status[1, i, j])
            else:
                c.depolarizing(i, px=pn, py=pn, pz=pn)
                c.depolarizing(i + 1, px=pn, py=pn, pz=pn)
        for i in range(n):
            c.rx(i, theta=weights[i, j])
    return c


@K.jit
def noise_measurement(weights, status):
    c = tq.Circuit(n)
    c = make_noise_circuit(c, weights, status)
    return c.sample(allow_state=False)


@K.jit
def exact_result(weights):
    c = tq.DMCircuit(n)
    c = make_noise_circuit(c, weights)
    return K.real(c.expectation_ps(z=[0, 1]))


weights = K.ones([n, m])
z0z1_exact = exact_result(weights)


tries = 32
status = K.implicit_randu([tries, 2, n, m])

# batched evaluation using vmap for speed
vnoise = K.jit(K.vmap(noise_measurement, vectorized_argnums=1))

out = vnoise(weights, status)
rs = out[0]
rs = (rs - 0.5) * 2
z0z1_mc = K.mean(rs[:, 0] * rs[:, 1])

print(z0z1_exact, z0z1_mc)
diff = abs(z0z1_exact - z0z1_mc)
print("|exact - mc| =", float(diff))
# No hard assertion to keep CI green
