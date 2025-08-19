"""
For hardware simlation, only sample interface is available and Monte Carlo simulation is enough
"""

import tyxonq as tq

n = 6
m = 4
pn = 0.003

K = tq.set_backend("pytorch")


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


@K.jit  # warning pytorch might be unable to do this exactly
def noise_measurement(weights, status):
    c = tq.Circuit(n)
    c = make_noise_circuit(c, weights, status)
    return c.sample(allow_state=True)


@K.jit  # warning pytorch might be unable to do this exactly
def exact_result(weights):
    c = tq.DMCircuit(n)
    c = make_noise_circuit(c, weights)
    return K.real(c.expectation_ps(z=[0, 1]))


weights = K.ones([n, m])
z0z1_exact = exact_result(weights)


tries = 2**15
status = K.implicit_randu([tries, 2, n, m])

# a micro benchmarking

tq.utils.benchmark(noise_measurement, weights, status[0])


rs = []
for i in range(tries):
    # can also be vmapped, but a tradeoff between number of trials here for further jit
    r = noise_measurement(weights, status[i])
    rs.append(r[0])

rs = (K.stack(rs) - 0.5) * 2
z0z1_mc = K.mean(rs[:, 0] * rs[:, 1])

print(z0z1_exact, z0z1_mc)

assert abs(z0z1_exact - z0z1_mc) < 0.03
