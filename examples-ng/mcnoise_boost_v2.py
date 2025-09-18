"""
Boost the Monte Carlo noise simulation (specifically the staging time)
on general error with circuit layerwise slicing: new paradigm,
essentially the same as v1, but much simpler
"""

import time
import sys
import torch

sys.path.insert(0, "../")
import tyxonq as tq

K = tq.set_backend("pytorch")

n = 6  # 10
nlayer = 5  # 4


def precompute(c):
    s = c.state()
    return tq.Circuit(c._nqubits, inputs=s)


def f1(seed, param, n, nlayer):
    if seed is not None:
        torch.manual_seed(seed)
    c = tq.Circuit(n)
    for i in range(n):
        c.h(i)
    for j in range(nlayer):
        for i in range(n - 1):
            c.cnot(i, i + 1)
            c.apply_general_kraus(tq.channels.phasedampingchannel(0.15), i)
            c.apply_general_kraus(tq.channels.phasedampingchannel(0.15), i + 1)
        for i in range(n):
            c.rx(i, theta=param[j, i])
    return K.real(c.expectation((tq.gates.z(), [int(n / 2)])))


def f2(seed, param, n, nlayer):
    c = tq.Circuit(n)
    for i in range(n):
        c.h(i)
    for j in range(nlayer):
        for i in range(n - 1):
            c.cnot(i, i + 1)
            c = precompute(c)
            c.apply_general_kraus(tq.channels.phasedampingchannel(0.15), i)
            c = precompute(c)
            c.apply_general_kraus(tq.channels.phasedampingchannel(0.15), i + 1)
        for i in range(n):
            c.rx(i, theta=param[j, i])
    return K.real(c.expectation((tq.gates.z(), [int(n / 2)])))


vagf1 = K.jit(K.value_and_grad(f1, argnums=1), static_argnums=(2, 3))
vagf2 = K.jit(K.value_and_grad(f2, argnums=1), static_argnums=(2, 3))

param = K.ones([nlayer, n])


def benchmark(f, tries=3):
    time0 = time.time()
    seed = 42
    print(f(seed, param, n, nlayer)[0])
    time1 = time.time()
    for _ in range(tries):
        print(f(seed, param, n, nlayer)[0])
    time2 = time.time()
    print(
        "staging time: ",
        time1 - time0,
        "running time: ",
        (time2 - time1) / tries,
    )


print("without layerwise slicing jit")
benchmark(vagf1)
print("=============================")
print("with layerwise slicing jit")
benchmark(vagf2)

# mac16 intel cpu: (6*5, pytorch) 1015, 0.0035; 31.68, 0.00082
