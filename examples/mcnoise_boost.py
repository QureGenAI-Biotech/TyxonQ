"""
Boost the Monte Carlo noise simulation (specifically the staging time)
on general error with circuit layerwise slicing
"""

import time
import sys

sys.path.insert(0, "../")
import tyxonq as tq

K = tq.set_backend("pytorch")

n = 3  # 10
nlayer = 2  # 4


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


def templatecnot(s, param, i):
    c = tq.Circuit(n, inputs=s)
    c.cnot(i, i + 1)
    return c.state()


def templatenoise(seed, s, param, i):
    c = tq.Circuit(n, inputs=s)
    status = torch.rand(1)[0]
    c.apply_general_kraus(tq.channels.phasedampingchannel(0.15), i, status=status)
    return c.state()


def templaterz(s, param, j):
    c = tq.Circuit(n, inputs=s)
    for i in range(n):
        c.rx(i, theta=param[j, i])
    return c.state()


def f2(seed, param, n, nlayer):
    c = tq.Circuit(n)
    for i in range(n):
        c.h(i)
    s = c.state()
    for j in range(nlayer):
        for i in range(n - 1):
            s = templatecnot(s, param, i)
            s = templatenoise(seed + j * n + i, s, param, i)
            s = templatenoise(seed + j * n + i + 1, s, param, i + 1)
        s = templaterz(s, param, j)
    return K.real(tq.expectation((tq.gates.z(), [int(n / 2)]), ket=s))


# warning pytorch might be unable to do this exactly
vagf1 = K.jit(K.value_and_grad(f1, argnums=1), static_argnums=(2, 3))
# warning pytorch might be unable to do this exactly
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

# 10*4: pytorch*T4: 235/0.36 vs. 26/0.04
