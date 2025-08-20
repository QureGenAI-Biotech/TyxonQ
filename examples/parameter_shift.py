"""
Demonstration on the correctness and efficiency of parameter shift gradient implementation
"""

import sys
import numpy as np
import torch

sys.path.insert(0, "../")

import tyxonq as tq
from tyxonq import experimental as E

# use backend handle K for portability across backends
K = tq.set_backend("pytorch")

n = 6
m = 3


def f1(param):
    c = tq.Circuit(n)
    for j in range(m):
        for i in range(n - 1):
            c.cnot(i, i + 1)
        for i in range(n):
            c.rx(i, theta=param[i, j])
    return K.real(c.expectation_ps(y=[n // 2]))


g1f1 = K.jit(K.grad(f1))

r1, ts, tr = tq.utils.benchmark(g1f1, K.ones([n, m], dtype="float32"))

g2f1 = K.jit(E.parameter_shift_grad(f1))

r2, ts, tr = tq.utils.benchmark(g2f1, K.ones([n, m], dtype="float32"))

np.testing.assert_allclose(
    r1.detach().cpu().numpy(), r2.detach().cpu().numpy(), atol=1e-5
)
print("equality test passed!")

# multiple weights args version


def f2(paramzz, paramx):
    c = tq.Circuit(n)
    for j in range(m):
        for i in range(n - 1):
            c.rzz(i, i + 1, theta=paramzz[i, j])
        for i in range(n):
            c.rx(i, theta=paramx[i, j])
    return K.real(c.expectation_ps(y=[n // 2]))


g1f2 = K.jit(K.grad(f2, argnums=(0, 1)))

r12, ts, tr = tq.utils.benchmark(
    g1f2, K.ones([n, m], dtype="float32"), K.ones([n, m], dtype="float32")
)

g2f2 = K.jit(E.parameter_shift_grad(f2, argnums=(0, 1)))

r22, ts, tr = tq.utils.benchmark(
    g2f2, K.ones([n, m], dtype="float32"), K.ones([n, m], dtype="float32")
)

np.testing.assert_allclose(
    r12[0].detach().cpu().numpy(), r22[0].detach().cpu().numpy(), atol=1e-5
)
np.testing.assert_allclose(
    r12[1].detach().cpu().numpy(), r22[1].detach().cpu().numpy(), atol=1e-5
)
print("multiple weight inputs: equality test passed!")

# sampled expectation version


def f3(param):
    c = tq.Circuit(n)
    for j in range(m):
        for i in range(n - 1):
            c.cnot(i, i + 1)
        for i in range(n):
            c.rx(i, theta=param[i, j])
    return K.real(c.sample_expectation_ps(y=[n // 2]))


g2f3 = K.jit(E.parameter_shift_grad(f3))

r2, ts, tr = tq.utils.benchmark(g2f3, K.ones([n, m], dtype="float32"))

np.testing.assert_allclose(
    r1.detach().cpu().numpy(), r2.detach().cpu().numpy(), atol=1e-5
)
print("analytical sampled expectation: equality test passed!")


# def f3(param):
#     c = tq.Circuit(n)
#     for j in range(m):
#         for i in range(n - 1):
#             c.cnot(i, i + 1)
#         for i in range(n):
#             c.rx(i, theta=param[i, j])
#     return K.real(c.sample_expectation_ps(y=[n // 2], shots=81920))


# g2f3 = K.jit(E.parameter_shift_grad(f3))

# r2, ts, tr = tq.utils.benchmark(g2f3, K.ones([n, m], dtype="float32"))
# print(r1 - r2)
# np.testing.assert_allclose(r1 - r2, np.zeros_like(r1.detach().cpu().numpy()), atol=1e-3)
# print("finite sampled expectation: equality test passed!")
