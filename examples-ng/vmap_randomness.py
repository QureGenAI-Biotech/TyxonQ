"""
Interplay between jit, vmap, randomness and backend
"""

import tyxonq as tq

K = tq.set_backend("pytorch")
n = 10
batch = 100

print("pytorch backend")


@K.jit
def f(a):
    return a + tq.backend.randn([n])


vf = K.jit(K.vmap(f))

from tyxonq import utils

r, _, _ = utils.benchmark(f, K.ones([n], dtype="float32"))
print(r)

r, _, _ = utils.benchmark(vf, K.ones([batch, n], dtype="float32"))
print(r[:2])
