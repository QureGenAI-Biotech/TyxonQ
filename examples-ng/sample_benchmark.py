"""
Sampling benchmark (chain-style API) with backend comparison.

- Use numeric backend RNG uniformly (nb.rng / nb.choice)
- Compare numpy, pytorch, cupynumeric (if available)
"""

from __future__ import annotations

import argparse
import time
import tyxonq as tq
from tyxonq.numerics.api import get_backend


def construct_circuit(n: int, nlayers: int, *, seed: int) -> tq.Circuit:
    c = tq.Circuit(n)
    for i in range(n):
        c.h(i)
    nb = get_backend(None)
    rng = nb.rng(seed)
    for _ in range(nlayers):
        for i in range(n):
            # uniform in [1, n-1]
            r = int(nb.choice(rng, n - 1, size=1)[0]) + 1
            c.cnot(i, (i + r) % n)
    for q in range(n):
        c.measure_z(q)
    return c


def run_once(c: tq.Circuit, shots: int) -> float:
    t0 = time.perf_counter()
    _ = (
        c.device(provider="simulator", device="statevector", shots=shots)
         .postprocessing(method=None)
         .run()
    )
    t1 = time.perf_counter()
    return (t1 - t0)


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    m = sum(values) / len(values)
    v = sum((x - m) * (x - m) for x in values) / len(values)
    return m, v ** 0.5


def benchmark_backend(backend_name: str, n_list: list[int], layers_list: list[int], shots: int, repeats: int, seed: int) -> None:
    try:
        nb = tq.set_backend(backend_name)
    except Exception as e:
        print(f"skip backend {backend_name}: {e}")
        return
    nb_name = getattr(nb, "name", str(backend_name)) if nb is not None else str(backend_name)
    print(f"\nBackend = {nb_name}")
    for n in n_list:
        for nl in layers_list:
            c = construct_circuit(n, nl, seed=seed)
            times: list[float] = []
            for _ in range(repeats):
                times.append(run_once(c, shots))
            avg, std = mean_std(times)
            print(f"n={n:2d}, layers={nl:2d}, shots={shots:5d} | time_avg={avg:.4f}s, time_std={std:.4f}s")

#TODO 增加不同backend特性的测试 例如 tensor的处理

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shots", type=int, default=8192)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n", type=int, nargs="*", default=[8, 10])
    parser.add_argument("--layers", type=int, nargs="*", default=[2, 6])
    args = parser.parse_args()

    # args.n = [8, 10]
    # args.layers = [2, 6]
    # args.shots = 8192
    # args.repeats = 3
    # args.seed = 0

    for b in ["numpy", "pytorch", "cupynumeric"]:
        benchmark_backend(b, args.n, args.layers, args.shots, args.repeats, args.seed)


if __name__ == "__main__":
    main()
