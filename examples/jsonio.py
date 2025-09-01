"""
JSON IO demo using the new chain-style API.

- Build a small circuit with supported ops
- Serialize to JSON string and file
- Load back and run both, compare counts-based expectations
"""

from __future__ import annotations

import json
import tyxonq as tq
from tyxonq.postprocessing.metrics import expectation


def build_demo_circuit() -> tq.Circuit:
    c = tq.Circuit(3)
    c.h(0).rx(1, theta=0.3).rz(2, theta=-0.5).cx(0, 2)
    c.measure_z(0).measure_z(1).measure_z(2)
    return c


def run_counts(c: tq.Circuit, shots: int = 4096):
    res = (
        c.device(provider="simulator", device="statevector", shots=shots)
         .postprocessing(method=None)
         .run()
    )
    payload = res if isinstance(res, dict) else (res[0] if res else {})
    return payload.get("result", {}), payload.get("metadata", {})


def main():
    c = build_demo_circuit()

    # Serialize to JSON string
    s = c.to_json_str(indent=2)
    print("JSON string (truncated):", s[:120], "...")

    # Save to file
    with open("circuit.json", "w", encoding="utf-8") as f:
        f.write(s)

    # Load from string
    c2 = tq.Circuit.from_json_str(s)

    # Load from file
    with open("circuit.json", "r", encoding="utf-8") as f:
        s_file = f.read()
    c3 = tq.Circuit.from_json_str(s_file)

    # Run and compare counts-based Z expectations
    counts1, _ = run_counts(c, shots=8192)
    counts2, _ = run_counts(c2, shots=8192)
    counts3, _ = run_counts(c3, shots=8192)

    ez1 = expectation(counts1, z=[0, 1, 2])
    ez2 = expectation(counts2, z=[0, 1, 2])
    ez3 = expectation(counts3, z=[0, 1, 2])
    print("E[Z⊗Z⊗Z] (orig, from_str, from_file):", ez1, ez2, ez3)

    # Draw one for visualization
    print("\nCircuit draw:\n", c.draw())


    # c.h(0)
    # c.h(2)
    # c.cnot(1, 2)
    # c.rxx(0, 2, theta=0.3)
    # c.u(2, theta=0.2, lbd=-1.2, phi=0.5)
    # c.cu(1, 0, lbd=1.0)
    # c.crx(0, 1, theta=-0.8)
    # c.r(1, theta=tq.backend.ones([]), alpha=0.2)
    # c.toffoli(0, 2, 1)
    # c.ccnot(0, 1, 2)
    # c.any(0, 1, unitary=tq.gates._xx_matrix)
    # c.multicontrol(1, 2, 0, ctrl=[0, 1], unitary=tq.gates._x_matrix)

if __name__ == "__main__":
    main()
