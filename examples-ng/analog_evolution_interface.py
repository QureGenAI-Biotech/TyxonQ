"""
Analog time-evolution interface demo (refactored).

This example shows how to:
- Build a simple Hamiltonian from Pauli strings
- Construct a Trotterized evolution circuit
- Run on the local statevector simulator and view Z-expectations
- Optionally compare with a small numeric evolve (dense) on CPU
"""

import numpy as np
import tyxonq as tq
from tyxonq.libs.circuits_library.trotter_circuit import build_trotter_circuit
from tyxonq.libs.quantum_library.dynamics import (
    PauliSumCOO as PauliStringSum2COO,  # compat alias for example text
    evolve_state as evolve_state_numeric,  # compat alias for example text
    expectation as expval_dense,  # compat alias for example text
)


def build_demo_hamiltonian():
    # Two-qubit Hamiltonian: H = 1.0 * Z0 Z1 + 0.5 * X0
    # Pauli encoding: 0=I,1=X,2=Y,3=Z
    # Z0Z1 -> [3,3];  X0 -> [1,0]
    terms = [[3, 3], [1, 0]]
    weights = [1.0, 0.5]
    return terms, weights


def run_trotter_example(time: float = 1.0, steps: int = 8):
    terms, weights = build_demo_hamiltonian()

    # Build trotterized circuit
    c = build_trotter_circuit(terms, weights=weights, time=time, steps=steps, num_qubits=2)

    # Run on local simulator (chain-style with postprocessing)
    results = (
        c.compile()
         .device(provider="local", device="statevector", shots=0)
         .postprocessing(method=None)
         .run()
    )
    for result in results:
        print("Simulator result:", result)


def compare_numeric(time: float = 1.0, steps: int = 256):
    terms, weights = build_demo_hamiltonian()
    # Build dense H for small n
    H = PauliStringSum2COO(terms, weights).to_dense()
    # Start from |11>
    psi0 = np.zeros(4, dtype=np.complex128); psi0[-1] = 1.0
    psi_t = evolve_state_numeric(H, psi0, time, steps=steps)
    e = expval_dense(psi_t, H)
    print("Numeric evolve expval(H):", e)


def main():
    tq.set_backend("pytorch")  # or "numpy"
    run_trotter_example(time=1.0, steps=8)
    compare_numeric(time=1.0, steps=256)


if __name__ == "__main__":
    main()
