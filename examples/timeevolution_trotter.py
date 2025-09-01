"""
Heisenberg time evolution via Trotterization (refactored to new API).

This example builds a Trotterized circuit from Pauli terms and runs it on the
local statevector simulator using the chainable run API. It avoids legacy
methods (exp1/state/expectation_ps).
"""

import tyxonq as tq
from tyxonq.libs.circuits_library.trotter_circuit import build_trotter_circuit


def heisenberg_chain_terms(num_qubits: int):
    # Encode Pauli strings as lists with 0=I,1=X,2=Y,3=Z on each qubit
    terms = []
    weights = []
    jx = jy = jz = 1.0
    for i in range(num_qubits - 1):
        # XX on neighbors i,i+1
        t_xx = [0] * num_qubits
        t_xx[i] = 1; t_xx[i + 1] = 1
        terms.append(t_xx); weights.append(jx)
        # YY
        t_yy = [0] * num_qubits
        t_yy[i] = 2; t_yy[i + 1] = 2
        terms.append(t_yy); weights.append(jy)
        # ZZ
        t_zz = [0] * num_qubits
        t_zz[i] = 3; t_zz[i + 1] = 3
        terms.append(t_zz); weights.append(jz)
    return terms, weights


def run_demo(num_qubits: int = 4, total_time: float = 0.5, steps: int = 10):
    tq.set_backend("numpy")
    terms, weights = heisenberg_chain_terms(num_qubits)
    c = build_trotter_circuit(terms, weights=weights, time=total_time, steps=steps, num_qubits=num_qubits)
    # Add measurements to observe Z on each qubit
    for q in range(num_qubits):
        c.measure_z(q)
    results = (
        c.compile()
         .device(provider="local", device="statevector", shots=0)
         .postprocessing(method=None)
         .run()
    )
    print("Simulator results:", results)
    # Also demonstrate omit-style auto completion: direct .run() works
    c2 = build_trotter_circuit(terms, weights=weights, time=total_time, steps=steps, num_qubits=num_qubits)
    for q in range(num_qubits):
        c2.measure_z(q)
    auto_results = c2.run()
    print("Auto-completed run results:", auto_results)
    return results


if __name__ == "__main__":
    run_demo()
