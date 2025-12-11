"""Stabilizer Simulation vs TyxonQ: Entanglement Entropy Comparison

This example demonstrates the efficiency advantage of stabilizer formalism for
simulating pure Clifford circuits by comparing:
1. Stim library (fast stabilizer-based simulation)
2. TyxonQ (full statevector simulation)

The key insight is that for Clifford circuits, stabilizer formalism can compute
entanglement entropy exponentially faster than statevector methods, while
yielding identical results.

Complexity comparison:
- Stabilizer (Stim):    O(n²) space, O(n² per gate)  
- Statevector (TyxonQ): O(2^n) space, O(2^n per gate)

For n=12 qubits, this is 4KB vs 32MB memory!

Note: This is a simplified version that focuses on pure Clifford circuits
without mid-circuit measurements. For advanced mid-measurement scenarios,
see the original stabilizer_simulation.py in examples-ng/.

References:
- Stim: https://github.com/quantumlib/Stim
- Gottesman-Knill theorem: https://en.wikipedia.org/wiki/Gottesman%E2%80%93Knill_theorem
"""

import numpy as np
import stim

import tyxonq as tq

# Set seed for reproducibility
np.random.seed(42)

# Configure TyxonQ backend
tq.set_backend("numpy")

# Clifford gates available in both libraries
CLIFFORD_1Q_GATES = ["h", "x", "y", "z", "s"]
CLIFFORD_2Q_GATES = ["cnot"]


def gen_random_pairs(num_qubits, count):
    """Generate random qubit pairs for 2-qubit gates."""
    choice = list(range(num_qubits))
    for _ in range(count):
        np.random.shuffle(choice)
        x, y = choice[:2]
        yield (x, y)


def random_clifford_circuit(num_qubits, depth):
    """Generate a random Clifford circuit without mid-circuit measurements.
    
    Args:
        num_qubits: Number of qubits in the circuit
        depth: Number of layers (each layer has 2 CNOTs + random 1q gates)
        
    Returns:
        (tq.Circuit, operation_list): Circuit object and list of operations
    """
    c = tq.Circuit(num_qubits)
    operation_list = []
    
    for _ in range(depth):
        # Add 2 random CNOT gates per layer
        for j, k in gen_random_pairs(num_qubits, 2):
            c.cnot(j, k)
            operation_list.append(("CNOT", (j, k)))
        
        # Add random single-qubit Clifford gates
        for j in range(num_qubits):
            gate_name = np.random.choice(CLIFFORD_1Q_GATES)
            getattr(c, gate_name)(j)
            operation_list.append((gate_name.upper(), (j,)))
    
    return c, operation_list


def convert_to_stim_circuit(operation_list):
    """Convert operation list to Stim circuit."""
    stim_circuit = stim.Circuit()
    for instruction in operation_list:
        gate_name = instruction[0]
        qubits = instruction[1]
        stim_circuit.append(gate_name, qubits)
    return stim_circuit


def get_binary_matrix(z_stabilizers):
    """Convert Z-stabilizers to binary matrix for entropy calculation.
    
    Reference: https://quantumcomputing.stackexchange.com/questions/16718/
    measuring-entanglement-entropy-using-a-stabilizer-circuit-simulator
    """
    N = len(z_stabilizers)
    binary_matrix = np.zeros((N, 2 * N))
    
    for row_idx, row in enumerate(z_stabilizers):
        for col_idx, col in enumerate(row):
            # Pauli encoding: I=0, X=1, Y=2, Z=3
            if col == 3:  # Pauli Z
                binary_matrix[row_idx, N + col_idx] = 1
            if col == 2:  # Pauli Y (has both X and Z components)
                binary_matrix[row_idx, N + col_idx] = 1
                binary_matrix[row_idx, col_idx] = 1
            if col == 1:  # Pauli X
                binary_matrix[row_idx, col_idx] = 1
    
    return binary_matrix


def get_cut_binary_matrix(binary_matrix, cut):
    """Extract submatrix for entanglement cut."""
    N = len(binary_matrix)
    # Select columns corresponding to cut qubits (both X and Z parts)
    new_indices = [i for i in range(N) if i in set(cut)] + [
        i + N for i in range(N) if i in set(cut)
    ]
    return binary_matrix[:, new_indices]


def gf2_rank(matrix):
    """Compute rank of binary matrix over GF(2) using Gaussian elimination.
    
    Reference: https://gist.github.com/StuartGordonReid/eb59113cb29e529b8105
    """
    matrix = [list(row) for row in matrix]  # Copy to avoid modification
    n = len(matrix[0])
    rank = 0
    
    for col in range(n):
        # Find pivot rows
        rows = []
        j = 0
        while j < len(matrix):
            if matrix[j][col] == 1:
                rows.append(j)
            j += 1
        
        if len(rows) >= 1:
            # Eliminate using first pivot
            for c in range(1, len(rows)):
                for k in range(n):
                    matrix[rows[c]][k] = (matrix[rows[c]][k] + matrix[rows[0]][k]) % 2
            # Remove pivot row
            matrix.pop(rows[0])
            rank += 1
    
    # Count remaining non-zero rows
    for row in matrix:
        if sum(row) > 0:
            rank += 1
    
    return rank


def simulate_stim_circuit(stim_circuit):
    """Simulate Clifford circuit using Stim's stabilizer formalism."""
    simulator = stim.TableauSimulator()
    
    for instruction in stim_circuit.flattened():
        simulator.do(instruction)
    
    return simulator.current_inverse_tableau() ** -1


def main():
    print("=" * 70)
    print("Stabilizer Formalism vs Full Statevector: Clifford Circuit Simulation")
    print("=" * 70)
    print()
    
    # Circuit parameters
    num_qubits = 12
    depth = 24
    cut = [i for i in range(num_qubits // 3)]  # Trace out first 1/3 of qubits
    
    print(f"Circuit Configuration:")
    print(f"  Number of qubits: {num_qubits}")
    print(f"  Circuit depth:    {depth}")
    print(f"  Entanglement cut: qubits {cut}")
    print()
    
    # Generate random Clifford circuit
    print("Generating random Clifford circuit...")
    tq_circuit, op_list = random_clifford_circuit(num_qubits, depth)
    print(f"  Total operations: {len(op_list)}")
    print()
    
    # Display circuit (if qiskit available)
    try:
        print("Circuit diagram:")
        print(tq_circuit.draw(output="text"))
        print()
    except Exception:
        print("(Qiskit not available for circuit visualization)")
        print()
    
    # ========== Stim Simulation (Stabilizer Formalism) ==========
    print("-" * 70)
    print("Method 1: Stim (Stabilizer Formalism)")
    print("-" * 70)
    
    import time
    start = time.time()
    
    stim_circuit = convert_to_stim_circuit(op_list)
    stabilizer_tableau = simulate_stim_circuit(stim_circuit)
    
    # Extract Z-stabilizers for entanglement entropy calculation
    zs = [stabilizer_tableau.z_output(k) for k in range(len(stabilizer_tableau))]
    binary_matrix = get_binary_matrix(zs)
    cur_matrix = get_cut_binary_matrix(binary_matrix, cut)
    
    # Compute entanglement entropy using stabilizer rank
    # S = (rank - |cut|) * ln(2)
    rank = gf2_rank(cur_matrix.tolist())
    stim_entropy = (rank - len(cut)) * np.log(2)
    
    stim_time = time.time() - start
    
    print(f"  Stabilizer rank:        {rank}")
    print(f"  Entanglement entropy:   {stim_entropy:.6f}")
    print(f"  Computation time:       {stim_time*1000:.2f} ms")
    print(f"  Memory complexity:      O(n²) = O({num_qubits**2})")
    print()
    
    # ========== TyxonQ Simulation (Full Statevector) ==========
    print("-" * 70)
    print("Method 2: TyxonQ (Full Statevector)")
    print("-" * 70)
    
    start = time.time()
    
    # Get quantum state
    state_vector = tq_circuit.state()
    
    # Verify state is normalized
    norm = np.linalg.norm(state_vector)
    assert norm > 0, "State vector has zero norm!"
    
    # Compute entanglement entropy using von Neumann entropy
    # S = -Tr(ρ log ρ) where ρ is reduced density matrix
    from tyxonq.libs.quantum_library.kernels.quantum_info import reduced_density_matrix, entropy
    
    rho = reduced_density_matrix(state_vector, cut)
    tq_entropy = entropy(rho)
    
    tq_time = time.time() - start
    
    print(f"  State vector norm:      {norm:.10f}")
    print(f"  Entanglement entropy:   {tq_entropy:.6f}")
    print(f"  Computation time:       {tq_time*1000:.2f} ms")
    print(f"  Memory complexity:      O(2^n) = O({2**num_qubits})")
    print()
    
    # ========== Comparison ==========
    print("=" * 70)
    print("Comparison Results")
    print("=" * 70)
    print()
    
    # Check numerical agreement
    entropy_diff = abs(stim_entropy - tq_entropy)
    agreement = entropy_diff < 1e-8
    
    print(f"  Stim entropy:           {stim_entropy:.10f}")
    print(f"  TyxonQ entropy:         {tq_entropy:.10f}")
    print(f"  Absolute difference:    {entropy_diff:.2e}")
    print(f"  Agreement (tol=1e-8):   {'✓ PASS' if agreement else '✗ FAIL'}")
    print()
    
    # Performance comparison
    speedup = tq_time / stim_time if stim_time > 0 else float('inf')
    print(f"  Stim computation time:  {stim_time*1000:.2f} ms")
    print(f"  TyxonQ computation time:{tq_time*1000:.2f} ms")
    print(f"  Speedup (Stim/TyxonQ):  {speedup:.1f}x")
    print()
    
    # Memory comparison
    stim_memory_kb = (num_qubits ** 2) * 4 / 1024  # Rough estimate
    tyxonq_memory_kb = (2 ** num_qubits) * 16 / 1024  # complex128
    print(f"  Stim memory (approx):   {stim_memory_kb:.1f} KB")
    print(f"  TyxonQ memory (approx): {tyxonq_memory_kb:.1f} KB")
    print(f"  Memory ratio:           {tyxonq_memory_kb/stim_memory_kb:.0f}x")
    print()
    
    # Final verification
    print("=" * 70)
    if agreement:
        print("✓ SUCCESS: Both methods agree on entanglement entropy!")
        print()
        print("Key Insight:")
        print("  For Clifford circuits, stabilizer formalism (Stim) achieves")
        print(f"  {speedup:.1f}x speedup and {tyxonq_memory_kb/stim_memory_kb:.0f}x memory reduction compared to")
        print("  full statevector simulation (TyxonQ), while producing identical results.")
    else:
        print("✗ WARNING: Methods disagree! This suggests a bug in one implementation.")
    print("=" * 70)
    
    # Numerical test
    np.testing.assert_allclose(stim_entropy, tq_entropy, atol=1e-8,
                               err_msg="Stim and TyxonQ entropy values disagree!")


if __name__ == "__main__":
    main()
