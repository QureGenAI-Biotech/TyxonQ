"""Demonstration of ZZ Crosstalk Noise Modeling (P1.2).

This example shows how to model realistic ZZ crosstalk noise in superconducting
quantum processors. ZZ crosstalk is an always-on coherent interaction between
neighboring qubits that causes unwanted conditional phase accumulation.

Physical Background
-------------------
In superconducting quantum processors, qubits are coupled through capacitive
or inductive interactions. Even when gates are not being applied, residual
coupling causes a ZZ interaction:

    H_ZZ = ξ · σ_z ⊗ σ_z

This leads to conditional phase errors that accumulate during:
1. Idle times (when one qubit waits for another)
2. Single-qubit gates (spectator qubit errors)
3. Measurement operations

Typical ZZ coupling strengths:
- IBM transmons: 1-5 MHz (arXiv:2108.12323)
- Google Sycamore: 0.1-1 MHz (tunable couplers reduce ZZ)
- Rigetti: 2-10 MHz (always-on coupling)

Author: TyxonQ Development Team
"""

import numpy as np
import scipy.linalg
import tyxonq as tq
from tyxonq.libs.quantum_library.noise import zz_crosstalk_hamiltonian
from tyxonq.libs.quantum_library.pulse_physics import (
    get_qubit_topology,
    get_crosstalk_couplings,
)


def example1_basic_zz_crosstalk():
    """Example 1: Basic ZZ crosstalk for 2 qubits."""
    print("=" * 70)
    print("Example 1: Basic ZZ Crosstalk Hamiltonian")
    print("=" * 70)
    
    # IBM typical ZZ coupling: ~3 MHz
    xi = 3e6  # Hz
    
    # Build ZZ Hamiltonian
    H_ZZ = zz_crosstalk_hamiltonian(xi, num_qubits=2)
    
    print(f"\nZZ coupling strength: ξ = {xi/1e6:.1f} MHz")
    print(f"Hamiltonian shape: {H_ZZ.shape}")
    print(f"\nZZ Hamiltonian (in units of ξ):")
    print(H_ZZ / xi)
    
    # Eigenvalues
    eigenvalues = np.linalg.eigvalsh(H_ZZ)
    print(f"\nEigenvalues: {eigenvalues / xi} × ξ")
    
    # Time evolution
    t = 100e-9  # 100 ns (typical single-qubit gate time)
    U_ZZ = scipy.linalg.expm(-1j * H_ZZ * t)
    
    # Conditional phase accumulated on |11⟩ state
    phi_zz = xi * t  # radians
    print(f"\nConditional phase after {t*1e9:.0f} ns:")
    print(f"  φ_ZZ = {phi_zz:.4f} rad = {phi_zz * 180/np.pi:.2f}°")
    
    # Effect on Bell state fidelity
    # Ideal Bell state: (|00⟩ + |11⟩) / √2
    # With ZZ: (|00⟩ + e^{iφ} |11⟩) / √2
    fidelity_loss = (1 - np.cos(phi_zz)) / 2
    print(f"  Bell state fidelity loss: {fidelity_loss * 100:.4f}%")
    
    print()


def example2_linear_chain_topology():
    """Example 2: 5-qubit linear chain with ZZ crosstalk."""
    print("=" * 70)
    print("Example 2: 5-Qubit Linear Chain Topology")
    print("=" * 70)
    
    # Create 5-qubit linear chain (like IBM Yorktown)
    topo = get_qubit_topology(
        num_qubits=5,
        topology="linear",
        zz_strength=3e6  # 3 MHz (IBM typical)
    )
    
    print(f"\nTopology: {topo.topology_type}")
    print(f"Number of qubits: {topo.num_qubits}")
    print(f"Edges (connected pairs): {topo.edges}")
    
    # Get realistic couplings for IBM transmon
    couplings = get_crosstalk_couplings(topo, qubit_model="transmon_ibm")
    
    print(f"\nZZ coupling strengths:")
    for edge, xi in couplings.items():
        print(f"  Qubits {edge}: {xi/1e6:.1f} MHz")
    
    # Analyze crosstalk map
    print(f"\nCrosstalk neighbor map:")
    for qubit in range(topo.num_qubits):
        neighbors = topo.get_neighbors(qubit)
        print(f"  Qubit {qubit}: neighbors {neighbors}")
    
    print()


def example3_grid_topology():
    """Example 3: 2D grid topology (3x3 qubits)."""
    print("=" * 70)
    print("Example 3: 2D Grid Topology (3×3 qubits)")
    print("=" * 70)
    
    # Create 3x3 grid
    topo = get_qubit_topology(
        num_qubits=9,
        topology="grid",
        grid_shape=(3, 3),
        zz_strength=2.5e6  # 2.5 MHz
    )
    
    print(f"\nTopology: {topo.topology_type}")
    print(f"Grid shape: 3 × 3")
    print(f"Number of edges: {len(topo.edges)}")
    
    # Visualize grid connectivity
    print(f"\nGrid layout:")
    print("  0 -- 1 -- 2")
    print("  |    |    |")
    print("  3 -- 4 -- 5")
    print("  |    |    |")
    print("  6 -- 7 -- 8")
    
    # Center qubit (4) has most neighbors
    center = 4
    neighbors = topo.get_neighbors(center)
    print(f"\nCenter qubit ({center}) neighbors: {neighbors}")
    print(f"  → Most susceptible to ZZ crosstalk!")
    
    # Corner qubit (0) has fewer neighbors
    corner = 0
    neighbors = topo.get_neighbors(corner)
    print(f"Corner qubit ({corner}) neighbors: {neighbors}")
    print(f"  → Less ZZ crosstalk")
    
    print()


def example4_ibm_heavy_hex():
    """Example 4: IBM Heavy-Hex topology (27 qubits)."""
    print("=" * 70)
    print("Example 4: IBM Heavy-Hex Topology (27 qubits)")
    print("=" * 70)
    
    # IBM Eagle/Heron processors use Heavy-Hex topology
    topo = get_qubit_topology(
        num_qubits=27,
        topology="heavy_hex",
        zz_strength=4e6  # 4 MHz (IBM Eagle typical)
    )
    
    print(f"\nTopology: {topo.topology_type}")
    print(f"Number of qubits: {topo.num_qubits}")
    print(f"Number of edges: {len(topo.edges)}")
    print(f"Average degree: {2 * len(topo.edges) / topo.num_qubits:.2f}")
    
    # Heavy-Hex layer structure: 5-4-5-4-5-4
    print(f"\nHeavy-Hex structure: 5-4-5-4-5-4 layers")
    print(f"  → Optimized for CNOT gates with reduced crosstalk")
    
    print()


def example5_custom_topology():
    """Example 5: Custom topology with asymmetric ZZ couplings."""
    print("=" * 70)
    print("Example 5: Custom Topology with Asymmetric Couplings")
    print("=" * 70)
    
    # Triangle topology with different coupling strengths
    edges = [(0, 1), (1, 2), (0, 2)]
    custom_couplings = {
        (0, 1): 5e6,   # Strong coupling: 5 MHz
        (1, 2): 3e6,   # Medium coupling: 3 MHz
        (0, 2): 0.5e6  # Weak coupling: 0.5 MHz (far apart)
    }
    
    topo = get_qubit_topology(
        num_qubits=3,
        topology="custom",
        edges=edges,
        custom_couplings=custom_couplings
    )
    
    print(f"\nCustom triangle topology:")
    print("     0")
    print("    /  \\")
    print("   /    \\")
    print("  1 ---- 2")
    
    print(f"\nAsymmetric ZZ couplings:")
    for edge, xi in topo.zz_couplings.items():
        print(f"  Edge {edge}: {xi/1e6:.1f} MHz")
    
    # Implication for gate scheduling
    print(f"\nScheduling implication:")
    print(f"  ⚠️ Cannot run parallel gates on (0,1) - strong ZZ!")
    print(f"  ✅ Can run parallel gates on (0,2) - weak ZZ")
    
    print()


def example6_zz_crosstalk_time_evolution():
    """Example 6: Time evolution with ZZ crosstalk."""
    print("=" * 70)
    print("Example 6: Time Evolution with ZZ Crosstalk")
    print("=" * 70)
    
    xi = 3e6  # 3 MHz
    H_ZZ = zz_crosstalk_hamiltonian(xi, num_qubits=2)
    
    # Simulate idle crosstalk during a 100 ns X gate on qubit 0
    # (while qubit 1 is idle)
    t_gate = 100e-9  # 100 ns
    
    # Initial state: |01⟩ (qubit 0 in |0⟩, qubit 1 in |1⟩)
    psi_0 = np.array([0, 1, 0, 0], dtype=np.complex128)  # |01⟩
    
    # Time evolution
    U_ZZ = scipy.linalg.expm(-1j * H_ZZ * t_gate)
    psi_final = U_ZZ @ psi_0
    
    # Accumulated phase
    phi_expected = -xi * t_gate  # |01⟩ has ZZ eigenvalue -xi
    phase_measured = np.angle(psi_final[1])  # Extract phase of |01⟩ component
    
    print(f"\nInitial state: |01⟩")
    print(f"Gate time: {t_gate*1e9:.0f} ns")
    print(f"\nZZ-induced phase:")
    print(f"  Expected: {phi_expected:.4f} rad = {phi_expected*180/np.pi:.2f}°")
    print(f"  Measured: {phase_measured:.4f} rad = {phase_measured*180/np.pi:.2f}°")
    
    # Compare to gate error budget
    single_qubit_error_budget = 1e-3  # 0.1% (state-of-art)
    zz_error = abs(1 - np.abs(psi_final[1]))
    
    print(f"\nError analysis:")
    print(f"  ZZ-induced error: {zz_error:.2e}")
    print(f"  Gate error budget: {single_qubit_error_budget:.2e}")
    print(f"  ZZ contribution: {zz_error/single_qubit_error_budget * 100:.1f}%")
    
    print()


def example7_hardware_comparison():
    """Example 7: Compare ZZ crosstalk across different hardware."""
    print("=" * 70)
    print("Example 7: ZZ Crosstalk Comparison Across Hardware")
    print("=" * 70)
    
    topo = get_qubit_topology(5, topology="linear")
    
    models = ["transmon_ibm", "transmon_google", "transmon_rigetti", "ion_ytterbium"]
    
    print(f"\nZZ coupling strengths for 5-qubit linear chain:\n")
    print(f"{'Model':<20} {'ZZ Coupling':<15} {'Notes'}")
    print("-" * 70)
    
    for model in models:
        couplings = get_crosstalk_couplings(topo, qubit_model=model)
        xi = list(couplings.values())[0]  # All edges have same coupling
        
        if xi == 0:
            coupling_str = "0 MHz"
            notes = "No ZZ crosstalk"
        else:
            coupling_str = f"{xi/1e6:.1f} MHz"
            if "ibm" in model:
                notes = "Moderate ZZ, echo mitigation"
            elif "google" in model:
                notes = "Tunable couplers reduce ZZ"
            elif "rigetti" in model:
                notes = "Strong ZZ, always-on"
            else:
                notes = ""
        
        print(f"{model:<20} {coupling_str:<15} {notes}")
    
    print("\nKey insights:")
    print("  • Superconducting qubits have significant ZZ crosstalk")
    print("  • Ion traps have NO ZZ crosstalk (motional coupling instead)")
    print("  • Tunable couplers (Google) dramatically reduce ZZ")
    print("  • IBM uses echo sequences for ZZ mitigation")
    
    print()


def example8_parallel_gate_scheduling():
    """Example 8: ZZ crosstalk impact on parallel gate scheduling."""
    print("=" * 70)
    print("Example 8: ZZ Crosstalk Impact on Parallel Gates")
    print("=" * 70)
    
    topo = get_qubit_topology(4, topology="linear", zz_strength=3e6)
    
    print(f"\n4-qubit linear chain: 0--1--2--3")
    print(f"ZZ coupling: 3 MHz")
    
    # Scenario 1: Sequential gates (no crosstalk)
    print(f"\n✅ Scenario 1: Sequential X gates")
    print(f"  Time 0-100ns:  X(0)")
    print(f"  Time 100-200ns: X(1)")
    print(f"  Time 200-300ns: X(2)")
    print(f"  Total time: 300 ns")
    print(f"  ZZ crosstalk: NONE (gates never overlap)")
    
    # Scenario 2: Parallel gates on non-neighbors (OK)
    print(f"\n✅ Scenario 2: Parallel X gates on non-neighbors")
    print(f"  Time 0-100ns: X(0) || X(2)")  # Qubits 0 and 2 not connected
    print(f"  Time 100-200ns: X(1) || X(3)")
    print(f"  Total time: 200 ns")
    print(f"  ZZ crosstalk: NONE (no direct coupling)")
    
    # Scenario 3: Parallel gates on neighbors (BAD)
    print(f"\n⚠️ Scenario 3: Parallel X gates on neighbors")
    print(f"  Time 0-100ns: X(0) || X(1)")  # Qubits 0 and 1 ARE connected!
    print(f"  Total time: 100 ns")
    
    t_gate = 100e-9
    xi = 3e6
    phi_zz = xi * t_gate
    
    print(f"  ZZ crosstalk: {phi_zz:.4f} rad = {phi_zz*180/np.pi:.2f}°")
    print(f"  → Accumulated conditional phase!")
    print(f"  → Requires compensation or echo sequences")
    
    print(f"\nMitigation strategies:")
    print(f"  1. Avoid parallel gates on neighbors (scheduling)")
    print(f"  2. Echo sequences (X-delay-X cancels ZZ)")
    print(f"  3. ZZ-aware gate calibration")
    print(f"  4. Tunable couplers (Google approach)")
    
    print()


def main():
    """Run all ZZ crosstalk examples."""
    print("\n" + "=" * 70)
    print(" ZZ Crosstalk Noise Modeling Examples (P1.2)")
    print("=" * 70 + "\n")
    
    example1_basic_zz_crosstalk()
    example2_linear_chain_topology()
    example3_grid_topology()
    example4_ibm_heavy_hex()
    example5_custom_topology()
    example6_zz_crosstalk_time_evolution()
    example7_hardware_comparison()
    example8_parallel_gate_scheduling()
    
    print("=" * 70)
    print("All examples completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
