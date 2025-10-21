"""
Variational Quantum Dynamics - Spin-Boson Model

This example demonstrates quantum dynamics simulation using TyxonQ's built-in
DynamicsNumericRuntime for chemistry applications.

Based on:
- src/tyxonq/applications/chem/runtimes/dynamics_numeric.py
- tests_mol_valid/test_dynamics.py

Key Features:
- Spin-Boson Model (SBM) Hamiltonian
- VQD (Variational Quantum Dynamics) algorithm
- p-VQD (Projected VQD) algorithm  
- Observable tracking (⟨Z⟩, ⟨X⟩)
- Comparison with exact Trotter evolution

Requirements:
- renormalizer package
- PyTorch backend
"""

import numpy as np
import time

import tyxonq as tq

print("=" * 70)
print("Variational Quantum Dynamics - Spin-Boson Model")
print("=" * 70)

# Check dependencies
try:
    import renormalizer
    from renormalizer import Op
    HAS_RENORMALIZER = True
except ImportError:
    print("\n⚠️  Warning: renormalizer not installed")
    print("  Install with: pip install renormalizer")
    print("\nSkipping example...")
    exit(0)

from tyxonq.applications.chem.chem_libs.hamiltonians_chem_library import sbm
from tyxonq.libs.hamiltonian_encoding.operator_encoding import (
    qubit_encode_op,
    qubit_encode_basis
)
from tyxonq.applications.chem.runtimes.dynamics_numeric import DynamicsNumericRuntime

# ==============================================================================
# Configuration
# ==============================================================================

print("\n[1/5] Configuration...")

# Spin-Boson Model parameters
EPSILON = 0.0  # Bias
DELTA = 1.0  # Tunneling
N_MODES = 1  # Number of bosonic modes
OMEGA_LIST = [1.0]  # Mode frequencies
G_LIST = [0.5]  # Coupling strengths
N_BASIS = 8  # Basis states per mode

# Dynamics parameters
N_LAYERS = 3
TIME_STEP = 0.1
N_STEPS = 50
ALGORITHM = "VQD"  # "VQD" or "p-VQD"

print(f"  Spin-Boson Model: ε={EPSILON}, Δ={DELTA}")
print(f"  Modes: {N_MODES}, ω={OMEGA_LIST}, g={G_LIST}")
print(f"  Algorithm: {ALGORITHM}, Layers: {N_LAYERS}")
print(f"  Time: {N_STEPS} steps × {TIME_STEP} = {N_STEPS * TIME_STEP}")

# ==============================================================================
# Build Hamiltonian
# ==============================================================================

print("\n[2/5] Building Hamiltonian...")

# Set backend
tq.set_backend("pytorch")

# Get Hamiltonian and basis from chemistry library
ham_terms = sbm.get_ham_terms(EPSILON, DELTA, N_MODES, OMEGA_LIST, G_LIST)
basis = sbm.get_basis(OMEGA_LIST, [N_BASIS] * N_MODES)

# Encode to qubits
ham_terms_spin, _ = qubit_encode_op(ham_terms, basis, "gray")
basis_spin = qubit_encode_basis(basis, "gray")

n_qubits = len(basis_spin)
print(f"  ✓ Hamiltonian terms: {len(ham_terms)}")
print(f"  ✓ Encoded to {n_qubits} qubits")

# ==============================================================================
# Initialize Dynamics Runtime
# ==============================================================================

print("\n[3/5] Initializing Dynamics Runtime...")

dynamics = DynamicsNumericRuntime(
    ham_terms_spin,
    basis_spin,
    n_layers=N_LAYERS,
    eps=1e-5,
    include_phase=False,  # McLachlan's principle
)

print(f"  ✓ Parameters: {dynamics.n_params}")
print(f"  ✓ Initial energy: {np.real(dynamics.state.conj() @ dynamics.h @ dynamics.state):.6f}")

# Add observables
dynamics.add_property_op("Z", Op("Z", "spin"))
dynamics.add_property_op("X", Op("X", "spin"))
print(f"  ✓ Observables added: Z, X")

# ==============================================================================
# Run Time Evolution
# ==============================================================================

print(f"\n[4/5] Running {ALGORITHM} evolution...")
print("-" * 70)

t_list = []
z_list = []
x_list = []
energy_list = []

time_start = time.time()

for step in range(N_STEPS):
    # Current time and observables
    t = dynamics.t
    props = dynamics.properties()
    z_val = float(np.asarray(props["Z"]).real)
    x_val = float(np.asarray(props["X"]).real)
    energy = float(np.real(dynamics.state.conj() @ dynamics.h @ dynamics.state))
    
    # Store
    t_list.append(t)
    z_list.append(z_val)
    x_list.append(x_val)
    energy_list.append(energy)
    
    # Print progress
    if step % 10 == 0:
        print(f"  Step {step:3d} | t={t:.3f} | "
              f"E={energy:.6f} | ⟨Z⟩={z_val:.4f} | ⟨X⟩={x_val:.4f}")
    
    # Time step
    if step < N_STEPS - 1:
        if ALGORITHM == "VQD":
            dynamics.step_vqd(TIME_STEP)
        elif ALGORITHM == "p-VQD":
            dynamics.step_pvqd(TIME_STEP)

time_end = time.time()

print("-" * 70)
print(f"  Total time: {time_end - time_start:.2f}s")
print(f"  Time per step: {(time_end - time_start) / N_STEPS:.3f}s")

# ==============================================================================
# Exact Evolution Comparison
# ==============================================================================

print("\n[5/6] Comparing with exact Trotter evolution...")

# Run exact evolution using Trotterization for comparison
try:
    from scipy.linalg import expm
    
    # Initial state
    psi_exact = dynamics.state_list[0].copy()
    
    # Evolve using matrix exponential (exact evolution)
    exact_z_list = []
    exact_x_list = []
    exact_energy_list = []
    
    # Hamiltonian for evolution
    H_dense = dynamics.h
    
    # Observable matrices
    from renormalizer.model import Model, Op
    from renormalizer import Mpo
    
    Z_mpo = Mpo(Model(basis_spin, [Op("Z", "spin")]))
    X_mpo = Mpo(Model(basis_spin, [Op("X", "spin")]))
    Z_mat = Z_mpo.todense()
    X_mat = X_mpo.todense()
    
    for step in range(N_STEPS):
        # Compute observables
        z_exact = float(np.real(psi_exact.conj() @ Z_mat @ psi_exact))
        x_exact = float(np.real(psi_exact.conj() @ X_mat @ psi_exact))
        e_exact = float(np.real(psi_exact.conj() @ H_dense @ psi_exact))
        
        exact_z_list.append(z_exact)
        exact_x_list.append(x_exact)
        exact_energy_list.append(e_exact)
        
        # Evolve
        if step < N_STEPS - 1:
            U = expm(-1j * H_dense * TIME_STEP)
            psi_exact = U @ psi_exact
            # Normalize
            psi_exact = psi_exact / np.linalg.norm(psi_exact)
    
    # Final fidelity
    final_vqd_state = dynamics.state
    fidelity = np.abs(np.vdot(psi_exact, final_vqd_state))**2
    
    print(f"  ✓ Exact evolution completed")
    print(f"  ✓ Final fidelity: {fidelity:.6f}")
    
    HAS_EXACT = True
    
except Exception as e:
    print(f"  ⚠️  Exact evolution not available: {e}")
    HAS_EXACT = False
    exact_z_list = None
    exact_x_list = None
    exact_energy_list = None
    fidelity = None

# ==============================================================================
# Analysis
# ==============================================================================

print("\n[6/6] Analysis...")

final_props = dynamics.properties()
final_z = float(np.asarray(final_props["Z"]).real)
final_x = float(np.asarray(final_props["X"]).real)
final_energy = energy_list[-1]

print(f"\n  Final time: {t_list[-1]:.3f}")
print(f"  Final energy: {final_energy:.6f}")
print(f"  Final ⟨Z⟩: {final_z:.6f}")
print(f"  Final ⟨X⟩: {final_x:.6f}")

# Energy conservation
energy_drift = abs(energy_list[-1] - energy_list[0])
print(f"\n  Energy conservation:")
print(f"    Energy drift: {energy_drift:.6e}")
if energy_drift < 1e-3:
    print("    ✓ Energy well conserved")
else:
    print("    ⚠️  Increase layers or decrease time step for better conservation")

# Comparison with exact (if available)
if HAS_EXACT:
    energy_error = abs(energy_list[-1] - exact_energy_list[-1])
    z_error = abs(z_list[-1] - exact_z_list[-1])
    x_error = abs(x_list[-1] - exact_x_list[-1])
    
    print(f"\n  Comparison with exact evolution:")
    print(f"    VQD energy:   {energy_list[-1]:.8f}")
    print(f"    Exact energy: {exact_energy_list[-1]:.8f}")
    print(f"    Energy error: {energy_error:.6e}")
    print(f"    ⟨Z⟩ error:    {z_error:.6e}")
    print(f"    ⟨X⟩ error:    {x_error:.6e}")
    print(f"    Fidelity:     {fidelity:.6f}")
    
    if fidelity > 0.99:
        print("    ✅ Excellent agreement with exact evolution!")
    elif fidelity > 0.95:
        print("    ✓ Good agreement with exact evolution")
    else:
        print("    ⚠️  Consider increasing layers for better accuracy")

# Optional: Plot
try:
    import matplotlib.pyplot as plt
    
    if HAS_EXACT:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Z observable
        ax = axes[0, 0]
        ax.plot(t_list, z_list, 'b-', label='VQD', linewidth=2)
        ax.plot(t_list, exact_z_list, 'r--', label='Exact', linewidth=2, alpha=0.7)
        ax.set_xlabel('Time')
        ax.set_ylabel('⟨Z⟩')
        ax.set_title('Z Observable')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # X observable
        ax = axes[0, 1]
        ax.plot(t_list, x_list, 'b-', label='VQD', linewidth=2)
        ax.plot(t_list, exact_x_list, 'r--', label='Exact', linewidth=2, alpha=0.7)
        ax.set_xlabel('Time')
        ax.set_ylabel('⟨X⟩')
        ax.set_title('X Observable')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Energy
        ax = axes[1, 0]
        ax.plot(t_list, energy_list, 'b-', label='VQD', linewidth=2)
        ax.plot(t_list, exact_energy_list, 'r--', label='Exact', linewidth=2, alpha=0.7)
        ax.set_xlabel('Time')
        ax.set_ylabel('Energy')
        ax.set_title('Energy Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Fidelity over time (recompute for each step)
        ax = axes[1, 1]
        fidelities = []
        for i in range(len(t_list)):
            psi_e = dynamics.state_list[0].copy()
            U_total = expm(-1j * dynamics.h * t_list[i])
            psi_e = U_total @ psi_e
            psi_e = psi_e / np.linalg.norm(psi_e)
            psi_v = dynamics.state_list[i]
            f = np.abs(np.vdot(psi_e, psi_v))**2
            fidelities.append(f)
        
        ax.plot(t_list, fidelities, 'g-', linewidth=2)
        ax.axhline(y=0.99, color='r', linestyle='--', alpha=0.5, label='99% threshold')
        ax.set_xlabel('Time')
        ax.set_ylabel('Fidelity')
        ax.set_title('Fidelity with Exact State')
        ax.set_ylim([max(0.9, min(fidelities) - 0.05), 1.01])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'Spin-Boson Model: {ALGORITHM} vs Exact Evolution', fontsize=14, fontweight='bold')
    else:
        # Fallback to simple plots without exact comparison
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        ax1.plot(t_list, z_list, 'b-', label='⟨Z⟩', linewidth=2)
        ax1.plot(t_list, x_list, 'r-', label='⟨X⟩', linewidth=2)
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Observable')
        ax1.set_title(f'Spin-Boson Dynamics ({ALGORITHM})')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(t_list, energy_list, 'k-', linewidth=2)
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Energy')
        ax2.set_title('Energy Evolution')
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('vqd_sbm_results.png', dpi=150)
    print(f"\n  📊 Plot saved: vqd_sbm_results.png")
except ImportError:
    print("\n  ℹ️  matplotlib not available, skipping plots")

print("\n" + "=" * 70)
print("Variational Quantum Dynamics Complete!")
print("=" * 70)

print("\n📚 Key Concepts:")
print("  - VQD: Variational Quantum Dynamics")
print("  - McLachlan's variational principle: minimizes ||dψ/dt + iHψ||")
print("  - p-VQD: Projected VQD for better long-time stability")
print("  - SBM: Two-level system coupled to bosonic bath")

print("\n🔬 Implementation:")
print("  - DynamicsNumericRuntime: src/tyxonq/applications/chem/runtimes/")
print("  - Tests: tests_mol_valid/test_dynamics.py")
print("  - SBM library: applications/chem/chem_libs/hamiltonians_chem_library/")

print("\n💡 Tips:")
print("  - Increase N_LAYERS for higher accuracy")
print("  - Use p-VQD for long-time evolution")
print("  - Monitor energy conservation")
print("  - Try different TIME_STEP values")

print("\n" + "=" * 70)
