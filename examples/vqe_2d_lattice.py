"""VQE on 2D Square Lattice Heisenberg Model.

This example demonstrates:
1. 2D lattice topology for quantum simulation
2. Heisenberg model on square lattice
3. Hardware-efficient ansatz with SWAP-based entangling layers
4. PyTorch optimization with learning rate scheduling

The 2D Heisenberg Hamiltonian is:
    H = Σ_{⟨i,j⟩} (Jx·XᵢXⱼ + Jy·YᵢYⱼ + Jz·ZᵢZⱼ)

where the sum is over nearest-neighbor pairs on a square lattice.

Author: TyxonQ Team
Date: 2025
"""

import numpy as np
import tyxonq as tq


class Grid2D:
    """2D square lattice coordinate system.
    
    Represents an n×m grid with periodic boundary conditions (optional).
    Provides utilities for nearest-neighbor connections and qubit indexing.
    """
    
    def __init__(self, n_rows, n_cols, periodic=False):
        """Initialize 2D grid.
        
        Args:
            n_rows: Number of rows
            n_cols: Number of columns
            periodic: If True, use periodic boundary conditions
        """
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.n_qubits = n_rows * n_cols
        self.periodic = periodic
    
    def coord_to_index(self, row, col):
        """Convert (row, col) coordinate to linear qubit index."""
        return row * self.n_cols + col
    
    def index_to_coord(self, idx):
        """Convert linear qubit index to (row, col) coordinate."""
        return (idx // self.n_cols, idx % self.n_cols)
    
    def nearest_neighbors(self):
        """Get all nearest-neighbor pairs on the lattice.
        
        Returns:
            List of (i, j) tuples representing adjacent qubits
        """
        neighbors = []
        for row in range(self.n_rows):
            for col in range(self.n_cols):
                idx = self.coord_to_index(row, col)
                
                # Right neighbor
                if col + 1 < self.n_cols:
                    neighbor = self.coord_to_index(row, col + 1)
                    neighbors.append((idx, neighbor))
                elif self.periodic and col + 1 == self.n_cols:
                    neighbor = self.coord_to_index(row, 0)
                    neighbors.append((idx, neighbor))
                
                # Down neighbor
                if row + 1 < self.n_rows:
                    neighbor = self.coord_to_index(row + 1, col)
                    neighbors.append((idx, neighbor))
                elif self.periodic and row + 1 == self.n_rows:
                    neighbor = self.coord_to_index(0, col)
                    neighbors.append((idx, neighbor))
        
        return neighbors


def build_2d_heisenberg_hamiltonian(grid, Jx=1.0, Jy=1.0, Jz=1.0):
    """Build 2D Heisenberg Hamiltonian on square lattice.
    
    H = Σ_{⟨i,j⟩} (Jx·XᵢXⱼ + Jy·YᵢYⱼ + Jz·ZᵢZⱼ)
    
    Args:
        grid: Grid2D object defining lattice topology
        Jx, Jy, Jz: Coupling strengths
        
    Returns:
        Hamiltonian matrix (2^n × 2^n)
    """
    from tyxonq.numerics.api import get_backend
    K = get_backend(None)
    
    n_qubits = grid.n_qubits
    dim = 2 ** n_qubits
    H = K.zeros((dim, dim), dtype=K.complex128)
    
    # Pauli matrices
    X = K.array([[0, 1], [1, 0]], dtype=K.complex128)
    Y = K.array([[0, -1j], [1j, 0]], dtype=K.complex128)
    Z = K.array([[1, 0], [0, -1]], dtype=K.complex128)
    I = K.eye(2, dtype=K.complex128)
    
    # Add nearest-neighbor interactions
    for i, j in grid.nearest_neighbors():
        # Build operators for bond (i, j)
        ops_x = []
        ops_y = []
        ops_z = []
        
        for q in range(n_qubits):
            if q == i or q == j:
                ops_x.append(X)
                ops_y.append(Y)
                ops_z.append(Z)
            else:
                ops_x.append(I)
                ops_y.append(I)
                ops_z.append(I)
        
        # Build full operators via Kronecker product
        XX_term = ops_x[0]
        YY_term = ops_y[0]
        ZZ_term = ops_z[0]
        for q in range(1, n_qubits):
            XX_term = K.kron(XX_term, ops_x[q])
            YY_term = K.kron(YY_term, ops_y[q])
            ZZ_term = K.kron(ZZ_term, ops_z[q])
        
        H = H + Jx * XX_term + Jy * YY_term + Jz * ZZ_term
    
    return H


def build_singlet_initial_state(c):
    """Initialize circuit with singlet pairs.
    
    Creates entangled pairs in singlet state |Ψ⁻⟩ = (|01⟩ - |10⟩)/√2
    on adjacent qubits.
    """
    n_qubits = c.num_qubits
    for i in range(0, n_qubits - 1, 2):
        j = i + 1
        c.x(i)
        c.h(i)
        c.cx(i, j)
        c.x(j)
    return c


def build_2d_entangling_layer(c, grid, params, layer_idx):
    """Build hardware-efficient entangling layer on 2D lattice.
    
    Uses SWAP gates for entanglement and parameterized rotations.
    
    Args:
        c: Circuit object
        grid: Grid2D lattice topology
        params: Parameters [2 * n_qubits] for this layer
        layer_idx: Layer index (for alternating patterns)
    """
    n_qubits = grid.n_qubits
    
    # Entangling layer: SWAP on nearest neighbors
    neighbors = grid.nearest_neighbors()
    for i, j in neighbors:
        # SWAP = (I + XX + YY + ZZ) / 2
        # For simplicity, use CX-CX-CX decomposition
        c.cx(i, j)
        c.cx(j, i)
        c.cx(i, j)
    
    # Parameterized rotation layer
    for q in range(n_qubits):
        c.rz(q, params[q])
        c.rx(q, params[n_qubits + q])
    
    return c


def compute_2d_vqe_energy(params, n_rows, n_cols, n_layers, hamiltonian):
    """Compute VQE energy expectation value on 2D lattice.
    
    Args:
        params: Parameters [n_layers, 2 * n_qubits]
        n_rows, n_cols: Lattice dimensions
        n_layers: Circuit depth
        hamiltonian: Hamiltonian matrix
        
    Returns:
        Energy expectation ⟨ψ|H|ψ⟩
    """
    from tyxonq.numerics.api import get_backend
    K = get_backend(None)
    
    grid = Grid2D(n_rows, n_cols, periodic=False)
    n_qubits = grid.n_qubits
    
    # Build variational circuit
    c = tq.Circuit(n_qubits)
    c = build_singlet_initial_state(c)
    
    for layer in range(n_layers):
        c = build_2d_entangling_layer(c, grid, params[layer], layer)
    
    # Get final state
    psi = c.wavefunction()
    psi = psi.reshape((-1, 1))
    
    # Compute ⟨ψ|H|ψ⟩
    from tyxonq.numerics import NumericBackend as NB
    expval = (NB.adjoint(psi) @ (hamiltonian @ psi))[0, 0]
    
    return K.real(expval)


def demonstrate_2d_lattice_vqe():
    """Demonstrate VQE on 2D square lattice."""
    print("\n" + "="*70)
    print("VQE ON 2D SQUARE LATTICE HEISENBERG MODEL")
    print("="*70)
    
    try:
        import torch
        tq.set_backend('pytorch')
        from tyxonq.numerics.api import get_backend
        K = get_backend(None)
        
        # System parameters
        n_rows, n_cols = 2, 2
        n_qubits = n_rows * n_cols
        n_layers = 2
        
        print(f"\nLattice Configuration:")
        print(f"  Dimensions: {n_rows}×{n_cols} = {n_qubits} qubits")
        print(f"  Topology: Square lattice (open boundary)")
        print(f"  Layers: {n_layers}")
        print(f"  Total parameters: {n_layers * 2 * n_qubits}")
        
        # Build lattice and Hamiltonian
        grid = Grid2D(n_rows, n_cols, periodic=False)
        print(f"\nNearest-neighbor bonds:")
        neighbors = grid.nearest_neighbors()
        print(f"  Total bonds: {len(neighbors)}")
        for i, (q1, q2) in enumerate(neighbors):
            row1, col1 = grid.index_to_coord(q1)
            row2, col2 = grid.index_to_coord(q2)
            print(f"  Bond {i+1}: ({row1},{col1})-({row2},{col2}) [q{q1}-q{q2}]")
        
        print(f"\nBuilding Heisenberg Hamiltonian...")
        H = build_2d_heisenberg_hamiltonian(grid, Jx=1.0, Jy=1.0, Jz=1.0)
        print(f"  Hamiltonian shape: {H.shape}")
        
        # Initialize parameters
        params = torch.nn.Parameter(torch.randn(n_layers, 2 * n_qubits, dtype=torch.float32) * 0.1)
        
        # Optimizer with learning rate scheduling
        optimizer = torch.optim.Adam([params], lr=0.01)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        
        print(f"\nStarting optimization...")
        print(f"  Optimizer: Adam (lr=0.01)")
        print(f"  Scheduler: Exponential decay (γ=0.9)")
        print(f"\n{'Iter':<6} {'Energy':<12} {'Improvement':<12}")
        print("-" * 40)
        
        prev_energy = None
        for iteration in range(30):
            # Forward pass
            optimizer.zero_grad()
            energy = compute_2d_vqe_energy(params, n_rows, n_cols, n_layers, H)
            
            # Backward pass
            energy.backward()
            
            # Update
            optimizer.step()
            
            # Learning rate decay
            if (iteration + 1) % 10 == 0:
                scheduler.step()
            
            # Report
            if iteration % 5 == 0:
                e_val = float(energy.detach())
                improvement = f"{prev_energy - e_val:+.6f}" if prev_energy is not None else "N/A"
                print(f"{iteration:<6} {e_val:<12.6f} {improvement:<12}")
                prev_energy = e_val
        
        print("\n" + "-" * 40)
        print(f"Final energy: {float(energy):.6f}")
        
        print("\n✓ 2D lattice VQE demonstration complete!")
        
        # Reset backend
        tq.set_backend('numpy')
        
    except ImportError:
        print("\n⚠️  PyTorch not available - skipping demonstration")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Run all demonstrations."""
    print("\n" + "="*70)
    print("2D LATTICE QUANTUM VARIATIONAL EIGENSOLVER")
    print("="*70)
    print("\nThis example demonstrates:")
    print("  • 2D square lattice topology")
    print("  • Heisenberg model with nearest-neighbor interactions")
    print("  • Hardware-efficient ansatz with SWAP gates")
    print("  • Singlet pair initialization")
    print("  • PyTorch optimization with learning rate scheduling")
    
    demonstrate_2d_lattice_vqe()
    
    print("\n" + "="*70)
    print("KEY CONCEPTS")
    print("="*70)
    print("\n1. 2D Lattice Topology:")
    print("   - Square grid with n_rows × n_cols qubits")
    print("   - Nearest-neighbor connectivity")
    print("   - Open or periodic boundary conditions")
    print("\n2. Heisenberg Model:")
    print("   - H = Σ (Jx·XX + Jy·YY + Jz·ZZ) over bonds")
    print("   - Captures quantum magnetism")
    print("   - Ground state reveals spin ordering")
    print("\n3. Hardware-Efficient Ansatz:")
    print("   - SWAP gates for entanglement")
    print("   - Parameterized single-qubit rotations")
    print("   - Layer-by-layer construction")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
