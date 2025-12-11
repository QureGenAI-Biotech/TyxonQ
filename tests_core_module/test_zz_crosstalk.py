"""Tests for ZZ crosstalk noise modeling.

Tests the implementation of ZZ crosstalk Hamiltonian and qubit topology
configurations for realistic pulse-level simulations.

Author: TyxonQ Development Team
"""

import pytest
import numpy as np
from tyxonq.libs.quantum_library.noise import zz_crosstalk_hamiltonian
from tyxonq.libs.quantum_library.pulse_physics import (
    get_qubit_topology,
    get_crosstalk_couplings,
    QubitTopology,
)


class TestZZCrosstalkHamiltonian:
    """Test ZZ crosstalk Hamiltonian construction."""
    
    def test_basic_zz_hamiltonian_2qubits(self):
        """Test basic ZZ Hamiltonian for 2 qubits."""
        xi = 3e6  # 3 MHz coupling
        H_ZZ = zz_crosstalk_hamiltonian(xi, num_qubits=2)
        
        # Check shape
        assert H_ZZ.shape == (4, 4), "H_ZZ should be 4x4 for 2 qubits"
        
        # Check Hermiticity
        assert np.allclose(H_ZZ, H_ZZ.conj().T), "H_ZZ should be Hermitian"
        
        # Check expected form: H_ZZ = xi * (Z ⊗ Z)
        Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
        H_expected = xi * np.kron(Z, Z)
        
        assert np.allclose(H_ZZ, H_expected), "H_ZZ does not match xi * Z⊗Z"
    
    def test_zz_eigenvalues(self):
        """Test that ZZ Hamiltonian has correct eigenvalues."""
        xi = 5e6  # 5 MHz
        H_ZZ = zz_crosstalk_hamiltonian(xi, num_qubits=2)
        
        # Eigenvalues of Z⊗Z are {1, -1, -1, 1} for basis {|00>, |01>, |10>, |11>}
        # So eigenvalues of xi*Z⊗Z are {xi, -xi, -xi, xi}
        eigenvalues = np.linalg.eigvalsh(H_ZZ)
        expected = np.array([-xi, -xi, xi, xi])
        
        assert np.allclose(np.sort(eigenvalues), np.sort(expected)), \
            f"ZZ eigenvalues incorrect: got {eigenvalues}, expected {expected}"
    
    def test_zz_conditional_phase(self):
        """Test ZZ Hamiltonian produces correct conditional phase."""
        xi = 1e6  # 1 MHz
        t = 100e-9  # 100 ns
        
        H_ZZ = zz_crosstalk_hamiltonian(xi, num_qubits=2)
        
        # Time evolution: U = exp(-i H_ZZ t)
        U_ZZ = scipy_expm(-1j * H_ZZ * t)
        
        # Expected conditional phase on |11> state
        expected_phase = np.exp(-1j * xi * t)
        
        # |11> is basis state [0, 0, 0, 1]
        psi_11 = np.array([0, 0, 0, 1], dtype=np.complex128)
        psi_final = U_ZZ @ psi_11
        
        # Phase should be accumulated
        assert np.allclose(psi_final, expected_phase * psi_11), \
            "Conditional phase incorrect"
    
    def test_zz_hamiltonian_3qubits(self):
        """Test ZZ Hamiltonian extended to 3 qubits."""
        xi = 2e6  # 2 MHz
        H_ZZ = zz_crosstalk_hamiltonian(xi, num_qubits=3)
        
        # Check shape (2^3 = 8)
        assert H_ZZ.shape == (8, 8), "H_ZZ should be 8x8 for 3 qubits"
        
        # Check Hermiticity
        assert np.allclose(H_ZZ, H_ZZ.conj().T), "H_ZZ should be Hermitian"
        
        # For 3 qubits, ZZ acts on qubits 0 and 1
        # H_ZZ = xi * Z⊗Z⊗I
        Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
        I = np.eye(2, dtype=np.complex128)
        H_expected = xi * np.kron(np.kron(Z, Z), I)
        
        assert np.allclose(H_ZZ, H_expected), \
            "3-qubit H_ZZ does not match xi * Z⊗Z⊗I"
    
    def test_negative_coupling(self):
        """Test that negative ZZ coupling works correctly."""
        xi_negative = -5e6  # -5 MHz (rare but possible)
        H_ZZ = zz_crosstalk_hamiltonian(xi_negative, num_qubits=2)
        
        # Should still be Hermitian
        assert np.allclose(H_ZZ, H_ZZ.conj().T), "Negative xi H_ZZ should be Hermitian"
        
        # Eigenvalues should be negative of positive case
        eigenvalues = np.linalg.eigvalsh(H_ZZ)
        expected = np.array([xi_negative, -xi_negative, -xi_negative, xi_negative])
        
        assert np.allclose(np.sort(eigenvalues), np.sort(expected)), \
            "Negative coupling eigenvalues incorrect"
    
    def test_invalid_num_qubits(self):
        """Test that num_qubits < 2 raises error."""
        with pytest.raises(ValueError, match="num_qubits must be >= 2"):
            zz_crosstalk_hamiltonian(1e6, num_qubits=1)
        
        with pytest.raises(ValueError, match="num_qubits must be >= 2"):
            zz_crosstalk_hamiltonian(1e6, num_qubits=0)


class TestQubitTopology:
    """Test qubit topology and connectivity configurations."""
    
    def test_linear_topology(self):
        """Test linear chain topology."""
        topo = get_qubit_topology(5, topology="linear", zz_strength=3e6)
        
        assert topo.num_qubits == 5
        assert topo.topology_type == "linear"
        
        # Linear chain should have 4 edges for 5 qubits
        expected_edges = [(0, 1), (1, 2), (2, 3), (3, 4)]
        assert topo.edges == expected_edges, \
            f"Linear edges incorrect: {topo.edges} != {expected_edges}"
        
        # All couplings should be 3 MHz
        for edge in expected_edges:
            assert topo.zz_couplings[edge] == 3e6, \
                f"Coupling for {edge} should be 3 MHz"
    
    def test_grid_topology(self):
        """Test 2D grid topology."""
        # 3x3 grid (9 qubits)
        topo = get_qubit_topology(9, topology="grid", grid_shape=(3, 3), zz_strength=2e6)
        
        assert topo.num_qubits == 9
        assert topo.topology_type == "grid"
        
        # 3x3 grid should have 12 edges (6 horizontal + 6 vertical)
        assert len(topo.edges) == 12, \
            f"3x3 grid should have 12 edges, got {len(topo.edges)}"
        
        # Check a few specific edges
        assert (0, 1) in topo.edges, "Horizontal edge (0,1) missing"
        assert (0, 3) in topo.edges, "Vertical edge (0,3) missing"
        assert (4, 5) in topo.edges, "Center horizontal edge missing"
    
    def test_grid_auto_shape(self):
        """Test grid auto-detection of shape."""
        # 4 qubits should auto-detect 2x2 grid
        topo = get_qubit_topology(4, topology="grid", zz_strength=1e6)
        
        # 2x2 grid has 4 edges
        assert len(topo.edges) == 4, \
            f"2x2 grid should have 4 edges, got {len(topo.edges)}"
    
    def test_heavy_hex_topology(self):
        """Test IBM Heavy-Hex topology (27 qubits)."""
        topo = get_qubit_topology(27, topology="heavy_hex", zz_strength=4e6)
        
        assert topo.num_qubits == 27
        assert topo.topology_type == "heavy_hex"
        
        # Heavy-Hex should have many edges (structure: 5-4-5-4-5-4)
        assert len(topo.edges) > 20, \
            "Heavy-Hex should have > 20 edges"
    
    def test_heavy_hex_wrong_qubits(self):
        """Test that Heavy-Hex requires exactly 27 qubits."""
        with pytest.raises(ValueError, match="Heavy-Hex topology requires 27 qubits"):
            get_qubit_topology(10, topology="heavy_hex")
    
    def test_custom_topology(self):
        """Test custom topology with user-defined edges."""
        # Triangle topology
        edges = [(0, 1), (1, 2), (0, 2)]
        custom_couplings = {
            (0, 1): 5e6,  # 5 MHz
            (1, 2): 3e6,  # 3 MHz
            (0, 2): 1e6   # 1 MHz (weak coupling)
        }
        
        topo = get_qubit_topology(
            3, topology="custom",
            edges=edges,
            custom_couplings=custom_couplings
        )
        
        assert topo.num_qubits == 3
        assert topo.topology_type == "custom"
        assert set(topo.edges) == set(edges)
        
        # Check custom couplings
        assert topo.zz_couplings[(0, 1)] == 5e6
        assert topo.zz_couplings[(1, 2)] == 3e6
        assert topo.zz_couplings[(0, 2)] == 1e6
    
    def test_custom_topology_no_edges(self):
        """Test that custom topology requires edges parameter."""
        with pytest.raises(ValueError, match="Custom topology requires 'edges'"):
            get_qubit_topology(3, topology="custom")
    
    def test_get_neighbors(self):
        """Test get_neighbors() method."""
        topo = get_qubit_topology(5, topology="linear", zz_strength=1e6)
        
        # Qubit 0: neighbor is 1
        assert topo.get_neighbors(0) == [1]
        
        # Qubit 2: neighbors are 1 and 3
        assert topo.get_neighbors(2) == [1, 3]
        
        # Qubit 4: neighbor is 3
        assert topo.get_neighbors(4) == [3]
    
    def test_get_coupling(self):
        """Test get_coupling() method."""
        topo = get_qubit_topology(3, topology="linear", zz_strength=2e6)
        
        # Connected qubits
        assert topo.get_coupling(0, 1) == 2e6
        assert topo.get_coupling(1, 2) == 2e6
        
        # Order should not matter
        assert topo.get_coupling(1, 0) == 2e6
        
        # Non-connected qubits
        assert topo.get_coupling(0, 2) == 0.0


class TestCrosstalkCouplings:
    """Test realistic crosstalk coupling configurations."""
    
    def test_ibm_transmon_couplings(self):
        """Test IBM transmon realistic couplings."""
        topo = get_qubit_topology(5, topology="linear")
        couplings = get_crosstalk_couplings(topo, qubit_model="transmon_ibm")
        
        # IBM typical: 3 MHz
        for edge in topo.edges:
            assert couplings[edge] == 3e6, \
                f"IBM coupling for {edge} should be 3 MHz"
    
    def test_google_transmon_couplings(self):
        """Test Google transmon realistic couplings (tunable couplers)."""
        topo = get_qubit_topology(4, topology="linear")
        couplings = get_crosstalk_couplings(topo, qubit_model="transmon_google")
        
        # Google typical: 0.5 MHz (tunable couplers reduce ZZ)
        for edge in topo.edges:
            assert couplings[edge] == 0.5e6, \
                f"Google coupling for {edge} should be 0.5 MHz"
    
    def test_ion_trap_no_zz(self):
        """Test ion trap has no ZZ crosstalk."""
        topo = get_qubit_topology(3, topology="linear")
        couplings = get_crosstalk_couplings(topo, qubit_model="ion_ytterbium")
        
        # Ion traps use motional coupling, no ZZ crosstalk
        for edge in topo.edges:
            assert couplings[edge] == 0.0, \
                f"Ion trap should have zero ZZ coupling for {edge}"
    
    def test_use_topology_couplings_directly(self):
        """Test that custom couplings in topology can be used directly."""
        custom = {(0, 1): 10e6, (1, 2): 5e6}
        topo = get_qubit_topology(
            3, topology="custom",
            edges=[(0, 1), (1, 2)],
            custom_couplings=custom
        )
        
        # Use topology.zz_couplings directly for custom values
        # get_crosstalk_couplings() is for model defaults only
        assert topo.zz_couplings[(0, 1)] == 10e6
        assert topo.zz_couplings[(1, 2)] == 5e6


# ============================================================================
# Helper functions
# ============================================================================

def scipy_expm(M):
    """Matrix exponential using scipy."""
    import scipy.linalg
    return scipy.linalg.expm(M)


# ============================================================================
# Integration tests
# ============================================================================

class TestZZCrosstalkIntegration:
    """Integration tests combining ZZ Hamiltonian and topology."""
    
    def test_full_system_evolution(self):
        """Test evolution of multi-qubit system with ZZ crosstalk."""
        # 3-qubit linear chain
        topo = get_qubit_topology(3, topology="linear", zz_strength=2e6)
        
        # Build full Hamiltonian with ZZ crosstalk on all edges
        # For simplicity, just test first edge (0,1)
        xi_01 = topo.get_coupling(0, 1)
        H_ZZ_01 = zz_crosstalk_hamiltonian(xi_01, num_qubits=3)
        
        # Check Hamiltonian is properly sized
        assert H_ZZ_01.shape == (8, 8), "3-qubit system should have 8x8 Hamiltonian"
        
        # Time evolution
        t = 50e-9  # 50 ns
        U = scipy_expm(-1j * H_ZZ_01 * t)
        
        # Verify unitarity
        assert np.allclose(U @ U.conj().T, np.eye(8)), \
            "Time evolution operator should be unitary"
    
    def test_realistic_ibm_crosstalk_simulation(self):
        """Test realistic IBM 5-qubit processor simulation."""
        # IBM 5-qubit "Yorktown" linear chain
        topo = get_qubit_topology(5, topology="linear")
        couplings = get_crosstalk_couplings(topo, qubit_model="transmon_ibm")
        
        # Typical values: ~3 MHz
        for edge, xi in couplings.items():
            assert 1e6 <= xi <= 10e6, \
                f"IBM coupling {xi/1e6:.1f} MHz should be in 1-10 MHz range"
    
    def test_grid_topology_crosstalk_map(self):
        """Test creating crosstalk map for 2D grid."""
        topo = get_qubit_topology(9, topology="grid", grid_shape=(3, 3), zz_strength=2.5e6)
        
        # Verify all qubits have at least 2 neighbors (except corners/edges)
        center_qubit = 4  # Center of 3x3 grid
        neighbors = topo.get_neighbors(center_qubit)
        
        # Center should have 4 neighbors
        assert len(neighbors) == 4, \
            f"Center qubit should have 4 neighbors, got {len(neighbors)}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
