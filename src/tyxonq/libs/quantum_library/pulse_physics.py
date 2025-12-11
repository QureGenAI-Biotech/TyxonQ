"""
Physical parameters and calibration data for pulse-level quantum simulation.

This module provides realistic hardware parameters for different qubit
technologies, enabling physics-based pulse simulations. Parameters are
collected from experimental literature and hardware specifications.

References
----------
- IBM Quantum: https://quantum.ibm.com/
- Google Quantum AI: Nature 574, 505–510 (2019)
- Rigetti Computing: arXiv:2001.08343
- QuTiP-qip: Quantum 6, 630 (2022)
- Koch et al., Phys. Rev. A 76, 042319 (2007) - Transmon theory

Qubit Technologies Supported
----------------------------
- Transmon (superconducting, charge-insensitive)
- Flux qubit (superconducting, tunable)
- Ion trap (trapped ions, high fidelity)
- Spin qubit (semiconductor, scalable)
"""

from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass


@dataclass
class QubitParameters:
    """
    Physical parameters for a qubit technology.
    
    Attributes
    ----------
    frequency : float
        Qubit transition frequency (Hz), e.g., 5e9 for 5 GHz
    anharmonicity : float
        Anharmonicity (Hz), negative for transmons (~-300 MHz)
    T1 : float
        Amplitude damping time (seconds), e.g., 50e-6 for 50 μs
    T2 : float
        Dephasing time (seconds), typically T2 ≤ 2*T1
    drive_strength : float
        Maximum Rabi frequency (Hz), e.g., 20e6 for 20 MHz
    readout_fidelity : float
        Single-shot measurement fidelity (0-1)
    gate_time : float
        Typical single-qubit gate duration (seconds)
    two_qubit_gate_time : float
        Typical two-qubit gate duration (seconds)
    """
    frequency: float
    anharmonicity: float
    T1: float
    T2: float
    drive_strength: float
    readout_fidelity: float
    gate_time: float
    two_qubit_gate_time: float
    
    def __post_init__(self):
        """Validate physical constraints."""
        if self.T2 > 2 * self.T1:
            import warnings
            warnings.warn(
                f"T2 ({self.T2:.2e}s) > 2*T1 ({2*self.T1:.2e}s) violates "
                "physical constraint. Setting T2 = 2*T1."
            )
            self.T2 = 2 * self.T1


# ============================================================================
# Superconducting Qubit Parameters
# ============================================================================

TRANSMON_IBM = QubitParameters(
    frequency=5.0e9,           # 5 GHz (typical IBM Quantum)
    anharmonicity=-330e6,      # -330 MHz (IBM Eagle processor)
    T1=100e-6,                 # 100 μs (state-of-art IBM, 2023)
    T2=150e-6,                 # 150 μs (limited by T1)
    drive_strength=20e6,       # 20 MHz Rabi frequency
    readout_fidelity=0.99,     # 99% readout fidelity
    gate_time=35e-9,           # 35 ns for single-qubit gates
    two_qubit_gate_time=300e-9 # 300 ns for CZ gate
)

TRANSMON_GOOGLE = QubitParameters(
    frequency=6.0e9,           # 6 GHz (Google Sycamore)
    anharmonicity=-210e6,      # -210 MHz
    T1=20e-6,                  # 20 μs (Sycamore, Nature 574, 505, 2019)
    T2=30e-6,                  # 30 μs
    drive_strength=25e6,       # 25 MHz
    readout_fidelity=0.997,    # 99.7% (Sycamore)
    gate_time=20e-9,           # 20 ns (fast gates)
    two_qubit_gate_time=12e-9  # 12 ns (iSWAP gate)
)

TRANSMON_RIGETTI = QubitParameters(
    frequency=4.5e9,           # 4.5 GHz
    anharmonicity=-250e6,      # -250 MHz
    T1=30e-6,                  # 30 μs
    T2=25e-6,                  # 25 μs
    drive_strength=15e6,       # 15 MHz
    readout_fidelity=0.95,     # 95%
    gate_time=50e-9,           # 50 ns
    two_qubit_gate_time=200e-9 # 200 ns (CZ gate)
)

FLUX_QUBIT = QubitParameters(
    frequency=8.0e9,           # 8 GHz (flux sweet spot)
    anharmonicity=-500e6,      # -500 MHz (larger than transmon)
    T1=5e-6,                   # 5 μs (flux noise limited)
    T2=3e-6,                   # 3 μs
    drive_strength=50e6,       # 50 MHz (strong coupling)
    readout_fidelity=0.90,     # 90%
    gate_time=20e-9,           # 20 ns
    two_qubit_gate_time=50e-9  # 50 ns
)


# ============================================================================
# Ion Trap Parameters
# ============================================================================

ION_TRAP_YTTERBIUM = QubitParameters(
    frequency=12.6e9,          # 12.6 GHz (171Yb+ hyperfine)
    anharmonicity=0.0,         # No anharmonicity (true 2-level)
    T1=1000.0,                 # Essentially infinite (hours)
    T2=0.5,                    # 0.5 s (magnetic field noise)
    drive_strength=100e3,      # 100 kHz (Rabi frequency)
    readout_fidelity=0.9995,   # 99.95% (state-of-art)
    gate_time=10e-6,           # 10 μs (single-qubit)
    two_qubit_gate_time=100e-6 # 100 μs (Mølmer-Sørensen gate)
)

ION_TRAP_CALCIUM = QubitParameters(
    frequency=3.2e9,           # 3.2 GHz (40Ca+ S-D transition)
    anharmonicity=0.0,         # True 2-level
    T1=1000.0,                 # Essentially infinite
    T2=1.0,                    # 1 s
    drive_strength=50e3,       # 50 kHz
    readout_fidelity=0.999,    # 99.9%
    gate_time=5e-6,            # 5 μs
    two_qubit_gate_time=50e-6  # 50 μs
)


# ============================================================================
# Semiconductor Spin Qubit Parameters
# ============================================================================

SPIN_QUBIT_SI = QubitParameters(
    frequency=20e9,            # 20 GHz (electron spin in Si/SiGe)
    anharmonicity=0.0,         # True 2-level
    T1=1e-3,                   # 1 ms (isotopically purified Si-28)
    T2=0.1e-3,                 # 100 μs (charge noise limited)
    drive_strength=10e6,       # 10 MHz (EDSR)
    readout_fidelity=0.95,     # 95% (spin-to-charge conversion)
    gate_time=100e-9,          # 100 ns
    two_qubit_gate_time=200e-9 # 200 ns (exchange coupling)
)

SPIN_QUBIT_GAAS = QubitParameters(
    frequency=15e9,            # 15 GHz (GaAs quantum dot)
    anharmonicity=0.0,         # True 2-level
    T1=100e-6,                 # 100 μs (nuclear spin bath)
    T2=2e-6,                   # 2 μs (strongly limited)
    drive_strength=5e6,        # 5 MHz
    readout_fidelity=0.85,     # 85%
    gate_time=200e-9,          # 200 ns
    two_qubit_gate_time=500e-9 # 500 ns
)


# ============================================================================
# Qubit Model Registry
# ============================================================================

QUBIT_MODELS: Dict[str, QubitParameters] = {
    # Superconducting qubits
    "transmon_ibm": TRANSMON_IBM,
    "transmon_google": TRANSMON_GOOGLE,
    "transmon_rigetti": TRANSMON_RIGETTI,
    "flux_qubit": FLUX_QUBIT,
    
    # Ion traps
    "ion_ytterbium": ION_TRAP_YTTERBIUM,
    "ion_calcium": ION_TRAP_CALCIUM,
    
    # Spin qubits
    "spin_si": SPIN_QUBIT_SI,
    "spin_gaas": SPIN_QUBIT_GAAS,
    
    # Aliases for convenience
    "transmon": TRANSMON_IBM,  # Default transmon
    "ion": ION_TRAP_YTTERBIUM,  # Default ion trap
    "spin": SPIN_QUBIT_SI,      # Default spin qubit
}


def get_qubit_params(model: str = "transmon") -> QubitParameters:
    """
    Retrieve physical parameters for a qubit technology.
    
    Parameters
    ----------
    model : str, optional
        Qubit model name (default "transmon")
        
        Superconducting:
        - "transmon", "transmon_ibm", "transmon_google", "transmon_rigetti"
        - "flux_qubit"
        
        Ion traps:
        - "ion", "ion_ytterbium", "ion_calcium"
        
        Spin qubits:
        - "spin", "spin_si", "spin_gaas"
        
    Returns
    -------
    QubitParameters
        Physical parameters object
        
    Examples
    --------
    >>> params = get_qubit_params("transmon_ibm")
    >>> params.T1
    1e-04  # 100 μs
    
    >>> params = get_qubit_params("ion")
    >>> params.gate_time
    1e-05  # 10 μs
    """
    model_lower = model.lower()
    if model_lower not in QUBIT_MODELS:
        available = ", ".join(QUBIT_MODELS.keys())
        raise ValueError(
            f"Unknown qubit model: {model}\n"
            f"Available models: {available}"
        )
    
    return QUBIT_MODELS[model_lower]


# ============================================================================
# Waveform Physical Constraints
# ============================================================================

@dataclass
class WaveformConstraints:
    """Physical constraints for pulse waveforms."""
    
    max_amplitude: float         # Maximum drive amplitude (dimensionless)
    max_duration: int            # Maximum pulse duration (samples)
    min_duration: int            # Minimum pulse duration (samples)
    sampling_rate: float         # ADC/DAC sampling rate (Hz)
    bandwidth_limit: float       # Low-pass filter cutoff (Hz)
    amplitude_resolution: float  # DAC resolution (volts or dimensionless)
    
    def validate_waveform(self, waveform: Any) -> bool:
        """Check if waveform satisfies constraints."""
        amp = abs(getattr(waveform, 'amp', 0))
        duration = getattr(waveform, 'duration', 0)
        
        if amp > self.max_amplitude:
            raise ValueError(
                f"Waveform amplitude {amp} exceeds maximum {self.max_amplitude}"
            )
        
        if duration > self.max_duration:
            raise ValueError(
                f"Waveform duration {duration} samples exceeds maximum {self.max_duration}"
            )
        
        if duration < self.min_duration:
            raise ValueError(
                f"Waveform duration {duration} samples below minimum {self.min_duration}"
            )
        
        return True


SUPERCONDUCTING_CONSTRAINTS = WaveformConstraints(
    max_amplitude=2.0,           # Typical drive strength limit
    max_duration=10000,          # 10k samples = 5 μs at 2 GHz
    min_duration=10,             # 10 samples = 5 ns
    sampling_rate=2.0e9,         # 2 GHz (standard AWG)
    bandwidth_limit=500e6,       # 500 MHz (mixer/amplifier limit)
    amplitude_resolution=0.001   # 10-bit DAC → ~1mV resolution
)

ION_TRAP_CONSTRAINTS = WaveformConstraints(
    max_amplitude=1.0,           # Laser power limit
    max_duration=1000000,        # 1M samples = 1 ms at 1 GHz
    min_duration=100,            # 100 samples = 100 ns
    sampling_rate=1.0e9,         # 1 GHz (AOD control)
    bandwidth_limit=100e6,       # 100 MHz
    amplitude_resolution=0.01    # 8-bit DAC
)

WAVEFORM_CONSTRAINTS: Dict[str, WaveformConstraints] = {
    "superconducting": SUPERCONDUCTING_CONSTRAINTS,
    "ion_trap": ION_TRAP_CONSTRAINTS,
    "default": SUPERCONDUCTING_CONSTRAINTS,
}


def get_waveform_constraints(platform: str = "superconducting") -> WaveformConstraints:
    """Get waveform constraints for a platform."""
    if platform not in WAVEFORM_CONSTRAINTS:
        raise ValueError(
            f"Unknown platform: {platform}. "
            f"Available: {list(WAVEFORM_CONSTRAINTS.keys())}"
        )
    return WAVEFORM_CONSTRAINTS[platform]


__all__ = [
    "QubitParameters",
    "WaveformConstraints",
    "QUBIT_MODELS",
    "get_qubit_params",
    "get_waveform_constraints",
    "QubitTopology",
    "get_qubit_topology",
    "get_crosstalk_couplings",
    
    # Specific models (for direct import)
    "TRANSMON_IBM",
    "TRANSMON_GOOGLE",
    "ION_TRAP_YTTERBIUM",
    "SPIN_QUBIT_SI",
]


# ============================================================================
# Qubit Topology and ZZ Crosstalk Configuration
# ============================================================================

@dataclass
class QubitTopology:
    """
    Qubit connectivity topology and ZZ crosstalk configuration.
    
    Defines which qubits are neighbors (can interact) and the strength
    of their ZZ coupling. Used for modeling always-on crosstalk noise.
    
    Attributes
    ----------
    num_qubits : int
        Total number of qubits in the system
    edges : List[Tuple[int, int]]
        List of connected qubit pairs (i, j) where i < j
        Example: [(0,1), (1,2), (2,3)] for linear chain
    zz_couplings : Dict[Tuple[int, int], float]
        ZZ coupling strength (Hz) for each edge
        Key: (qubit_i, qubit_j) with i < j
        Value: ξ_ij in Hz (typically 0.1-10 MHz)
    topology_type : str
        Topology name: 'linear', 'grid', 'heavy_hex', 'custom'
    """
    num_qubits: int
    edges: List[Tuple[int, int]]
    zz_couplings: Dict[Tuple[int, int], float]
    topology_type: str = "custom"
    
    def get_neighbors(self, qubit: int) -> List[int]:
        """Get all neighboring qubits of a given qubit."""
        neighbors = []
        for i, j in self.edges:
            if i == qubit:
                neighbors.append(j)
            elif j == qubit:
                neighbors.append(i)
        return sorted(neighbors)
    
    def get_coupling(self, qubit_i: int, qubit_j: int) -> float:
        """Get ZZ coupling strength between two qubits (returns 0 if not connected)."""
        i, j = min(qubit_i, qubit_j), max(qubit_i, qubit_j)
        return self.zz_couplings.get((i, j), 0.0)


def get_qubit_topology(
    num_qubits: int,
    topology: str = "linear",
    zz_strength: float = 1e6,  # 1 MHz default
    **kwargs
) -> QubitTopology:
    """
    Create qubit topology with ZZ crosstalk configuration.
    
    Parameters
    ----------
    num_qubits : int
        Number of qubits in the system
    topology : str
        Topology type: 'linear', 'grid', 'heavy_hex', or 'custom'
    zz_strength : float
        Default ZZ coupling strength (Hz) for all edges
        Typical values:
        - IBM: 1-5 MHz (arXiv:2108.12323)
        - Google: 0.1-1 MHz (with tunable couplers)
        - Rigetti: 2-10 MHz (always-on coupling)
    **kwargs : dict
        Additional topology-specific parameters:
        - For 'grid': grid_shape=(rows, cols)
        - For 'custom': edges=[(i,j), ...], custom_couplings={(i,j): xi, ...}
    
    Returns
    -------
    QubitTopology
        Configured topology with ZZ couplings
    
    Examples
    --------
    >>> # Linear chain (default)
    >>> topo = get_qubit_topology(5, topology="linear", zz_strength=3e6)
    >>> topo.edges  # [(0,1), (1,2), (2,3), (3,4)]
    
    >>> # 2D grid
    >>> topo = get_qubit_topology(9, topology="grid", grid_shape=(3,3))
    
    >>> # IBM Heavy-Hex
    >>> topo = get_qubit_topology(27, topology="heavy_hex")
    
    >>> # Custom topology
    >>> edges = [(0,1), (1,2), (0,2)]  # Triangle
    >>> custom_xi = {(0,1): 5e6, (1,2): 3e6, (0,2): 1e6}
    >>> topo = get_qubit_topology(
    ...     3, topology="custom",
    ...     edges=edges,
    ...     custom_couplings=custom_xi
    ... )
    """
    if topology == "linear":
        # Linear chain: 0--1--2--3--...
        edges = [(i, i+1) for i in range(num_qubits - 1)]
        zz_couplings = {edge: zz_strength for edge in edges}
        
    elif topology == "grid":
        # 2D rectangular grid
        grid_shape = kwargs.get("grid_shape", None)
        if grid_shape is None:
            # Auto-detect square grid
            import math
            rows = int(math.sqrt(num_qubits))
            cols = (num_qubits + rows - 1) // rows
        else:
            rows, cols = grid_shape
        
        if rows * cols < num_qubits:
            raise ValueError(
                f"Grid shape {grid_shape} too small for {num_qubits} qubits"
            )
        
        edges = []
        # Horizontal edges
        for r in range(rows):
            for c in range(cols - 1):
                q1 = r * cols + c
                q2 = r * cols + c + 1
                if q1 < num_qubits and q2 < num_qubits:
                    edges.append((q1, q2))
        
        # Vertical edges
        for r in range(rows - 1):
            for c in range(cols):
                q1 = r * cols + c
                q2 = (r + 1) * cols + c
                if q1 < num_qubits and q2 < num_qubits:
                    edges.append((q1, q2))
        
        zz_couplings = {edge: zz_strength for edge in edges}
        
    elif topology == "heavy_hex":
        # IBM Heavy-Hex topology (27-qubit Falcon processor)
        # Simplified version - users can customize for specific processors
        if num_qubits != 27:
            raise ValueError(
                f"Heavy-Hex topology requires 27 qubits, got {num_qubits}"
            )
        
        # Heavy-Hex edges (simplified, actual IBM layout is more complex)
        # This is a placeholder - real Heavy-Hex requires specific coordinates
        edges = []
        # Layer structure: 5-4-5-4-5-4 (27 qubits total)
        layers = [5, 4, 5, 4, 5, 4]
        offset = 0
        
        for layer_idx, layer_size in enumerate(layers):
            # Horizontal connections within layer
            for i in range(layer_size - 1):
                edges.append((offset + i, offset + i + 1))
            
            # Vertical connections to next layer
            if layer_idx < len(layers) - 1:
                next_layer_size = layers[layer_idx + 1]
                for i in range(min(layer_size, next_layer_size)):
                    edges.append((offset + i, offset + layer_size + i))
            
            offset += layer_size
        
        zz_couplings = {edge: zz_strength for edge in edges}
        
    elif topology == "custom":
        # User-provided custom topology
        edges = kwargs.get("edges", [])
        custom_couplings = kwargs.get("custom_couplings", {})
        
        if not edges:
            raise ValueError(
                "Custom topology requires 'edges' parameter: "
                "edges=[(i,j), ...]"
            )
        
        # Use custom couplings if provided, otherwise use default zz_strength
        zz_couplings = {}
        for edge in edges:
            i, j = min(edge), max(edge)
            zz_couplings[(i, j)] = custom_couplings.get(
                (i, j), 
                custom_couplings.get((j, i), zz_strength)
            )
    
    else:
        raise ValueError(
            f"Unknown topology: {topology}. "
            f"Available: 'linear', 'grid', 'heavy_hex', 'custom'"
        )
    
    return QubitTopology(
        num_qubits=num_qubits,
        edges=edges,
        zz_couplings=zz_couplings,
        topology_type=topology
    )


def get_crosstalk_couplings(
    topology: QubitTopology,
    qubit_model: str = "transmon_ibm"
) -> Dict[Tuple[int, int], float]:
    """
    Get realistic ZZ crosstalk couplings for a specific qubit model.
    
    Uses hardware-calibrated ZZ coupling strengths from literature:
    - IBM transmons: 1-5 MHz (arXiv:2108.12323)
    - Google Sycamore: 0.1-1 MHz (tunable couplers)
    - Rigetti: 2-10 MHz (always-on coupling)
    
    Parameters
    ----------
    topology : QubitTopology
        Qubit connectivity topology
    qubit_model : str
        Qubit model name (e.g., 'transmon_ibm', 'transmon_google')
    
    Returns
    -------
    Dict[Tuple[int, int], float]
        ZZ coupling strengths (Hz) for each connected qubit pair
    
    Example
    -------
    >>> topo = get_qubit_topology(5, topology="linear")
    >>> couplings = get_crosstalk_couplings(topo, qubit_model="transmon_ibm")
    >>> couplings  # {(0,1): 3e6, (1,2): 3e6, ...} (~3 MHz for IBM)
    """
    # Default ZZ coupling strengths by qubit model (Hz)
    ZZ_DEFAULTS = {
        "transmon_ibm": 3e6,      # 3 MHz (typical IBM)
        "transmon_google": 0.5e6,  # 0.5 MHz (tunable couplers)
        "transmon_rigetti": 5e6,   # 5 MHz (always-on)
        "transmon": 3e6,           # Default to IBM
        "flux_qubit": 10e6,        # 10 MHz (strong coupling)
        "ion_ytterbium": 0.0,      # No ZZ crosstalk (motional coupling)
        "ion_calcium": 0.0,        # No ZZ crosstalk
        "ion": 0.0,                # No ZZ crosstalk
        "spin_si": 0.0,            # Exchange coupling (not ZZ)
        "spin_gaas": 0.0,          # Exchange coupling
        "spin": 0.0,               # Exchange coupling
    }
    
    default_zz = ZZ_DEFAULTS.get(qubit_model, 1e6)  # 1 MHz fallback
    
    # Override all coupling values with model-specific defaults
    # This allows users to get realistic couplings without manually specifying
    couplings = {}
    for edge in topology.edges:
        i, j = min(edge), max(edge)
        # Always use model default (ignore topology.zz_couplings)
        couplings[(i, j)] = default_zz
    
    return couplings
