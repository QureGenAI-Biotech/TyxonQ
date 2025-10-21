"""Standard Quantum Noise Channels.

This module provides standard quantum noise channel implementations as
Kraus operator representations. These channels model realistic quantum
hardware imperfections and are used across all simulators.

All functions return Kraus operators {K₀, K₁, ...} satisfying the
completeness relation: ∑ᵢ K†ᵢKᵢ = I (CPTP map).

Author: TyxonQ Development Team
"""

from __future__ import annotations

from typing import List
import numpy as np


__all__ = [
    "depolarizing_channel",
    "amplitude_damping_channel", 
    "phase_damping_channel",
    "pauli_channel",
    "measurement_channel",
]


def depolarizing_channel(p: float) -> List[np.ndarray]:
    """Depolarizing channel: uniform random Pauli errors.
    
    Models isotropic noise where each Pauli error (X, Y, Z) occurs
    with equal probability p/3.
    
    Kraus operators:
        K₀ = √(1-p) · I
        K₁ = √(p/3) · X
        K₂ = √(p/3) · Y
        K₃ = √(p/3) · Z
    
    Args:
        p: Total error probability (0 ≤ p ≤ 1)
        
    Returns:
        List of 4 Kraus operators (2×2 complex matrices)
        
    Physical interpretation:
        - p=0: No noise (identity channel)
        - p=1: Complete depolarization → maximally mixed state
        - Common model for isotropic decoherence
        
    Example:
        >>> kraus_ops = depolarizing_channel(p=0.1)
        >>> # Apply to density matrix via apply_kraus_density()
    """
    p = float(p)
    if not (0 <= p <= 1):
        raise ValueError(f"Depolarizing parameter p={p} must be in [0,1]")
    
    I = np.eye(2, dtype=np.complex128)
    X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    
    K0 = np.sqrt(1 - p) * I
    K1 = np.sqrt(p / 3) * X
    K2 = np.sqrt(p / 3) * Y
    K3 = np.sqrt(p / 3) * Z
    
    return [K0, K1, K2, K3]


def amplitude_damping_channel(gamma: float) -> List[np.ndarray]:
    """Amplitude damping channel: energy relaxation |1⟩ → |0⟩.
    
    Models T₁ relaxation process in superconducting qubits and other
    quantum systems. Energy dissipates to environment.
    
    Kraus operators:
        K₀ = [[1, 0], [0, √(1-γ)]]
        K₁ = [[0, √γ], [0, 0]]
    
    Args:
        gamma: Damping parameter (0 ≤ γ ≤ 1)
              Physical: γ ≈ 1 - exp(-t_gate/T₁)
        
    Returns:
        List of 2 Kraus operators (2×2 complex matrices)
        
    Physical interpretation:
        - γ=0: No damping (identity)
        - γ=1: Complete relaxation to |0⟩
        - Models spontaneous emission, energy loss
        - NOT Hermitian-preserving (non-unital channel)
        
    Example:
        >>> # For T₁=50μs gate time t=100ns
        >>> gamma = 1 - np.exp(-0.1/50)  # ≈ 0.002
        >>> kraus_ops = amplitude_damping_channel(gamma)
    """
    g = float(gamma)
    if not (0 <= g <= 1):
        raise ValueError(f"Amplitude damping gamma={g} must be in [0,1]")
    
    K0 = np.array([[1.0, 0.0], [0.0, np.sqrt(1 - g)]], dtype=np.complex128)
    K1 = np.array([[0.0, np.sqrt(g)], [0.0, 0.0]], dtype=np.complex128)
    
    return [K0, K1]


def phase_damping_channel(lmbda: float) -> List[np.ndarray]:
    """Phase damping channel: pure dephasing (no energy loss).
    
    Models T₂ dephasing process where quantum coherence decays but
    populations remain unchanged. Diagonal elements preserved.
    
    Kraus operators:
        K₀ = [[1, 0], [0, √(1-λ)]]
        K₁ = [[0, 0], [0, √λ]]
    
    Args:
        lmbda: Dephasing parameter (0 ≤ λ ≤ 1)
               Physical: λ ≈ 1 - exp(-t_gate/T₂)
        
    Returns:
        List of 2 Kraus operators (2×2 complex matrices)
        
    Physical interpretation:
        - λ=0: No dephasing (identity)
        - λ=1: Complete dephasing → classical mixture
        - Preserves |0⟩⟨0| and |1⟩⟨1| populations
        - Destroys off-diagonal coherence terms
        - Models pure T₂* process (T₂ = T₂* + T₁/2)
        
    Example:
        >>> # For T₂=30μs gate time t=100ns
        >>> lmbda = 1 - np.exp(-0.1/30)  # ≈ 0.0033
        >>> kraus_ops = phase_damping_channel(lmbda)
    """
    l = float(lmbda)
    if not (0 <= l <= 1):
        raise ValueError(f"Phase damping lambda={l} must be in [0,1]")
    
    K0 = np.array([[1.0, 0.0], [0.0, np.sqrt(1 - l)]], dtype=np.complex128)
    K1 = np.array([[0.0, 0.0], [0.0, np.sqrt(l)]], dtype=np.complex128)
    
    return [K0, K1]


def pauli_channel(px: float, py: float, pz: float) -> List[np.ndarray]:
    """Pauli channel: asymmetric X/Y/Z errors.
    
    Generalization of depolarizing channel allowing independent error
    rates for each Pauli operator. More realistic for certain hardware.
    
    Kraus operators:
        K₀ = √(1 - px - py - pz) · I
        K₁ = √px · X
        K₂ = √py · Y
        K₃ = √pz · Z
    
    Args:
        px: Probability of X error (bit flip)
        py: Probability of Y error (bit+phase flip)
        pz: Probability of Z error (phase flip)
        
    Returns:
        List of 4 Kraus operators (2×2 complex matrices)
        
    Constraints:
        px + py + pz ≤ 1 (probability conservation)
        
    Physical interpretation:
        - Dephasing-dominant: pz >> px, py
        - Bit-flip-dominant: px >> py, pz
        - Depolarizing: px = py = pz = p/3
        
    Example:
        >>> # Dephasing-dominant noise
        >>> kraus_ops = pauli_channel(px=0.01, py=0.01, pz=0.05)
    """
    px = float(px)
    py = float(py) 
    pz = float(pz)
    
    if px < 0 or py < 0 or pz < 0:
        raise ValueError(f"Pauli channel probabilities must be non-negative, got px={px}, py={py}, pz={pz}")
    
    p_total = px + py + pz
    if p_total > 1:
        raise ValueError(f"Sum of Pauli probabilities {p_total} exceeds 1")
    
    I = np.eye(2, dtype=np.complex128)
    X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    
    p0 = max(0.0, 1.0 - p_total)
    
    K0 = np.sqrt(p0) * I
    K1 = np.sqrt(px) * X
    K2 = np.sqrt(py) * Y
    K3 = np.sqrt(pz) * Z
    
    return [K0, K1, K2, K3]


def measurement_channel(p: float) -> List[np.ndarray]:
    """Measurement channel: probabilistic projective measurement.
    
    Models measurement-induced phase transition (MIPT) and other protocols
    involving stochastic measurements. With probability p, perform Z-basis
    measurement; with probability 1-p, apply identity.
    
    Kraus operators:
        K₀ = √p · |0⟩⟨0|  (project to |0⟩)
        K₁ = √p · |1⟩⟨1|  (project to |1⟩)
        K₂ = √(1-p) · I   (no measurement)
    
    Args:
        p: Measurement probability (0 ≤ p ≤ 1)
        
    Returns:
        List of 3 Kraus operators (2×2 complex matrices)
        
    Physical interpretation:
        - p=0: No measurement (identity channel)
        - p=1: Certain measurement → classical state
        - 0<p<1: Stochastic measurement (MIPT regime)
        - Competes with unitary evolution in MIPT
        
    Applications:
        - Measurement-induced phase transitions (MIPT)
        - Quantum Zeno effect
        - Monitored quantum circuits
        - Post-selection protocols
        
    Example:
        >>> # MIPT with 10% measurement rate
        >>> kraus_ops = measurement_channel(p=0.1)
        >>> # Apply in random circuit evolution
    """
    p = float(p)
    if not (0 <= p <= 1):
        raise ValueError(f"Measurement probability p={p} must be in [0,1]")
    
    # Projectors onto computational basis
    K0 = np.sqrt(p) * np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)  # |0⟩⟨0|
    K1 = np.sqrt(p) * np.array([[0.0, 0.0], [0.0, 1.0]], dtype=np.complex128)  # |1⟩⟨1|
    K2 = np.sqrt(1 - p) * np.eye(2, dtype=np.complex128)  # Identity
    
    return [K0, K1, K2]
