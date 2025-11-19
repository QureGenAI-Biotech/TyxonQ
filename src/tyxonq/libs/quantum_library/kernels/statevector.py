from __future__ import annotations

from typing import Any, Sequence
import numpy as np
from ....numerics import NumericBackend as nb
from ....numerics.api import ArrayBackend


__all__ = [
    "init_statevector",
    "apply_1q_statevector",
    "apply_2q_statevector",
    "apply_kqubit_unitary",
    "expect_z_statevector",
    "apply_kraus_statevector",
]


def init_statevector(num_qubits: int, backend: ArrayBackend | None = None) -> Any:
    K = backend or nb
    if num_qubits <= 0:
        return K.array([1.0 + 0.0j], dtype=K.complex128)
    dim = 1 << num_qubits
    data = [1.0 + 0.0j] + [0.0 + 0.0j] * (dim - 1)
    return K.array(data, dtype=K.complex128)


def apply_1q_statevector(backend: Any, state: Any, gate2: Any, qubit: int, num_qubits: int) -> Any:
    K = backend or nb
    # CRITICAL: Use backend operations throughout to preserve gradient chain
    # Use the backend's reshape and asarray to maintain PyTorch autograd if present
    psi = K.reshape(K.asarray(state), (2,) * num_qubits)
    g2 = K.asarray(gate2)
    letters = list("abcdefghijklmnopqrstuvwxyz")
    # Reserve 'a','b' for gate indices; use distinct axis symbols starting from 'c'
    axes = letters[2:2 + num_qubits]
    in_axes = axes.copy(); in_axes[qubit] = 'b'
    out_axes = axes.copy(); out_axes[qubit] = 'a'
    spec = f"ab,{''.join(in_axes)}->{''.join(out_axes)}"
    arr = K.einsum(spec, g2, psi)
    # CRITICAL: Don't call K.asarray on the result - K.einsum already returns backend tensor
    return K.reshape(arr, (-1,))


def apply_2q_statevector(backend: Any, state: Any, gate4: Any, q0: int, q1: int, num_qubits: int) -> Any:
    if q0 == q1:
        return state
    K = backend or nb
    psi = K.reshape(K.asarray(state), (2,) * num_qubits)
    letters = list("abcdefghijklmnopqrstuvwxyz")
    # Reserve 'a','b','c','d' for gate indices; use distinct axis symbols starting from 'e'
    axes = letters[4:4 + num_qubits]
    in_axes = axes.copy(); in_axes[q0] = 'c'; in_axes[q1] = 'd'
    out_axes = axes.copy(); out_axes[q0] = 'a'; out_axes[q1] = 'b'
    spec = f"abcd,{''.join(in_axes)}->{''.join(out_axes)}"
    g4 = K.reshape(K.asarray(gate4), (2, 2, 2, 2))
    arr = K.einsum(spec, g4, psi)
    # CRITICAL: Don't call K.asarray on the result - K.einsum already returns backend tensor
    return K.reshape(arr, (-1,))


def expect_z_statevector(state: Any, qubit: int, num_qubits: int, backend: ArrayBackend | None = None) -> Any:
    K = backend or nb
    s = K.reshape(K.asarray(state), (2,) * num_qubits)
    s_perm = K.moveaxis(s, qubit, 0)
    s2 = K.abs(K.reshape(s_perm, (2, -1))) ** 2  # type: ignore[operator]
    sums = K.sum(s2, axis=1)
    return sums[0] - sums[1]


def apply_kqubit_unitary(state: Any, unitary: Any, qubit_indices: Sequence[int], num_qubits: int, backend: ArrayBackend | None = None) -> Any:
    """Apply k-qubit unitary to statevector.
    
    This is the core numerical routine for applying arbitrary unitary matrices
    to one or more qubits in a statevector. It uses tensor reshaping and 
    permutation to efficiently apply the unitary transformation.
    
    Args:
        state: Statevector of shape (2^num_qubits,)
        unitary: Unitary matrix of shape (2^k, 2^k) where k = len(qubit_indices)
        qubit_indices: List of target qubit indices (LSB-first ordering)
        num_qubits: Total number of qubits in the system
        backend: Optional numeric backend (NumPy, PyTorch, etc.)
        
    Returns:
        Updated statevector after unitary application
        
    Notes:
        - Uses LSB-first qubit ordering (axis 0 = qubit 0)
        - Efficient tensor contraction via transpose + matmul
        - Compatible with all numeric backends (numpy, pytorch, cupy)
    """
    K = backend or nb
    k = len(qubit_indices)
    
    if k == 0:
        return state
    
    # Convert inputs to backend arrays
    psi = K.asarray(state,dtype=K.complex128)
    U = K.asarray(unitary,dtype=K.complex128)  # Keep tensor type to preserve autograd
    
    # Reshape unitary to proper shape
    dim = 1 << k  # 2^k
    U = K.reshape(U, (dim, dim))
    
    # Build axis permutation: move target qubits to the end
    axes_all = list(range(num_qubits))
    target_axes = [int(q) for q in qubit_indices]
    keep_axes = [ax for ax in axes_all if ax not in target_axes]
    perm_forward = keep_axes + target_axes
    perm_inverse = np.argsort(perm_forward).tolist()
    
    # Reshape to tensor and permute
    psi_nd = K.reshape(psi, (2,) * num_qubits)
    psi_perm = K.transpose(psi_nd, perm_forward)
    
    # Reshape to matrix: [remaining dims, 2^k]
    remaining_dim = 1 << (num_qubits - k)
    psi_matrix = K.reshape(psi_perm, (remaining_dim, dim))
    
    # Apply unitary: (remaining, dim) @ (dim, dim).T -> (remaining, dim)
    psi_out = psi_matrix @ K.transpose(U)
    
    # Reshape back and inverse permute
    psi_out_nd = K.reshape(psi_out, (2,) * num_qubits)
    psi_result = K.transpose(psi_out_nd, perm_inverse)
    
    return K.reshape(psi_result, (-1,))


def apply_kraus_statevector(
    state: Any,
    kraus_operators: Sequence[Any],
    qubit: int,
    num_qubits: int,
    status: float | None = None,
    backend: ArrayBackend | None = None
) -> Any:
    """Apply Kraus channel to statevector via stochastic unraveling.
    
    For statevector simulation, Kraus operators are applied via Monte Carlo
    trajectory method: randomly select one Kraus operator based on Born rule,
    then apply it and renormalize.
    
    Physical interpretation:
        |ψ⟩ → Kᵢ|ψ⟩ / ||Kᵢ|ψ⟩||  with probability ||Kᵢ|ψ⟩||²
    
    Args:
        state: Input statevector of shape (2^num_qubits,)
        kraus_operators: List of Kraus operators {K₀, K₁, ..., Kₙ}
                        Each Kᵢ is a 2×2 matrix for single-qubit channel
        qubit: Target qubit index
        num_qubits: Total number of qubits
        status: Random variable in [0,1] for stochastic selection.
               If None, uses uniform random sampling.
        backend: Optional numeric backend
        
    Returns:
        Updated statevector after Kraus channel application
        
    Notes:
        - Completeness: ∑ᵢ K†ᵢKᵢ = I ensures valid CPTP map
        - Stochastic unraveling: single trajectory approximation
        - For exact mixed state evolution, use density matrix simulator
        
    Example:
        >>> # Amplitude damping with γ=0.1
        >>> K0 = [[1, 0], [0, sqrt(0.9)]]
        >>> K1 = [[0, sqrt(0.1)], [0, 0]]
        >>> state_out = apply_kraus_statevector(state, [K0, K1], qubit=0, num_qubits=2, status=0.5)
    """
    K = backend or nb
    
    # Convert Kraus operators to backend arrays
    kraus_list = [K.asarray(op) for op in kraus_operators]
    
    # Compute selection probabilities via Born rule
    # P(i) = ||Kᵢ|ψ⟩||² = ⟨ψ|K†ᵢKᵢ|ψ⟩
    probabilities = []
    for kraus_op in kraus_list:
        # Apply Kraus operator to get Kᵢ|ψ⟩
        psi_transformed = apply_1q_statevector(K, state, kraus_op, qubit, num_qubits)
        # Compute norm squared: ||Kᵢ|ψ⟩||²
        norm_sq = K.real(K.sum(K.conj(psi_transformed) * psi_transformed))
        probabilities.append(norm_sq)
    
    # Normalize probabilities (should sum to 1 due to completeness)
    probs_array = K.array(probabilities, dtype=K.float64)
    prob_sum = K.sum(probs_array)
    probs_normalized = probs_array / prob_sum
    
    # Select Kraus operator based on status (random variable)
    if status is None:
        # Use random sampling if status not provided
        import random
        status = random.random()
    
    # Cumulative probability distribution for selection
    probs_np = K.to_numpy(probs_normalized)
    cumsum_np = np.cumsum(probs_np)
    
    # Find which Kraus operator to apply
    selected_idx = 0
    for idx, cum_prob in enumerate(cumsum_np):
        if status <= float(cum_prob):
            selected_idx = idx
            break
    
    # Apply selected Kraus operator
    selected_kraus = kraus_list[selected_idx]
    psi_out = apply_1q_statevector(K, state, selected_kraus, qubit, num_qubits)
    
    # Renormalize (important for probability conservation)
    norm = K.sqrt(K.real(K.sum(K.conj(psi_out) * psi_out)))
    psi_normalized = psi_out / norm
    
    return psi_normalized
