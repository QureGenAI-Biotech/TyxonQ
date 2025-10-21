from __future__ import annotations

from typing import Any
import numpy as np
from ....numerics import NumericBackend as nb
from ....numerics.api import ArrayBackend


def init_density(num_qubits: int, backend: ArrayBackend | None = None) -> Any:
    K = backend or nb
    dim = 1 << num_qubits
    rho = K.zeros((dim, dim), dtype=K.complex128)
    # set |0...0><0...0|
    one = K.array(1.0 + 0.0j, dtype=K.complex128)
    rho_np = K.to_numpy(rho)
    rho_np[0, 0] = 1.0 + 0.0j
    return K.asarray(rho_np)


def apply_1q_density(backend: Any, rho: Any, U: Any, q: int, n: int) -> Any:
    K = backend or nb
    psi = K.reshape(K.asarray(rho), (2,) * (2 * n))
    letters = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
    reserved = set(['a', 'b', 'x', 'y'])
    # choose axis symbols disjoint from reserved
    axis_symbols = [ch for ch in letters if ch not in reserved]
    r_axes = axis_symbols[:n]
    c_axes = axis_symbols[n:2 * n]
    r_in = r_axes.copy(); c_in = c_axes.copy()
    r_in[q] = 'a'; c_in[q] = 'b'
    r_out = r_axes.copy(); c_out = c_axes.copy()
    r_out[q] = 'x'; c_out[q] = 'y'
    spec = f"xa,{''.join(r_in + c_in)},by->{''.join(r_out + c_out)}"
    U_bk = K.asarray(U)
    U_conj = K.conj(U_bk)
    # use indices 'yb' to represent conjugate-transpose without materializing a transpose
    spec = f"xa,{''.join(r_in + c_in)},yb->{''.join(r_out + c_out)}"
    out = K.einsum(spec, U_bk, psi, U_conj)
    return K.reshape(K.asarray(out), (1 << n, 1 << n))


def apply_2q_density(backend: Any, rho: Any, U4: Any, q0: int, q1: int, n: int) -> Any:
    if q0 == q1:
        return rho
    K = backend or nb
    psi = K.reshape(K.asarray(rho), (2,) * (2 * n))
    letters = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
    reserved = set(['a', 'b', 'c', 'd', 'w', 'x', 'y', 'z'])
    axis_symbols = [ch for ch in letters if ch not in reserved]
    r_axes = axis_symbols[:n]
    c_axes = axis_symbols[n:2 * n]
    r_in = r_axes.copy(); c_in = c_axes.copy()
    r_in[q0] = 'a'; r_in[q1] = 'b'
    c_in[q0] = 'c'; c_in[q1] = 'd'
    r_out = r_axes.copy(); c_out = c_axes.copy()
    r_out[q0] = 'w'; r_out[q1] = 'x'
    c_out[q0] = 'y'; c_out[q1] = 'z'
    spec = f"wxab,{''.join(r_in + c_in)},yzcd->{''.join(r_out + c_out)}"
    U4n = K.asarray(np.reshape(np.asarray(U4), (2, 2, 2, 2)))
    U4c = K.conj(U4n)
    # specify indices 'yzcd' for conj(U4) to represent conjugate-transpose on the appropriate axes
    out = K.einsum(spec, U4n, psi, U4c)
    return K.reshape(K.asarray(out), (1 << n, 1 << n))


def exp_z_density(backend: Any, rho: Any, q: int, n: int) -> Any:
    # Fast path via diagonal populations; correct for Z expectation
    K = backend or nb
    dim = 1 << n
    diag = K.real(K.diag(rho))
    bits = (np.arange(dim) >> (n - 1 - q)) & 1
    signs = 1.0 - 2.0 * bits
    return K.sum(K.asarray(K.to_numpy(diag) * signs))


def apply_kraus_density(
    rho: Any,
    kraus_operators: Any,
    qubit: int,
    num_qubits: int,
    backend: ArrayBackend | None = None
) -> Any:
    """Apply Kraus channel to density matrix.
    
    For density matrix simulation, Kraus operators define a completely positive
    trace-preserving (CPTP) map via:
    
        ρ → ∑ᵢ Kᵢ ρ K†ᵢ
    
    This is the exact evolution of the mixed state under the quantum channel.
    
    Args:
        rho: Input density matrix of shape (2^num_qubits, 2^num_qubits)
        kraus_operators: List of Kraus operators {K₀, K₁, ..., Kₙ}
                        Each Kᵢ is a 2×2 matrix for single-qubit channel
        qubit: Target qubit index
        num_qubits: Total number of qubits
        backend: Optional numeric backend
        
    Returns:
        Updated density matrix after channel application
        
    Notes:
        - Completeness: ∑ᵢ K†ᵢKᵢ = I ensures trace preservation
        - Uses efficient tensor network contraction via einsum
        - Exact mixed state evolution (no sampling)
        
    Example:
        >>> # Depolarizing channel with p=0.1
        >>> p = 0.1
        >>> K0 = sqrt(1-p) * np.eye(2)
        >>> K1 = sqrt(p/3) * np.array([[0,1],[1,0]])  # X
        >>> K2 = sqrt(p/3) * np.array([[0,-1j],[1j,0]])  # Y  
        >>> K3 = sqrt(p/3) * np.array([[1,0],[0,-1]])  # Z
        >>> rho_out = apply_kraus_density(rho, [K0,K1,K2,K3], qubit=0, num_qubits=2)
    """
    K = backend or nb
    
    # Reshape density matrix to tensor form
    n = num_qubits
    psi = K.reshape(K.asarray(rho), (2,) * (2 * n))
    
    # Build einsum specification for Kraus channel
    # ρ' = ∑ᵢ Kᵢ ρ K†ᵢ
    letters = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
    reserved = {'a', 'b', 'x', 'y'}
    axis_symbols = [ch for ch in letters if ch not in reserved]
    
    r_axes = axis_symbols[:n]  # row indices
    c_axes = axis_symbols[n:2 * n]  # column indices
    
    # Input axes: replace target qubit indices
    r_in = r_axes.copy()
    c_in = c_axes.copy()
    r_in[qubit] = 'a'
    c_in[qubit] = 'b'
    
    # Output axes
    r_out = r_axes.copy()
    c_out = c_axes.copy()
    r_out[qubit] = 'x'
    c_out[qubit] = 'y'
    
    # Einsum specification: Kᵢ @ ρ @ K†ᵢ
    # "xa" for Kᵢ, "yb" for K†ᵢ (conjugate transpose)
    spec = f"xa,{''.join(r_in + c_in)},yb->{''.join(r_out + c_out)}"
    
    # Apply all Kraus operators and sum
    out = None
    for kraus_op in kraus_operators:
        K_bk = K.asarray(kraus_op)
        K_conj = K.conj(K_bk)
        
        # Compute Kᵢ ρ K†ᵢ
        term = K.einsum(spec, K_bk, psi, K_conj)
        
        if out is None:
            out = term
        else:
            out = out + term
    
    # Reshape back to matrix form
    dim = 1 << n
    return K.reshape(K.asarray(out), (dim, dim))


__all__ = [
    "init_density",
    "apply_1q_density",
    "apply_2q_density",
    "exp_z_density",
    "apply_kraus_density",
]


