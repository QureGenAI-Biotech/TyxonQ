from __future__ import annotations

"""
Numerical Quantum Natural Gradient (QNG) utilities.

This module provides a backend-agnostic, minimal implementation of the
Fubini-Study metric (quantum Fisher information matrix) evaluation using a
numerical Jacobian. It is placed under compiler/stages/gradients to keep all
gradient-related utilities together while avoiding legacy dependencies.

APIs:
  - qng_metric(f, params, eps, kernel): Return the QNG matrix via finite diffs.
  - dynamics_matrix(f, params, eps): Convenience alias with kernel='dynamics'.

Notes:
  - This implementation only depends on NumPy and expects `f(params)` to return
    a state vector (1D complex array-like). It does not rely on legacy
    backends or autodiff frameworks.
  - If higher performance or autodiff-backed QNG is desired, this utility can
    be swapped to use device-provided vectorization in the future.
"""

from typing import Any, Callable, Literal

import numpy as np


KernelType = Literal["qng", "dynamics"]


def _to_numpy(x: Any) -> np.ndarray:
    """Best-effort conversion to a 1D NumPy array."""
    try:
        import torch  # type: ignore

        if isinstance(x, torch.Tensor):  # pragma: no cover - optional
            x = x.detach().cpu().numpy()
    except Exception:  # pragma: no cover - optional
        pass
    a = np.asarray(x)
    return a.reshape(-1)


def _central_diff_jacobian(
    f: Callable[[Any], Any], params: Any, eps: float
) -> np.ndarray:
    """Compute numerical Jacobian d psi / d theta via central differences.

    params can be array-like; the function perturbs flattened parameters and
    reconstructs the original shape for each evaluation.
    """
    p = _to_numpy(params)
    shape = np.shape(params)
    size = p.size

    def _reshape(v: np.ndarray) -> Any:
        return v.reshape(shape)

    psi0 = _to_numpy(f(_reshape(p)))
    dim = psi0.size
    jac = np.zeros((size, dim), dtype=complex)

    for i in range(size):
        dp = np.zeros_like(p, dtype=float)
        dp[i] = eps
        psi_plus = _to_numpy(f(_reshape(p + dp)))
        psi_minus = _to_numpy(f(_reshape(p - dp)))
        jac[i, :] = (psi_plus - psi_minus) / (2.0 * eps)

    return jac  # shape: (num_params, state_dim)


def qng_metric(
    f: Callable[[Any], Any],
    params: Any,
    eps: float = 1e-5,
    *,
    kernel: KernelType = "qng",
) -> np.ndarray:
    """Compute the Quantum Natural Gradient metric (QFIM / Fubini-Study).

    Arguments:
      f: Callable mapping parameters -> state vector (1D complex array-like).
      params: Parameter array-like with arbitrary shape.
      eps: Finite-difference step size for central differences.
      kernel: 'qng' for projected metric; 'dynamics' for <dpsi|dpsi> metric.

    Returns:
      A real symmetric matrix of shape (num_params, num_params).
    """
    psi = _to_numpy(f(params))
    jac = _central_diff_jacobian(f, params, eps)  # (P, D)

    # Inner products: <a|b> using complex conjugation
    # G_ij = <dpsi_i | dpsi_j> - <dpsi_i|psi><psi|dpsi_j> (for 'qng')
    dpsi_dag_dpsi = jac.conj() @ jac.T  # (P, P)
    if kernel == "dynamics":
        G = dpsi_dag_dpsi
    else:
        dpsi_dag_psi = jac.conj() @ psi  # (P,)
        G = dpsi_dag_dpsi - np.outer(dpsi_dag_psi, dpsi_dag_psi.conj())

    # Ensure real symmetric output (numerical noise)
    G = np.real_if_close(0.5 * (G + G.T.conj()))
    return np.asarray(G, dtype=float)


def dynamics_matrix(
    f: Callable[[Any], Any], params: Any, eps: float = 1e-5
) -> np.ndarray:
    """Convenience alias for qng_metric with kernel='dynamics'."""
    return qng_metric(f, params, eps=eps, kernel="dynamics")


