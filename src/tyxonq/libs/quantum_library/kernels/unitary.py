from __future__ import annotations

"""Unitary matrices for core gates and retrieval helpers.

This module provides canonical unitary matrices for a small set of native
gates used across the refactored core. It intentionally depends only on
NumPy (not on the simulator numerics backends) because unitary specification
is a static definition and is used by compiler/tests without executing on a
device.

Exposed API:
    - get_unitary(name: str, *params) -> np.ndarray

Notes
-----
All matrices are returned with dtype=complex128 and a conventional qubit
ordering, where multi-qubit unitaries are ordered on computational basis
|00>, |01>, |10>, |11> (control is the first wire for CX).
"""

import numpy as np

__all__ = ["get_unitary"]


def _u_h() -> np.ndarray:
    factor = 1.0 / np.sqrt(2.0)
    return factor * np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.complex128)


def _u_rz(theta: float) -> np.ndarray:
    e_m = np.exp(-1j * theta / 2.0)
    e_p = np.exp(1j * theta / 2.0)
    return np.diag([e_m, e_p]).astype(np.complex128)


def _u_cx() -> np.ndarray:
    # Control on the first qubit, target on the second.
    return np.array(
        [[1.0, 0.0, 0.0, 0.0],
         [0.0, 1.0, 0.0, 0.0],
         [0.0, 0.0, 0.0, 1.0],
         [0.0, 0.0, 1.0, 0.0]],
        dtype=np.complex128,
    )


def get_unitary(name: str, *params: float) -> np.ndarray:
    """Return the unitary matrix for a supported gate.

    Parameters
    ----------
    name:
        Gate name. Supported: "h", "rz", "cx".
    *params:
        Gate parameters. For "rz", expects one angle ``theta``.

    Returns
    -------
    np.ndarray
        The unitary matrix with dtype=complex128.

    Raises
    ------
    ValueError
        If the gate is unknown or parameters are missing.
    """
    key = name.lower()
    if key == "h":
        return _u_h()
    if key == "rz":
        if len(params) != 1:
            raise ValueError("rz expects exactly one parameter: theta")
        (theta,) = params
        return _u_rz(float(theta))
    if key == "cx":
        return _u_cx()
    raise ValueError(f"Unknown gate name: {name}")


