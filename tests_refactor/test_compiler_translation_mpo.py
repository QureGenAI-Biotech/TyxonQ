from __future__ import annotations

import numpy as np

from tyxonq.compiler.translation.mpo_converters import contract_mpo_to_matrix


def _make_identity_mpo(n: int):
    # Build MPO tensors for identity: each site (1,2,2,1) with identity on physical legs
    Ts = []
    I = np.eye(2, dtype=np.complex128)
    for _ in range(n):
        T = np.zeros((1, 2, 2, 1), dtype=np.complex128)
        T[0, :, :, 0] = I
        Ts.append(T)
    return Ts


def test_contract_mpo_identity_gives_identity_matrix():
    Ts = _make_identity_mpo(3)
    M = contract_mpo_to_matrix(Ts)
    dim = 2 ** 3
    np.testing.assert_allclose(M, np.eye(dim, dtype=np.complex128))


