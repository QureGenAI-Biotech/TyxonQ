"""用独立 PySCF 收缩逐自旋块验证 selected-CI 密度。"""
import numpy as np
import pytest
from pyscf.fci import cistring, selected_ci
from tyxonq.applications.chem.algorithms.sqd.gpu.rdm_factorized import spin_rdms, spin_sum


@pytest.mark.parametrize("n,ne,cap", [(4,(1,1),2), (5,(3,2),6), (6,(2,4),9), (4,(4,4),1), (4,(2,0),3)])
def test_spin_blocks_match_pyscf(n, ne, cap):
    rng = np.random.default_rng(1994)
    labels = tuple(np.sort(rng.permutation(cistring.make_strings(range(n),k))[:cap]) for k in ne)
    c = rng.normal(size=tuple(map(len,labels)))
    vec = selected_ci._as_SCIvector(c/np.linalg.norm(c), labels)
    spin = spin_rdms(vec,n,ne)
    reference = selected_ci.make_rdm1s(vec,n,ne), selected_ci.make_rdm2s(vec,n,ne)
    for blocks, other in zip(spin, reference):
        for x, y in zip(blocks, other):
            np.testing.assert_allclose(x,y,rtol=0,atol=2e-14)
    d1, d2 = spin_sum(*spin)
    np.testing.assert_allclose(d1, selected_ci.make_rdm1(vec,n,ne), atol=2e-14)
    np.testing.assert_allclose(d2, selected_ci.make_rdm2(vec,n,ne), atol=2e-14)
