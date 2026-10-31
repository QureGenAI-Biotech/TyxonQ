"""固定浴/组态空间与重建嵌入的有限差分诊断，不将近似误差当迁移误差。"""

from dataclasses import fields, replace
import json

import numpy as np
from pyscf import gto, scf
from pyscf.fci import cistring

from tyxonq.applications.chem.algorithms.ewf import EWF
from tyxonq.applications.chem.algorithms.ewf.cluster import build_cluster
from tyxonq.applications.chem.algorithms.ewf.density import assemble_density
from tyxonq.applications.chem.algorithms.sqd import solve_sci, sci_result_to_solver_result


def test_fixed_bath_and_determinant_spaces_finite_difference():
    """H₂O 中固定 MO 坐标的浴子空间，并在相同 determinant pool 重新求 CI。

    不冻结 CI 系数或全局密度。只沿一个 O–H 分量做多步长诊断，避免把
    浴截断、组态筛选变化及振幅响应混为一谈。完整重建作为另一路对照。
    """
    mol = gto.M(atom="O 0 0 0; H 0 0 0.96; H 0.9 0 -0.24", basis="sto-3g", verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-12)
    options = dict(solver="FCI", energy_functional="density",
                   bath_options={"bathtype": "dmet", "canonicalize": False})
    reference = EWF(mf, **options)
    reference.kernel()
    gradient = reference.nuclear_gradient()[1, 2]
    coords = mol.atom_coords()
    overlap = mf.get_ovlp()
    pools = [tuple(cistring.make_strings(range(f.cluster.norb), n) for n in f.cluster.nelec)
             for f in reference.result.fragments]

    def displaced_energy(displacement, fixed):
        geometry = coords.copy()
        geometry[1, 2] += displacement
        displaced = scf.RHF(mol.set_geom_(geometry, unit="Bohr", inplace=False)).run(conv_tol=1e-12)
        if not fixed:
            emb = EWF(displaced, **options)
            return emb.kernel()
        # 该非简并小体系只需对齐各个规范 MO 的符号；禁止在此重排或优化浴。
        cross = gto.intor_cross("int1e_ovlp", mol, displaced.mol)
        mo_overlap = mf.mo_coeff.T @ cross @ displaced.mo_coeff
        assert np.all(np.abs(np.diag(mo_overlap)) > 0.99)
        aligned = displaced.mo_coeff * np.sign(np.diag(mo_overlap))
        transport = aligned @ mf.mo_coeff.T @ overlap
        inputs = []
        for fragment, result, strings in zip(reference.fragments, reference.result.fragments, pools):
            updates = {field.name: transport @ getattr(result.bath, field.name)
                       for field in fields(result.bath) if field.name.startswith("c_")}
            bath = replace(result.bath, **updates)
            cluster = build_cluster(displaced, bath, canonicalize=False)
            assert (cluster.norb, cluster.nelec) == (result.cluster.norb, result.cluster.nelec)
            sci = solve_sci(strings, cluster.h_eff, cluster.eri, cluster.norb, cluster.nelec,
                            tol=1e-13, max_cycle=200)
            solution = sci_result_to_solver_result(sci, cluster, include_cisd=False)
            inputs.append((cluster, solution, transport @ fragment.c_proj))
        return assemble_density(displaced, inputs).energy

    # 零位移仍应重现同一物理空间和密度能量。
    assert abs(displaced_energy(0, True) - reference.e_tot) < 1e-9
    report = {"analytic_hartree_per_bohr": float(gradient)}
    for fixed, key in ((True, "fixed_bath_pool_fd"), (False, "rebuilt_fd")):
        values = [(displaced_energy(h, fixed) - displaced_energy(-h, fixed)) / (2 * h)
                  for h in (1e-3, 5e-4, 2.5e-4)]
        assert np.all(np.isfinite(values))
        # 只验收差分稳定性，不要求近似解析梯度等于重建能量导数。
        assert abs(values[-1] - values[-2]) < 2e-5
        report[key] = values
        report[key + "_minus_analytic"] = values[-1] - float(gradient)
    print("fixed_space_gradient_diagnostic=" + json.dumps(report, sort_keys=True))
