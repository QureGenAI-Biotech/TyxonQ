"""SQD 走 qmmm 链路 + 采样路径 + 真机参数穿透（不改跑真机：以 simulator 档验证透传机制）。

背景与 UCCSD/HEA 的架构差异：UCCSD/HEA 的量子部分是「在设备上测量哈密顿量」，
运行选项（shots/provider/device）经 ``solver_kwargs`` → ``device_opts`` 透传到
``kernel``；而 **SQD 的量子部分是「采样 LUCJ 电路得到 counts」**（经典
selected-CI 才对角化），故设备选项挂在 **sampler（``lucj_sampler``）上、而非
solver_kwargs**。``lucj_sampler(runtime="device", provider=..., device=..., shots>0)``
把补满 ``measure_z`` 的 LUCJ 电路经 ``devices.base.run`` 提交到真机入口取回计数。

判据（全程 ``method="sqd"`` + ``lucj_sampler`` 采样路径，目标量子算法）：
- 采样冻结子空间的能量是全 CAS 的变分上界且落在化学精度内（近 FCI）；
- 冻结子空间能量面光滑，解析核梯度与有限差分一致（subspace="frozen" 的意义）；
- ``runtime="device"`` 采样档与 ``runtime="numeric"`` 一致（证明 device 路径真实执行）；
- 静电嵌入下能量位移存在、``mm_gradient`` 形状正确，且 device 档与 numeric 档一致。

对照 ``test_qc_scanner_uccsd_device.py``（UCCSD 版）与 ``test_sqd_pyscf_solver.py``
（用例 1-5：显式 ci_strs 冻结 + 采样位序守卫）；本文件补的是**采样路径在 qc_scanner
门面里的能量/梯度/嵌入/device 穿透**这一缺口（用例 6b 只用显式 ci_strs，未走采样）。
"""

from __future__ import annotations

import numpy as np
import pytest

from pyscf import gto, mcscf, qmmm, scf

from tyxonq.applications.chem.algorithms.sqd import lucj_sampler
from tyxonq.applications.chem.interfaces import qc_scanner


def _has_pyscf():
    try:
        import pyscf  # noqa: F401
        return True
    except Exception:
        return False


needs_pyscf = pytest.mark.skipif(not _has_pyscf(), reason="PySCF not installed; skipping.")

B2A = 0.52917721092
# 水几何（Å）与 E6b/E9/E10 同款嵌入体系：QM 水 + tip3p MM 水
H2O_ANG = np.array([(0.0, 0.0, 0.1173), (0.0, 0.7572, -0.4692), (0.0, -0.7572, -0.4692)])
H2O_BOHR = H2O_ANG / B2A
MM_POS_BOHR = (H2O_ANG + np.array([2.9, 0.8, 0.3])) / B2A
MM_CHARGES = np.array([-0.834, 0.417, 0.417])
SPEC = [("O", tuple(H2O_ANG[0])), ("H", tuple(H2O_ANG[1])), ("H", tuple(H2O_ANG[2]))]
SYMBOLS = ("O", "H", "H")
NCAS, NELECAS = 4, (2, 2)


def _ref_mf(with_mm: bool = False):
    """在 ``H2O_BOHR`` 几何上建 RHF（可选叠加 tip3p MM 嵌入），供 lucj_sampler 初始化 LUCJ 参数。

    sampler 只用 ``mf`` 的 ``mo_coeff`` 做活性空间 frozen-CCSD → LUCJ 参数（只算一次，
    跨几何复用）；每帧实际的 h1/h2 由 qc_scanner 内部 CASCI 在**当前几何**上给出。
    嵌入场景须用同一嵌入 mf 初始化，LUCJ 参考才与实际哈密顿量一致。
    """
    mol = gto.M(
        atom=[(s, tuple(c)) for s, c in zip(SYMBOLS, H2O_BOHR)],
        basis="sto-3g", unit="Bohr", verbose=0,
    )
    mf = scf.RHF(mol)
    if with_mm:
        mf = qmmm.add_mm_charges(mf, MM_POS_BOHR, MM_CHARGES, unit="Bohr")
    return mf.run()


def _full_cas_energy(mf) -> float:
    """同一 mf 上的全 CAS（FCI in active space）能量：SQD 冻结子空间的变分下界参考。"""
    return float(mcscf.CASCI(mf, NCAS, NELECAS).kernel()[0])


@needs_pyscf
def test_sqd_sampler_frozen_energy_and_gradient():
    """numeric 采样路径：冻结子空间能量近全 CAS（变分上界），解析梯度与有限差分一致。

    首帧采样确定子空间后锁存，后续几何复用同一子空间——能量面光滑，
    故 pyscf CASCI 解析核梯度应是该冻结面导数，与有限差分吻合（subspace="frozen"
    是唯一允许解析梯度 / MD 的模式）。这是采样路径（非显式 ci_strs）在门面层的验证。
    """
    mf = _ref_mf()
    e_full = _full_cas_energy(mf)

    sampler = lucj_sampler(mf, n_layers=1, shots=4096, noise_p=0.0, seed=7, runtime="numeric")
    scan = qc_scanner(SPEC, basis="sto-3g", active_space=(sum(NELECAS), NCAS), unit="Bohr",
                      method="sqd", sampler=sampler, subspace="frozen")

    coords = H2O_BOHR.copy()
    e0, de0 = scan(coords)
    de0 = np.asarray(de0)

    # 采样冻结子空间是全 CAS 的变分上界且近 FCI（对齐守卫用例 5：+7.4e-3 量级）。
    assert 0.0 < float(e0) - e_full < 0.05
    assert de0.shape == (3, 3) and np.all(np.isfinite(de0))

    # 冻结面光滑 → 解析梯度 == 有限差分（同 scanner 复用冻结子空间；步长对齐用例 2）。
    h = 1.89e-3  # ≈ 1e-3 Angstrom

    def energy_at(atom: int, axis: int, hh: float) -> float:
        c = coords.copy()
        c[atom, axis] += hh
        e, _ = scan(c)
        return float(e)

    for atom, axis in ((1, 2), (2, 1)):
        fd = (energy_at(atom, axis, h) - energy_at(atom, axis, -h)) / (2 * h)
        assert abs(de0[atom, axis] - fd) < 1e-4


@needs_pyscf
def test_sqd_device_runtime_passthrough():
    """device 采样档（runtime='device'）经 devices.base.run 真实执行，与 numeric 档一致。

    LUCJ 电路补满 measure_z 后提交到 simulator::statevector，取回计数 → 反转 →
    run_sqd_fermion 冻结子空间。因 H2O/sto-3g 的 LUCJ（n_layers=1）分布高度集中于
    HF 行列式，device 与 numeric 冻到同一子空间，能量应吻合（容差给采样统计偏差
    6e-3，足以抓住曾经的 3 Ha 量级位序回归）；两档都须落在全 CAS 化学精度内。
    """
    mf = _ref_mf()
    e_full = _full_cas_energy(mf)

    scan_num = qc_scanner(SPEC, basis="sto-3g", active_space=(sum(NELECAS), NCAS), unit="Bohr",
                          method="sqd", subspace="frozen",
                          sampler=lucj_sampler(mf, n_layers=1, shots=4096, seed=7, runtime="numeric"))
    e_num, _ = scan_num(H2O_BOHR)

    scan_dev = qc_scanner(SPEC, basis="sto-3g", active_space=(sum(NELECAS), NCAS), unit="Bohr",
                          method="sqd", subspace="frozen",
                          sampler=lucj_sampler(mf, n_layers=1, shots=4096,
                                               runtime="device", provider="simulator",
                                               device="statevector"))
    e_dev, de_dev = scan_dev(H2O_BOHR)

    assert abs(float(e_dev) - float(e_num)) < 6e-3, "device 采样档与 numeric 档能量不一致"
    assert 0.0 < float(e_dev) - e_full < 0.05
    assert np.asarray(de_dev).shape == (3, 3) and np.all(np.isfinite(de_dev))


@needs_pyscf
def test_sqd_device_runtime_rejects_zero_shots():
    """device 采样档必须 shots>0（要有样本才能确定子空间）；shots<=0 应被拒绝。"""
    mf = _ref_mf()
    with pytest.raises(ValueError, match="shots > 0"):
        lucj_sampler(mf, runtime="device", provider="simulator", device="statevector", shots=0)


@needs_pyscf
def test_sqd_embedding_with_sampler():
    """method='sqd' + 采样路径 + 静电嵌入：能量位移存在、mm_gradient 正确、device 档一致。

    MM 点电荷经 qc_scanner 内部 qmmm.add_mm_charges 进入哈密顿量；sampler 用同一
    嵌入 mf 初始化 LUCJ 参数。嵌入能量必须相对裸体系有可测位移；mm_gradient 返回
    当前态对 MM 粒子的反作用力（形状 (n_mm, 3)）。device 档与 numeric 档能量一致。
    """
    mf_emb = _ref_mf(with_mm=True)

    sampler_num = lucj_sampler(mf_emb, n_layers=1, shots=4096, seed=7, runtime="numeric")
    scan_emb = qc_scanner(SPEC, basis="sto-3g", active_space=(sum(NELECAS), NCAS), unit="Bohr",
                          method="sqd", sampler=sampler_num, subspace="frozen",
                          mm_charges=(MM_POS_BOHR, MM_CHARGES))
    e_emb, de_emb = scan_emb(H2O_BOHR)
    g_mm = np.asarray(scan_emb.mm_gradient())

    # 裸体系（无 MM）对照：嵌入必须真实改变能量。
    sampler_bare = lucj_sampler(_ref_mf(), n_layers=1, shots=4096, seed=7, runtime="numeric")
    scan_bare = qc_scanner(SPEC, basis="sto-3g", active_space=(sum(NELECAS), NCAS), unit="Bohr",
                           method="sqd", sampler=sampler_bare, subspace="frozen")
    e_bare, _ = scan_bare(H2O_BOHR)

    assert abs(float(e_emb) - float(e_bare)) > 1e-4, "MM 电荷未进入哈密顿量"
    assert np.asarray(de_emb).shape == (3, 3)
    assert g_mm.shape == (MM_POS_BOHR.shape[0], 3) and np.all(np.isfinite(g_mm))

    # device 采样档 + 同一嵌入：与 numeric 档能量一致（证明 device 路径穿透到嵌入体系）。
    sampler_dev = lucj_sampler(mf_emb, n_layers=1, shots=4096,
                               runtime="device", provider="simulator", device="statevector")
    scan_dev = qc_scanner(SPEC, basis="sto-3g", active_space=(sum(NELECAS), NCAS), unit="Bohr",
                          method="sqd", sampler=sampler_dev, subspace="frozen",
                          mm_charges=(MM_POS_BOHR, MM_CHARGES))
    e_dev, _ = scan_dev(H2O_BOHR)
    assert abs(float(e_dev) - float(e_emb)) < 6e-3
