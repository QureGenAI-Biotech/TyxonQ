"""Tier 1a 磁材料轨道验收：S(q,ω) 引擎（M1 ED 基准 + M2 定标门）。

M1（ED 基准）：S(q,ω) 的强峰支撑集 ⊂ H 的 ED 激发能谱 {E_m-E_n}
（簇质心对齐，容差 = 谱分辨率：max(ω 网格步长, Hann 主瓣宽)）。
物理直觉参考：无限 XX 链单粒子色散 ω(q)=2J|cos q|（有限开边界链
色散折叠、双线不可分属正常谱分辨）。

M2（芯片定标预演）：数值精确档 vs 线路档（simulator statevector，
Trotter 化）同一模型 S(q,ω) 一致——真机定标（ZNE 校准等）沿用同一
对照结构。

单位：原子单位 ħ=1。设备档 provider="simulator"。
"""

from __future__ import annotations

import numpy as np
import pytest

from tyxonq.applications.chem.dynamics.models.spin_lattice import (
    heisenberg_chain,
)
from tyxonq.applications.chem.dynamics.observables import (
    build_pauli_matrix,
    pauli_expectation,
)
from tyxonq.applications.chem.dynamics.sqw import (
    SQW_RESULT_SCHEMA,
    compute_spin_correlations,
    compute_sqw,
)


# ------------------------------------------------------------------ #
# 模型编码正确性
# ------------------------------------------------------------------ #
def test_heisenberg_chain_pauli_items():
    model = heisenberg_chain(3, j_xy=1.0, j_z=0.5, h=0.2, dmi=0.3)
    items = model.pauli_items()
    labels = {(tuple(sorted(ops)), round(float(c), 6)) for c, ops in items}
    assert (tuple(sorted([("X", 0), ("X", 1)])), 1.0) in labels
    assert (tuple(sorted([("Y", 0), ("Y", 1)])), 1.0) in labels
    assert (tuple(sorted([("Z", 0), ("Z", 1)])), 0.5) in labels
    # DMI 反对称：+D X_iY_j - D Y_iX_j
    assert (tuple(sorted([("X", 0), ("Y", 1)])), 0.3) in labels
    assert (tuple(sorted([("Y", 0), ("X", 1)])), -0.3) in labels
    # 纵场
    assert ((("Z", 0),), 0.2) in labels


def test_initial_bitstrings_and_pbc():
    """初态位串一律 little-endian（右端 = qubit 0）。"""
    m = heisenberg_chain(4, j_xy=1.0, j_z=0.0)
    assert m.initial_bitstring("fm") == "1111"
    assert m.initial_bitstring("afm") == "0101"  # q0=1,q1=0,q2=1,q3=0
    assert m.initial_bitstring("dw") == "0100"  # q2=1（中间翻转）
    m_pbc = heisenberg_chain(3, j_xy=1.0, j_z=0.0, pbc=True)
    assert any(i == 2 and j == 0 for _, i, j, _ in m_pbc.couplings)


def test_to_pulse_program_produces_waves():
    """演示级 DAQC 映射：波形 → lqcloud waves 可构造（方言层冒烟）。"""
    from tyxonq.compiler.platforms.lqcloud.lqwave_adapter import (
        is_waves_dict,
        program_to_waves,
    )

    model = heisenberg_chain(3, j_xy=0.5, j_z=0.0, h=0.3)
    prog = model.to_pulse_program(duration=80)
    waves = program_to_waves(prog)
    assert is_waves_dict(waves)
    assert "0.xy" in waves and "0.z" in waves


# ------------------------------------------------------------------ #
# M1：ED 基准（谱支撑集 ⊂ ED 激发能）
# ------------------------------------------------------------------ #
def _cluster_peaks(omega, spec, *, threshold=0.15, gap=0.3):
    """强峰聚合为簇（间隔 < gap 归并），返回簇质心列表。"""
    flat = spec / (spec.max() or 1.0)
    rows, _ = np.where(flat > threshold)
    peak_oms = np.unique(omega[rows])
    if len(peak_oms) == 0:
        return []
    clusters = [[peak_oms[0]]]
    for w in peak_oms[1:]:
        if w - clusters[-1][-1] < gap:
            clusters[-1].append(w)
        else:
            clusters.append([w])
    return [float(np.mean(c)) for c in clusters]


def test_sqw_peaks_on_ed_excitations_m1():
    """XX 链 n=4：S(q,ω) 强峰簇质心 ⊂ ED 激发能 ± 谱分辨率。"""
    model = heisenberg_chain(4, j_xy=0.5, j_z=0.0)
    times = np.linspace(0.0, 60.0, 601)
    out = compute_sqw(model, times, ops=("X",), n_momentum=16, n_frequency=600)
    assert out["schema"] == SQW_RESULT_SCHEMA
    H = build_pauli_matrix(model.pauli_items(), 4)
    evals = np.linalg.eigvalsh(H)
    diffs = np.unique(np.round(np.abs(np.subtract.outer(evals, evals)), 6))
    diffs = diffs[diffs > 1e-6]
    centers = _cluster_peaks(out["omega"], out["Sqw"])
    assert len(centers) >= 2, "应至少出现两个可分辨峰簇"
    # 谱分辨率 = max(ω 步长 0.0524, Hann 主瓣 π/T≈0.0524)；对齐容差 0.15
    for c in centers:
        assert min(abs(c - d) for d in diffs) < 0.15, \
            f"峰簇质心 {c:.4f} 未落在 ED 激发能上"


# ------------------------------------------------------------------ #
# M2 预演：数值 vs 线路（simulator）一致性
# ------------------------------------------------------------------ #
@pytest.mark.skip(reason="M2 线路档定标留待真机统一做（用户 2026-11-12 拍板，不阻塞）")
def test_sqw_numeric_vs_circuits_m2():
    """M2 预演：踢动线性响应两档（数值 vs simulator 线路）S(q,ω) 一致性。

    保留实现供真机统一验收时启用；当前 skip 不阻塞离线主线。
    """
    from tyxonq.applications.chem.dynamics.sqw import compute_sqw_kicked

    model = heisenberg_chain(4, j_xy=0.5, j_z=0.0)
    times = np.linspace(0.0, 4.0, 41)
    num = compute_sqw_kicked(model, times, h_kick=0.1, n_frequency=100)
    circ = compute_sqw_kicked(
        model, times, h_kick=0.1, provider="simulator",
        shots=32768, n_frequency=100,
    )
    n1 = num["Sqw"] / (abs(num["Sqw"]).max() or 1.0)
    c1 = circ["Sqw"] / (abs(circ["Sqw"]).max() or 1.0)
    assert float(np.corrcoef(n1.ravel(), c1.ravel())[0, 1]) > 0.85


def test_correlations_exact_at_t_zero():
    """t=0 一致性：C(0)_{ij} = ⟨ψ0|S^α_i S^β_j|ψ0⟩。

    i=j 时 Z_i·Z_i = I → ⟨Z_i²⟩=+1（Pauli items 惯例不允许同比特重复，
    故参照分两种情形断言）。
    """
    model = heisenberg_chain(3, j_xy=0.5, j_z=0.0)
    psi0 = model.initial_state("dw")
    res = compute_spin_correlations(
        model.pauli_items(), psi0, [0.0], 3, ops=("Z",)
    )
    corr = res["corr"]
    for j in range(3):
        for i in range(3):
            if i == j:
                assert abs(corr[0, i, j, 0, 0] - 1.0) < 1e-12, "Z_i² = I"
            else:
                expect = pauli_expectation(psi0, [(1.0, [("Z", i), ("Z", j)])], 3)
                assert abs(corr[0, i, j, 0, 0] - expect) < 1e-12


def test_compute_spin_correlations_provider_guard():
    model = heisenberg_chain(2, j_xy=0.5, j_z=0.0)
    with pytest.raises(ValueError):
        compute_spin_correlations(
            model.pauli_items(), model.initial_state("dw"),
            np.linspace(0, 0.2, 3), 2, provider="bogus",
        )
    with pytest.raises(ValueError):
        compute_spin_correlations(
            model.pauli_items(), model.initial_state("dw"),
            np.linspace(0, 0.2, 3), 2, ops=("W",),
        )
