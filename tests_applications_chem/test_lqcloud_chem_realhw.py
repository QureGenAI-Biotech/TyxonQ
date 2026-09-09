"""LQCloud 真机上的量子化学方法（HEA / UCCSD / SQD）最小可用测试。

三层测试策略（与 ``examples/run_chem_on_lqcloud.py`` 共用同一条统一链式 ``.run()``）：

1. **simulator 冒烟**（总是跑，无需 token/SDK/联网）：三个方法都跑在
   ``provider="simulator", device="statevector"`` 上，验证「分子/哈密顿量 → 线路 →
   统一链式 .run() → 计数 → 能量」链路本身正确。

2. **lqcloud 离线降门门**（总是跑，用 fake SDK 替身，无需 token/不提交/不联网）：
   在 ``Circuit.compile`` 处截获 chem 方法真正要提交的线路，用 ``to_lqcloud`` 直译成
   lqcloud 原生门，断言关键门就位——UCCSD 的 ``cry``（曾缺失、后解阻）、SQD 的
   fSim 家族 2-qubit matchgate（``two_mode_fock_matrix``）被分解为 ``cx/rz/rx/h``
   而非落成 ``su2``。这是「能上真机」的离线充分条件。

3. **lqcloud 真机提交**（skipif 无 token / 无 SDK / 无 device；显式 opt-in）：
   ``provider="lqcloud"`` + 真机设备名，实提交并断言能量有限。**会消耗云资源、
   可能阻塞在队列**，故仅在环境变量齐备时才跑。

本文件属 ``tests_applications_chem``：UCCSD/SQD 依赖 PySCF，**不带** KMP 标志。
token 只从环境变量读取、绝不回显、绝不写入任何持久化位置。
"""

from __future__ import annotations

import importlib.util
import math
import os
import sys
from types import SimpleNamespace

import pytest


# --------------------------------------------------------------------------- #
# 环境探测（全部只读；token 绝不回显、绝不持久化）
# --------------------------------------------------------------------------- #
def _has_pyscf() -> bool:
    """UCCSD / SQD 依赖 PySCF；缺失时优雅跳过而非崩溃。"""
    return importlib.util.find_spec("pyscf") is not None


def _has_lqcloud_sdk() -> bool:
    """真机提交依赖官方 lqcloud SDK；离线降门门用 fake 替身，不需要它。"""
    return importlib.util.find_spec("lqcloud") is not None


def _lqcloud_token() -> str | None:
    """从环境变量读取 LQCloud token（只读、绝不回显、绝不写入记忆/文件）。"""
    return os.getenv("TYXONQ_LQCLOUD_API_KEY") or os.getenv("LQCLOUD_API_KEY") or None


def _lqcloud_device() -> str | None:
    """真机设备名（如 MQ02）；未设则跳过真机提交测试。"""
    return os.getenv("TYXONQ_LQCLOUD_DEVICE") or None


def _lqcloud_shots(default: int) -> int:
    """真机 shots（可经 TYXONQ_LQCLOUD_SHOTS 覆盖）；默认取「最小可用」量级。"""
    raw = os.getenv("TYXONQ_LQCLOUD_SHOTS")
    return int(raw) if raw else default


needs_pyscf = pytest.mark.skipif(
    not _has_pyscf(), reason="PySCF not installed; skipping chem UCCSD/SQD test."
)

# 真机提交的跳过原因在模块加载时算一次（token/device/SDK 齐备才跑）。
_REALHW_SKIP: str | None = None
if not _has_lqcloud_sdk():
    _REALHW_SKIP = "lqcloud extra not installed"
elif not _lqcloud_token():
    _REALHW_SKIP = "no LQCloud token in env (TYXONQ_LQCLOUD_API_KEY / LQCLOUD_API_KEY)"
elif not _lqcloud_device():
    _REALHW_SKIP = "no LQCloud device in env (TYXONQ_LQCLOUD_DEVICE)"
needs_realhw = pytest.mark.skipif(
    _REALHW_SKIP is not None, reason=_REALHW_SKIP or "LQCloud real hardware available"
)


# --------------------------------------------------------------------------- #
# fake lqcloud SDK 替身：只暴露 .data，供离线降门门检视门名（不联网、不提交）。
# 与 tests_core_module/test_lqcloud_lowering.py 的替身同构。
# --------------------------------------------------------------------------- #
class _FakeGate:
    def __init__(self, name):
        self.name = name


class _FakeQubit:
    def __init__(self, index):
        self.index = index


class _FakeQuantumCircuit:
    def __init__(self, num_qubits, num_clbits):
        self.num_qubits = num_qubits
        self.num_clbits = num_clbits
        self.qubits = [_FakeQubit(i) for i in range(num_qubits)]
        self.data = []

    def _append(self, name, *qubits):
        self.data.append((_FakeGate(name), [self.qubits[q] for q in qubits], []))

    def h(self, q): self._append("h", q)
    def x(self, q): self._append("x", q)
    def y(self, q): self._append("y", q)
    def z(self, q): self._append("z", q)
    def s(self, q): self._append("s", q)
    def sdg(self, q): self._append("sdg", q)
    def t(self, q): self._append("t", q)
    def tdg(self, q): self._append("tdg", q)
    def reset(self, q): self._append("reset", q)
    def rx(self, theta, q): self._append("rx", q)
    def ry(self, theta, q): self._append("ry", q)
    def rz(self, theta, q): self._append("rz", q)
    def cx(self, q0, q1): self._append("cx", q0, q1)
    def cy(self, q0, q1): self._append("cy", q0, q1)
    def cz(self, q0, q1): self._append("cz", q0, q1)
    def swap(self, q0, q1): self._append("swap", q0, q1)
    def iswap(self, q0, q1): self._append("iswap", q0, q1)
    def crx(self, theta, q0, q1): self._append("crx", q0, q1)
    def cry(self, theta, q0, q1): self._append("cry", q0, q1)
    def crz(self, theta, q0, q1): self._append("crz", q0, q1)
    def su2(self, matrix, q): self._append("su2", q)
    def barrier(self, *qubits): self._append("barrier", *qubits)
    def measure(self, qubit, clbit): self._append("measure", qubit)


def _names(qc) -> list:
    """降门后 lqcloud 线路的门名序列。"""
    return [item[0].name for item in qc.data]


# --------------------------------------------------------------------------- #
# chem 方法构建器：与 examples/run_chem_on_lqcloud.py 逐行对齐（simulator/真机同一条链）。
# --------------------------------------------------------------------------- #
_H2_GEOM_ANG = "H 0 0 0; H 0 0 0.74"
_BOHR_PER_ANG = 0.52917721092


def _hea_energy(provider: str, device: str, shots: int) -> float:
    """HEA 最小案例：2 比特、1 层，H = Z0 + 0.5·X0X1 + 0.3·Y0Y1（不依赖 PySCF）。"""
    from tyxonq.applications.chem.algorithms import HEA

    hamiltonian = [
        (1.0, [("Z", 0)]),
        (0.5, [("X", 0), ("X", 1)]),
        (0.3, [("Y", 0), ("Y", 1)]),
    ]
    hea = HEA(n_qubits=2, layers=1, hamiltonian=hamiltonian, runtime="device")
    return float(hea.energy(shots=shots, provider=provider, device=device))


def _uccsd_energy(provider: str, device: str, shots: int) -> float:
    """UCCSD 最小案例：预置 H2（sto-3g），device 档经统一链式 .run() 取计数估能量。"""
    from tyxonq.applications.chem import molecule
    from tyxonq.applications.chem.algorithms.vqe.uccsd import UCCSD

    ucc = UCCSD(molecule.h2)
    return float(ucc.energy(shots=shots, runtime="device", provider=provider, device=device))


def _sqd_energy(provider: str, device: str, shots: int) -> float:
    """SQD 最小案例：H2 活性空间 (2e,2o)，LUCJ 采样 1 层，frozen 子空间。

    设备选项挂在 ``lucj_sampler`` 上（runtime/provider/device/shots），而非
    solver_kwargs——这是 SQD 设备穿透的约定（量子部分是「采样 LUCJ 线路得 counts」）。
    """
    import numpy as np
    from pyscf import gto, scf

    from tyxonq.applications.chem.algorithms.sqd import lucj_sampler
    from tyxonq.applications.chem.interfaces import qc_scanner

    mol = gto.M(atom=_H2_GEOM_ANG, basis="sto-3g", unit="Angstrom", verbose=0)
    mf = scf.RHF(mol).run()
    sampler = lucj_sampler(
        mf, n_layers=1, shots=shots, runtime="device", provider=provider, device=device
    )
    scan = qc_scanner(
        _H2_GEOM_ANG, basis="sto-3g", active_space=(2, 2), unit="Angstrom",
        method="sqd", sampler=sampler, subspace="frozen",
    )
    coords_bohr = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]) / _BOHR_PER_ANG
    energy, _gradient = scan(coords_bohr)
    return float(energy)


# --------------------------------------------------------------------------- #
# 离线降门门：截获 chem 方法首次要提交的线路，用 fake SDK 直译成 lqcloud 原生门。
# --------------------------------------------------------------------------- #
class _Captured(Exception):
    """哨兵异常：捕获到首条待编译线路后立即中断 chem 运行（不提交、不联网）。"""


def _lower_first_circuit(build_energy_fn, monkeypatch) -> list:
    """跑 chem 方法，在首次 ``Circuit.compile`` 处截获线路，再走 lqcloud 委托编译离线降门。

    返回降门后的门名序列。全程不联网、不提交、不需要 token——只验证「chem 线路能被
    lqcloud 编译成原生门」，这是能上真机的离线充分条件。

    关键：走 ``compile(platform="lqcloud")`` **委托路径**（先由 gate_compiler 核心分解
    matchgate/rotations，再 to_lqcloud 直译），而非裸调 ``to_lqcloud``——后者只翻译、
    不分解，SQD 的 2-qubit matchgate 会因未经核心分解而 raise（与真机 .run() 路径一致）。
    """
    from tyxonq.compiler.api import compile as compile_api
    from tyxonq.core.ir import Circuit

    # fake SDK：让委托末端的 to_lqcloud 落到可检视 .data 的替身（无需真 SDK，也不联网）。
    monkeypatch.setitem(
        sys.modules, "lqcloud", SimpleNamespace(QuantumCircuit=_FakeQuantumCircuit)
    )

    captured: list = []

    def _fake_compile(self, *args, **kwargs):
        captured.append(self)
        raise _Captured()

    # 在类对象上打补丁：所有 Circuit 实例（含 chem 内部构建的）都会命中。
    monkeypatch.setattr(Circuit, "compile", _fake_compile)

    try:
        # provider/device 无所谓（compile 会在提交前中断）；shots>0 满足 device 档要求。
        build_energy_fn("simulator", "statevector", 1024)
    except _Captured:
        pass

    assert captured, "chem 方法未经 Circuit.compile（无法离线降门校验）"
    # 走委托编译：核心先分解 matchgate/rotations，再直译为 lqcloud 对象（fake SDK 替身）。
    result = compile_api(captured[0], platform="lqcloud", output="lqcloud")
    return _names(result["compiled_source"])


# =========================================================================== #
# 第 1 层：simulator 冒烟（总是跑；验证统一链式 .run() 链路本身）
# =========================================================================== #
def test_hea_simulator_energy():
    """HEA 在 statevector 模拟器上取计数估能量：有限浮点即通过（不依赖 PySCF）。"""
    e = _hea_energy("simulator", "statevector", shots=1024)
    assert isinstance(e, float) and math.isfinite(e)


@needs_pyscf
def test_uccsd_simulator_energy():
    """UCCSD/H2 在 statevector 模拟器上估能量：应落在化学合理区间 (-3, 0) Ha。"""
    e = _uccsd_energy("simulator", "statevector", shots=2048)
    assert isinstance(e, float) and math.isfinite(e)
    assert -3.0 < e < 0.0


@needs_pyscf
def test_sqd_simulator_energy():
    """SQD/H2 (2e,2o) 在 statevector 模拟器上采样估能量：近 FCI，落 (-3, 0) Ha。"""
    e = _sqd_energy("simulator", "statevector", shots=4096)
    assert isinstance(e, float) and math.isfinite(e)
    assert -3.0 < e < 0.0


# =========================================================================== #
# 第 2 层：lqcloud 离线降门门（总是跑；fake 替身，无需 token/不提交）
# =========================================================================== #
def test_hea_lowers_to_lqcloud(monkeypatch):
    """HEA 线路能直译成 lqcloud 原生门：含 ry/cx 与终端 measure，无未支持门报错。"""
    names = _lower_first_circuit(_hea_energy, monkeypatch)
    assert "measure" in names, "缺少终端测量（lqcloud 要求每次测量前 barrier + measure）"
    assert "ry" in names and "cx" in names, "HEA ansatz 的 ry/cx 未就位"


@needs_pyscf
def test_uccsd_lowers_to_lqcloud(monkeypatch):
    """UCCSD 线路能直译：关键是 ``cry``（SDK 原生受控旋转）就位——曾缺失导致无法降门。"""
    names = _lower_first_circuit(_uccsd_energy, monkeypatch)
    assert "measure" in names
    assert "cry" in names, "UCCSD 依赖 cry；缺失说明受控旋转映射回归"


@needs_pyscf
def test_sqd_lowers_to_lqcloud(monkeypatch):
    """SQD/LUCJ 线路能直译：fSim 家族 2-qubit matchgate 被分解为 cx/rz/rx/h，不落 su2。

    LUCJ 的轨道旋转块是 ``two_mode_fock_matrix``（4x4 数守恒 matchgate）。SDK 的 su2
    只吃单比特 2x2，故 2-qubit 块必须按 fSim = iSWAP 型 + CPhase 分解为原生门；
    能直译成功（不 raise）本身即证明分解触发，再断言分解产物 cx/rz/rx/h 就位。
    """
    names = _lower_first_circuit(_sqd_energy, monkeypatch)
    assert "measure" in names
    for gate in ("cx", "rz", "rx", "h"):
        assert gate in names, f"matchgate 分解产物 {gate} 缺失"
    assert "su2" not in names, "2-qubit matchgate 不应落成单比特 su2"


# =========================================================================== #
# 第 3 层：lqcloud 真机提交（skipif 无 token/SDK/device；会消耗云资源、可能阻塞队列）
# =========================================================================== #
def _submit_realhw(build_energy_fn, default_shots: int) -> float:
    """用环境变量里的 token/device 把 chem 线路实提交到 LQCloud 真机，返回能量。

    token 只从环境变量读取并交给 ``tq.set_token``，绝不回显、绝不写入持久化位置。
    """
    import tyxonq as tq

    tq.set_token(_lqcloud_token(), provider="lqcloud")
    device = _lqcloud_device()
    shots = _lqcloud_shots(default_shots)
    return build_energy_fn("lqcloud", device, shots)


@needs_realhw
def test_hea_lqcloud_realhw():
    """HEA 实提交 LQCloud 真机：能量为有限浮点（可能阻塞在队列，opt-in 才跑）。"""
    e = _submit_realhw(_hea_energy, default_shots=1024)
    assert isinstance(e, float) and math.isfinite(e)


@needs_realhw
@needs_pyscf
def test_uccsd_lqcloud_realhw():
    """UCCSD/H2 实提交 LQCloud 真机：能量落化学合理区间 (-3, 0) Ha。"""
    e = _submit_realhw(_uccsd_energy, default_shots=2048)
    assert isinstance(e, float) and math.isfinite(e)
    assert -3.0 < e < 0.0


@needs_realhw
@needs_pyscf
def test_sqd_lqcloud_realhw():
    """SQD/H2 (2e,2o) 实提交 LQCloud 真机：LUCJ 采样能量落 (-3, 0) Ha。"""
    e = _submit_realhw(_sqd_energy, default_shots=4096)
    assert isinstance(e, float) and math.isfinite(e)
    assert -3.0 < e < 0.0
