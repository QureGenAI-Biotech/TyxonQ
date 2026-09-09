"""在 LQCloud 真机上运行 TyxonQ 量子化学方法（HEA / UCCSD / SQD）的最小可用示例。

默认离线
--------
三个方法都跑在 TyxonQ 的 statevector 模拟器上（``provider="simulator"``）：
无需 token、不联网、不提交任务，只验证「分子/哈密顿量 → 线路 → 统一链式 .run() →
计数 → 能量」这条完整链路能跑通。

显式 ``--run-online``
--------------------
切到 ``provider="lqcloud"``：**同一套** ``.energy()`` / ``scan()`` 调用会把线路编译成
LQCloud 原生门再提交。其中 SQD 的 LUCJ 轨道旋转块是 fSim 家族的 2-qubit matchgate
（``two_mode_fock_matrix``），由 native compiler 的 ``decompose/matchgate`` 分解为
rz/rx/h/cx（fSim = iSWAP 型 + CPhase），这正是 SQD 能落到 LQCloud 真机的关键。

核心设计：simulator 与真机走**同一条统一链式 .run()**，仅 provider/device 不同，
调用方代码零分叉——这是 TyxonQ 的核心特色。

用法::

    python examples/run_chem_on_lqcloud.py                       # 离线，三方法跑模拟器
    python examples/run_chem_on_lqcloud.py --method hea          # 离线只跑 HEA
    python examples/run_chem_on_lqcloud.py --run-online --device MQ02 --shots 1024

真机 shots：本示例统一默认 1024，只求「最小可用」验证链路（HEA/UCCSD/SQD 都能落到
LQCloud 真机并取回计数）；若要化学精度可自行提高（如 UCCSD ≥ 2048，HEA ≥ 8192）。
"""

from __future__ import annotations

import argparse
import getpass
import importlib.util
import os
from typing import Optional

import tyxonq as tq


# H2 平衡键长（Angstrom）与 Angstrom->Bohr 换算；SQD 扫描坐标以 Bohr 传入。
_H2_GEOM_ANG = "H 0 0 0; H 0 0 0.74"
_BOHR_PER_ANG = 0.52917721092


def _has_pyscf() -> bool:
    """UCCSD / SQD 依赖 PySCF；缺失时优雅跳过而非崩溃。"""
    return importlib.util.find_spec("pyscf") is not None


def run_hea(provider: str, device: str, shots: int) -> float:
    """HEA 最小案例：2 比特、1 层，哈密顿量 H = Z0 + 0.5·X0X1 + 0.3·Y0Y1。

    HEA 不依赖 PySCF（哈密顿量直接以 Pauli 列表给出），故离线/真机都可跑。
    """
    from tyxonq.applications.chem.algorithms import HEA

    hamiltonian = [
        (1.0, [("Z", 0)]),
        (0.5, [("X", 0), ("X", 1)]),
        (0.3, [("Y", 0), ("Y", 1)]),
    ]
    hea = HEA(n_qubits=2, layers=1, hamiltonian=hamiltonian, runtime="device")
    return float(hea.energy(shots=shots, provider=provider, device=device))


def run_uccsd(provider: str, device: str, shots: int) -> float:
    """UCCSD 最小案例：预置 H2 分子（sto-3g），device 档经统一链式 .run() 取计数估能量。"""
    from tyxonq.applications.chem import molecule
    from tyxonq.applications.chem.algorithms.vqe.uccsd import UCCSD

    ucc = UCCSD(molecule.h2)
    return float(
        ucc.energy(shots=shots, runtime="device", provider=provider, device=device)
    )


def run_sqd(provider: str, device: str, shots: int) -> float:
    """SQD 最小案例：H2 活性空间 (2e,2o)，LUCJ 采样 1 层，frozen 子空间。

    SQD 的量子部分是「采样 LUCJ 线路得 counts」，故设备选项挂在 ``lucj_sampler`` 上
    （runtime/provider/device/shots），而非 solver_kwargs——这是 SQD 设备穿透的约定。
    """
    import numpy as np
    from pyscf import gto, scf

    from tyxonq.applications.chem.algorithms.sqd import lucj_sampler
    from tyxonq.applications.chem.interfaces import qc_scanner

    mol = gto.M(atom=_H2_GEOM_ANG, basis="sto-3g", unit="Angstrom", verbose=0)
    mf = scf.RHF(mol).run()

    sampler = lucj_sampler(
        mf,
        n_layers=1,
        shots=shots,
        runtime="device",
        provider=provider,
        device=device,
    )
    scan = qc_scanner(
        _H2_GEOM_ANG,
        basis="sto-3g",
        active_space=(2, 2),
        unit="Angstrom",
        method="sqd",
        sampler=sampler,
        subspace="frozen",
    )
    coords_bohr = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]) / _BOHR_PER_ANG
    energy, _gradient = scan(coords_bohr)
    return float(energy)


# 每个方法的入口 + 是否需要 PySCF + 默认 shots（统一 1024，只求最小可用验证链路）。
_METHODS = {
    "hea": (run_hea, False, 1024),
    "uccsd": (run_uccsd, True, 1024),
    "sqd": (run_sqd, True, 1024),
}


def resolve_token() -> str:
    """优先环境变量，缺失时安全读取；绝不把 key 写进代码或回显到终端历史。"""
    api_key = os.getenv("TYXONQ_LQCLOUD_API_KEY") or os.getenv("LQCLOUD_API_KEY")
    if not api_key:
        api_key = getpass.getpass("请输入 LQCloud API key: ")
    if not api_key:
        raise SystemExit("未提供 LQCloud API key，无法真机提交。")
    tq.set_token(api_key, provider="lqcloud")
    return api_key


def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(
        description="通过 TyxonQ 统一链式 .run() 在 LQCloud 上运行 HEA/UCCSD/SQD"
    )
    parser.add_argument(
        "--method",
        choices=["hea", "uccsd", "sqd", "all"],
        default="all",
        help="选择要运行的方法（默认全部）",
    )
    parser.add_argument("--device", help="真机运行时使用的 LQCloud 设备名称")
    parser.add_argument(
        "--shots", type=int, default=None, help="覆盖该方法的默认 shots（离线默认较小）"
    )
    parser.add_argument(
        "--run-online", action="store_true", help="显式连接 LQCloud 平台并提交真机任务"
    )
    args = parser.parse_args(argv)

    if args.run_online:
        if not args.device:
            raise SystemExit("真机运行必须提供 --device。")
        resolve_token()
        provider, device = "lqcloud", args.device
    else:
        provider, device = "simulator", "statevector"
        print("离线模式：跑在 statevector 模拟器上，未登录、未查询设备、未提交任务。")

    names = list(_METHODS) if args.method == "all" else [args.method]
    print(f"provider={provider}  device={device}\n")

    for name in names:
        runner, needs_pyscf, default_shots = _METHODS[name]
        shots = args.shots if args.shots is not None else default_shots
        if needs_pyscf and not _has_pyscf():
            print(f"[{name.upper()}] 跳过：未安装 PySCF。")
            continue
        try:
            energy = runner(provider, device, shots)
        except Exception as exc:  # noqa: BLE001
            print(f"[{name.upper()}] 运行失败：{type(exc).__name__}: {exc}")
            continue
        print(f"[{name.upper()}] shots={shots}  energy={energy:.8f} Ha")

    if not args.run_online:
        print("\n离线模式完成：加 --run-online --device <名称> 即把同一条链路基线路提交真机。")


if __name__ == "__main__":
    main()
