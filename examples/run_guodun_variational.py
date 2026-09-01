"""国盾 5 比特参数化门线路示例。

默认只在本地编译并打印 QCIS 统计。只有显式传入 ``--run-online`` 时，
才会通过 TyxonQ ``Circuit.device(...).run(...)`` 提交一个任务并等待结果。
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from datetime import datetime

import tyxonq as tq


def build_variational_circuit() -> tq.Circuit:
    """构造包含两层旋转和链式纠缠的 5 比特线路。"""
    circuit = tq.Circuit(5)

    # 第一层：制备叠加态，并给每个逻辑比特施加不同的参数旋转。
    for qubit in range(5):
        circuit.h(qubit)
    for qubit, angle in enumerate((0.31, -0.47, 0.63, -0.79, 0.95)):
        circuit.ry(qubit, angle)

    # 两轮 RZZ 只连接相邻逻辑比特，便于映射到真机的一条可用耦合路径。
    for left, right, angle in (
        (0, 1, 0.37),
        (1, 2, -0.41),
        (2, 3, 0.53),
        (3, 4, -0.59),
    ):
        circuit.rzz(left, right, angle)

    for qubit, angle in enumerate((-0.22, 0.34, -0.46, 0.58, -0.70)):
        circuit.rx(qubit, angle)

    for left, right, angle in (
        (0, 1, -0.29),
        (1, 2, 0.43),
        (2, 3, -0.61),
        (3, 4, 0.73),
    ):
        circuit.rzz(left, right, angle)

    for qubit, angle in enumerate((0.11, -0.17, 0.23, -0.31, 0.41)):
        circuit.rz(qubit, angle)

    # 国盾编译器要求显式测量，不会自动补测量门。
    circuit.add_measure(0, 1, 2, 3, 4)
    return circuit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="国盾 5 比特参数化门线路示例")
    parser.add_argument("--device", default="gd_qc1")
    parser.add_argument("--shots", type=int, default=256)
    parser.add_argument(
        "--physical-qubits",
        required=True,
        help="五个逻辑比特对应的当前可用物理比特，例如 0,6,12,7,1",
    )
    parser.add_argument(
        "--run-online",
        action="store_true",
        help="提交一个任务并等待结果；省略时只做本地编译",
    )
    return parser.parse_args()


def _parse_mapping(raw: str) -> list[int]:
    """读取并校验五个逻辑比特的物理映射。"""
    mapping = [int(item.strip()) for item in raw.split(",")]
    if len(mapping) != 5:
        raise ValueError("--physical-qubits 必须恰好提供 5 个物理比特")
    if len(set(mapping)) != 5 or any(qubit < 0 for qubit in mapping):
        raise ValueError("物理比特必须非负且不能重复")
    return mapping


def _provider_meta(result: dict) -> dict:
    """从 TyxonQ 统一结果中取得国盾任务元数据。"""
    driver_info = result.get("result_meta", {})
    if not isinstance(driver_info, dict):
        return {}
    provider_meta = driver_info.get("result_meta", {})
    return provider_meta if isinstance(provider_meta, dict) else {}


def main() -> None:
    args = parse_args()
    physical_qubits = _parse_mapping(args.physical_qubits)
    circuit = build_variational_circuit()

    # 始终先离线编译；这一步不读取密钥，也不会连接平台。
    circuit.compile(
        compile_engine="guodun",
        output="qcis",
        options={"physical_qubits": physical_qubits, "optimization_level": 0},
    )
    qcis = circuit._compiled_source
    gate_stats = Counter(
        line.split()[0] for line in qcis.splitlines() if line.strip()
    )
    print(f"逻辑到物理映射: {dict(enumerate(physical_qubits))}")
    print(f"QCIS 指令统计: {dict(gate_stats)}")

    if not args.run_online:
        print("离线检查完成：未登录、未下载配置、未提交任务。")
        return

    token = os.getenv("TYXONQ_GUODUN_TOKEN")
    if not token:
        raise RuntimeError(
            "在线模式需要先设置 TYXONQ_GUODUN_TOKEN；不要把密钥写进源码。"
        )
    tq.set_token(token, provider="guodun", device=args.device)
    exp_name = f"tyxonq_guodun_tutorial_5q_{datetime.now():%Y%m%d_%H%M%S_%f}"
    print(
        f"提交一个任务：device={args.device}, shots={args.shots}, "
        f"exp_name={exp_name}"
    )

    # 使用 TyxonQ 高层接口；驱动会重新下载当前配置并检查全部 CZ 边。
    results = circuit.device(
        provider="guodun",
        device=args.device,
    ).run(
        shots=args.shots,
        physical_qubits=physical_qubits,
        wait_async_result=True,
        exp_name=exp_name,
    )

    for result in results:
        counts = result.get("result") or {}
        metadata = _provider_meta(result)
        print(f"query ID: {metadata.get('query_id')}")
        print(f"任务状态: {result.get('uni_status', 'unknown')}")
        print(
            "counts: "
            + json.dumps(dict(sorted(counts.items())), ensure_ascii=False)
        )
        print(f"总 shots: {sum(counts.values())}")
        error = result.get("error")
        if error:
            print(f"错误信息: {error}")


if __name__ == "__main__":
    main()
