"""国盾门线路最小示例。

默认只在本地生成并打印 QCIS，不登录国盾平台。只有显式传入
``--run-online`` 时才提交；提交成功后立即打印 query ID 并退出，不查询结果。
"""

from __future__ import annotations

import argparse
import os
from collections import Counter
from datetime import datetime

import tyxonq as tq
from tyxonq.devices import base as device_base


def build_bell_circuit() -> tq.Circuit:
    """构造 Bell 线路，并显式测量两个逻辑比特。"""
    circuit = tq.Circuit(2)
    circuit.h(0).cx(0, 1).add_measure(0, 1)
    return circuit


def build_x_circuit() -> tq.Circuit:
    """构造首次在线权限探测使用的最小 X + M 线路。"""
    circuit = tq.Circuit(1)
    circuit.x(0).add_measure(0)
    return circuit


def count_qcis_instructions(qcis: str) -> dict[str, int]:
    """按 QCIS 每行的第一个 token 统计指令数。"""
    counts = Counter(
        line.split()[0]
        for line in qcis.splitlines()
        if line.strip()
    )
    return dict(counts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="国盾门线路本地 QCIS 示例")
    parser.add_argument("--device", default="gd_test")
    parser.add_argument("--shots", type=int, default=100)
    parser.add_argument(
        "--exp-name",
        default=None,
        help="实验名称；省略时自动生成带微秒时间戳的唯一名称",
    )
    parser.add_argument(
        "--circuit",
        choices=("bell", "x"),
        default="bell",
        help="默认离线展示 Bell；首次在线探测应显式选择 x",
    )
    parser.add_argument(
        "--physical-qubits",
        default=None,
        help="逻辑比特对应的物理比特，例如 60,55；省略时按线路宽度从 0 开始映射",
    )
    parser.add_argument(
        "--run-online",
        action="store_true",
        help="只提交一次并打印 query ID；不会查询实验结果",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    circuit = build_x_circuit() if args.circuit == "x" else build_bell_circuit()
    physical_qubits = (
        [int(item.strip()) for item in args.physical_qubits.split(",")]
        if args.physical_qubits is not None
        else list(range(circuit.num_qubits))
    )

    # 第一段始终是纯本地编译，不读取 SDK 密钥，也不创建平台对象。
    circuit.compile(
        compile_engine="guodun",
        output="qcis",
        options={"physical_qubits": physical_qubits},
    )
    qcis = circuit._compiled_source
    print(f"逻辑到物理映射: {dict(enumerate(physical_qubits))}")
    print(f"QCIS 指令统计: {count_qcis_instructions(qcis)}")
    print("生成的 QCIS:")
    print(qcis)

    if not args.run_online:
        print("离线模式完成：未登录、未下载配置、未提交任务。")
        return

    token = os.getenv("TYXONQ_GUODUN_TOKEN")
    if not token:
        raise RuntimeError(
            "在线模式需要先设置 TYXONQ_GUODUN_TOKEN；不要把密钥写进源码。"
        )
    tq.set_token(token, provider="guodun", device=args.device)
    exp_name = args.exp_name or (
        f"tyxonq_{args.device}_{args.circuit}_{datetime.now():%Y%m%d_%H%M%S_%f}"
    )
    if not exp_name.strip():
        raise ValueError("exp_name 不能为空")
    print(
        f"即将在线提交：device={args.device}, circuits=1, shots={args.shots}, "
        f"exp_name={exp_name}。"
    )
    # 直接调用设备层，只取得任务句柄；不要在这里调用 get_result()。
    tasks = device_base.run(
        provider="guodun",
        device=args.device,
        source=qcis,
        shots=args.shots,
        exp_name=exp_name,
    )
    query_ids = [task.handle.id for task in tasks]
    print(f"提交成功，query ID: {query_ids}", flush=True)
    print("提交后已停止：未查询结果、未创建第二个任务。", flush=True)


if __name__ == "__main__":
    main()
