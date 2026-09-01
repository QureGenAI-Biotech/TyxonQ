"""通过 TyxonQ 在 LQCloud 真机运行四比特门线路。

为避免误提交，默认先离线展示线路；显式增加 ``--run-online`` 后，
同一文件会使用 TyxonQ 的 ``lqcloud`` provider 提交并等待真机结果。
"""

from __future__ import annotations

import argparse
import getpass
import os
from collections import Counter
from typing import List, Optional

import tyxonq as tq


def build_circuit() -> tq.Circuit:
    """构造包含参数旋转和两比特纠缠的四比特线路。"""
    circuit = tq.Circuit(4)
    circuit.h(0).ry(1, 0.4).rz(2, -0.3).h(3)
    circuit.cx(0, 1).cx(1, 2).cz(2, 3)
    return circuit.add_measure(0, 1, 2, 3)


def parse_physical_qubits(value: Optional[str]) -> Optional[List[int]]:
    """把 ``0,1,2,3`` 转换成物理比特列表。"""
    if value is None:
        return None
    qubits = [int(item.strip()) for item in value.split(",") if item.strip()]
    if len(qubits) != 4:
        raise ValueError("--physical-qubits 必须提供 4 个物理比特，例如 0,1,2,3。")
    if any(q < 0 for q in qubits) or len(set(qubits)) != len(qubits):
        raise ValueError("物理比特必须是互不重复的非负整数。")
    return qubits


def instruction_summary(lqcloud_circuit) -> Counter:
    """统计官方线路对象中的指令名称。"""
    return Counter(str(item[0].name) for item in lqcloud_circuit.data)


def provider_metadata(result: dict) -> dict:
    """从 TyxonQ 统一结果中取出 LQCloud 原始元数据。"""
    outer = result.get("result_meta", {})
    if not isinstance(outer, dict):
        return {}
    inner = outer.get("result_meta", outer)
    return inner if isinstance(inner, dict) else {}


def main() -> None:
    parser = argparse.ArgumentParser(description="通过 TyxonQ 运行 LQCloud 四比特真机线路")
    parser.add_argument("--device", help="在线运行时使用的 LQCloud 设备名称")
    parser.add_argument("--shots", type=int, default=64)
    parser.add_argument(
        "--physical-qubits",
        help="逻辑比特 0..3 对应的物理比特，例如 0,1,2,3",
    )
    parser.add_argument(
        "--run-online",
        action="store_true",
        help="显式连接平台并提交一个任务",
    )
    args = parser.parse_args()

    physical_qubits = parse_physical_qubits(args.physical_qubits)
    circuit = build_circuit()

    # 转换是纯本地操作，用于先检查官方 SDK 最终收到的门。
    from tyxonq.devices.hardware.lqcloud.translator import to_lqcloud

    native_circuit = to_lqcloud(circuit)
    print("逻辑到物理映射:", physical_qubits or "由 LQCloud 自动选择")
    print("LQCloud 指令统计:", dict(instruction_summary(native_circuit)))

    if not args.run_online:
        print("离线模式完成：未登录、未查询设备、未提交任务。")
        return

    if not args.device:
        raise SystemExit("在线运行必须提供 --device。")
    if args.shots <= 0:
        raise SystemExit("--shots 必须大于 0。")

    # 优先使用环境变量；缺失时安全读取，不把 key 写进代码或终端历史。
    api_key = os.getenv("TYXONQ_LQCLOUD_API_KEY") or os.getenv("LQCLOUD_API_KEY")
    if not api_key:
        api_key = getpass.getpass("请输入 LQCloud API key: ")
    tq.set_token(api_key, provider="lqcloud")

    run_options = {}
    if physical_qubits is not None:
        run_options["physical_qubits"] = physical_qubits
    print(f"即将提交一个任务: device={args.device}, shots={args.shots}")
    results = circuit.run(
        provider="lqcloud",
        device=args.device,
        shots=args.shots,
        wait_async_result=True,
        **run_options,
    )
    result = results[0]
    metadata = provider_metadata(result)
    print("任务状态:", result.get("uni_status"))
    print("任务 ID:", metadata.get("job_id"))
    print("counts:", result.get("result", {}))
    print("总 shots:", sum(result.get("result", {}).values()))


if __name__ == "__main__":
    main()
