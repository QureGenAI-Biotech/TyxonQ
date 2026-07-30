"""导入 RiverONE ``TQVQC`` checkpoint 并生成可执行的 QASM2。

直接运行本文件时使用下方“用户配置区”。命令行参数仍可临时覆盖这些配置。
默认在本地模拟器执行；选择真机设备时才访问服务器。
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import getpass
import os
from pathlib import Path

import numpy as np

import tyxonq as tq
from tyxonq.applications.qml import (
    load_riverone_vqc,
    riverone_to_qasm2,
)


# ============================== 用户配置区 ==============================
CHECKPOINT: Path | None = None  # 可直接填写 checkpoint 路径。
VQC_INDEX = 0
AMPLITUDES: Path | None = None  # 可填一维 .npy；None 表示使用演示振幅。
SHOTS = 1024
DEVICE = "simulator"  # 可选 "simulator"、"homebrew_s2"、"homebrew_s3"。
TYXONQ_API_KEY = ""
# =======================================================================


_BASES = ("X", "Y", "Z")
_DEVICES = ("simulator", "homebrew_s2", "homebrew_s3")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="导入 RiverONE TQVQC，并生成可模拟或提交的三份 QASM2。"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=CHECKPOINT,
        help="RiverONE checkpoint；也可在程序开头的 CHECKPOINT 中设置。",
    )
    parser.add_argument(
        "--vqc-index",
        type=int,
        default=VQC_INDEX,
        help="选择第几条 VQC；默认使用程序开头的 VQC_INDEX。",
    )
    parser.add_argument(
        "--amplitudes",
        type=Path,
        default=AMPLITUDES,
        help="一维 .npy 振幅；默认使用程序开头的 AMPLITUDES。",
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=SHOTS,
        help="每个测量基的 shots；默认使用程序开头的 SHOTS。",
    )
    parser.add_argument(
        "--device",
        choices=_DEVICES,
        default=DEVICE,
        help="执行设备；默认使用本地 simulator。",
    )
    args = parser.parse_args(argv)
    if args.checkpoint is None:
        parser.error("请在用户配置区设置 CHECKPOINT，或使用 --checkpoint 指定。")
    if args.device not in _DEVICES:
        parser.error(f"设备必须是 {_DEVICES} 之一。")
    if args.vqc_index < 0:
        parser.error("--vqc-index 不能为负数。")
    if args.shots < 1:
        parser.error("--shots 必须大于 0。")
    if args.amplitudes is not None and args.amplitudes.suffix.lower() != ".npy":
        parser.error("--amplitudes 必须是 .npy 文件。")
    return args


def _task_id(task: object) -> str:
    handle = getattr(task, "handle", None)
    value = getattr(handle, "id", None) or getattr(task, "id", None)
    return str(value) if value is not None else "unknown"


def _load_amplitudes(path: Path | None, n_qubits: int) -> np.ndarray:
    if path is None:
        print("未提供 --amplitudes，使用 [1, 2, ..., 2^n] 确定性演示输入。")
        return np.arange(1, (1 << n_qubits) + 1, dtype=np.float64)
    return np.load(path, allow_pickle=False)


def _print_result(basis: str, values: Sequence[float]) -> None:
    items = ", ".join(f"{basis}{wire}={value:.8f}" for wire, value in enumerate(values))
    print(f"{basis}: {items}")


def _expectations_from_counts(
    counts: Mapping[str, int], n_qubits: int
) -> tuple[float, ...]:
    """按 TyxonQ 的 q0-first 位序从采样 counts 计算单比特期望值。"""

    samples: list[tuple[str, int]] = []
    for raw_bits, raw_count in counts.items():
        bits = str(raw_bits).replace(" ", "")
        if len(bits) != n_qubits or any(bit not in "01" for bit in bits):
            raise ValueError(f"模拟器返回了非法测量态：{raw_bits!r}")
        samples.append((bits, int(raw_count)))

    shots = sum(count for _, count in samples)
    if shots <= 0:
        raise ValueError("模拟器没有返回有效 shots。")

    return tuple(
        sum((1 if bits[wire] == "0" else -1) * count for bits, count in samples)
        / shots
        for wire in range(n_qubits)
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    spec = load_riverone_vqc(args.checkpoint, vqc_index=args.vqc_index)
    amplitudes = _load_amplitudes(args.amplitudes, spec.n_qubits)

    qasm_by_basis = riverone_to_qasm2(spec, amplitudes)
    print(
        f"已生成 RiverONE VQC {args.vqc_index} 的三份 QASM2："
        f"{spec.n_qubits} qubits。"
    )

    if args.device == "simulator":
        # 模拟器当前逐条接收 QASM，避免触发批量 QASM 的解析限制。
        results_by_basis: dict[str, tuple[float, ...]] = {}
        for basis in _BASES:
            tasks = tq.api.submit_task(
                provider="simulator",
                device="statevector",
                source=qasm_by_basis[basis],
                shots=args.shots,
            )
            task_list = tasks if isinstance(tasks, list) else [tasks]
            if len(task_list) != 1:
                raise RuntimeError(
                    f"{basis} 基本地模拟预期返回 1 个任务，实际为 {len(task_list)} 个。"
                )

            task = task_list[0]
            details = tq.api.get_task_details(task)
            if details.get("error"):
                raise RuntimeError(f"{basis} 基本地模拟失败：{details['error']}")
            counts = details.get("result") or {}
            results_by_basis[basis] = _expectations_from_counts(
                counts, spec.n_qubits
            )

        print(
            "X/Y/Z 本地模拟完成："
            f"device=statevector，每个测量基 shots={args.shots}"
        )
        for basis in _BASES:
            _print_result(basis, results_by_basis[basis])
        return 0

    token = (
        TYXONQ_API_KEY
        or os.getenv("TYXONQ_API_KEY")
        or getpass.getpass("请输入 TYXONQ_API_KEY：")
    )
    if not token:
        raise SystemExit("未提供 TYXONQ_API_KEY，任务未提交。")

    tq.set_token(token, provider="tyxonq", device=args.device)
    tasks = tq.api.submit_task(
        provider="tyxonq",
        device=args.device,
        source=[qasm_by_basis[basis] for basis in _BASES],
        shots=args.shots,
    )
    task_list = tasks if isinstance(tasks, list) else [tasks]
    if len(task_list) != 1:
        raise RuntimeError(
            f"预期返回 1 个批量任务，实际为 {len(task_list)} 个。"
        )
    print(
        f"X/Y/Z 批量任务已提交：task_id={_task_id(task_list[0])}，"
        f"device={args.device}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
