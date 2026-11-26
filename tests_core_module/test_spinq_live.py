"""SpinQ 真机冒烟；默认跳过，需 -m live、TYXONQ_SPINQ_LIVE=1 和凭证。

凭证来自 TYXONQ_SPINQ_USERNAME / TYXONQ_SPINQ_KEYFILE，或 .pypirc
[spinqcloud] 的 username/keyfile；TYXONQ_SPINQ_DEVICE 默认 gemini_vp，
TYXONQ_SPINQ_SHOTS 默认 16，TYXONQ_SPINQ_HOST 可选。凭证不会打印或写入文件。
"""

from __future__ import annotations

import importlib.util
import os

import pytest

from tyxonq import Circuit
from tyxonq.devices.hardware.spinq.driver import _resolve_credentials

pytestmark = pytest.mark.live


def _credentials_ready() -> bool:
    if os.getenv("TYXONQ_SPINQ_LIVE") != "1":
        return False
    if importlib.util.find_spec("spinqit") is None:
        return False
    try:
        _resolve_credentials({})
    except RuntimeError:
        return False
    return True


_READY = _credentials_ready()


def _ghz_circuit(num_qubits: int) -> Circuit:
    circuit = Circuit(num_qubits).h(0)
    for qubit in range(num_qubits - 1):
        circuit.cx(qubit, qubit + 1)
    for qubit in range(num_qubits):
        circuit.measure_z(qubit)
    return circuit


def _x_circuit(num_qubits: int) -> Circuit:
    circuit = Circuit(num_qubits)
    for qubit in range(num_qubits):
        circuit.x(qubit)
    for qubit in range(num_qubits):
        circuit.measure_z(qubit)
    return circuit


def _assert_x_counts(counts, num_qubits: int, shots: int) -> None:
    """20 比特联合读出允许噪声，但全 1 应为主峰且每个比特多数为 1。"""
    all_ones = "1" * num_qubits
    assert counts.get(all_ones, 0) > 0
    assert max(counts, key=counts.get) == all_ones
    for bit in range(num_qubits):
        ones = sum(count for bits, count in counts.items() if bits[bit] == "1")
        assert ones > shots // 2


@pytest.mark.skipif(not _READY, reason="需 TYXONQ_SPINQ_LIVE=1、SpinQit SDK 和有效凭证")
@pytest.mark.parametrize(
    ("device", "num_qubits", "kind"),
    [
        pytest.param("Triangulum-pro-1", 3, "ghz", id="triangulum-pro-1-3q"),
        # SQC-25 标称 25 比特；当前 coupling_map 暴露 20 个活跃比特。
        # 用确定性 X 冒烟覆盖全部活跃比特，避免深层 GHZ 的相干噪声干扰接入验证。
        pytest.param("SQC-25", 20, "x", id="sqc-25-active-20q"),
    ],
)
def test_spinq_real_hardware(device, num_qubits, kind):
    shots = int(os.getenv("TYXONQ_SPINQ_SHOTS", "32"))
    circuit = _ghz_circuit(num_qubits) if kind == "ghz" else _x_circuit(num_qubits)
    results = circuit.run(
        provider="spinq",
        device=device,
        shots=shots,
        host=os.getenv("TYXONQ_SPINQ_HOST"),
        wait_async_result=True,
    )
    assert isinstance(results, list) and len(results) == 1
    result = results[0]
    counts = result["result"]
    assert result["uni_status"] == "completed"
    counts_shots = sum(counts.values())
    assert counts_shots == result["result_meta"]["counts_shots"]
    # SpinQit 对仅返回概率的平台逐态 round(probability*shots)，可能产生 ±1 舍入差。
    assert abs(counts_shots - shots) <= 1
    assert all(len(bitstring) == num_qubits for bitstring in counts)
    if kind == "ghz":
        assert counts.get("0" * num_qubits, 0) + counts.get("1" * num_qubits, 0) > 0
    else:
        _assert_x_counts(counts, num_qubits, shots)
    assert result["result_meta"]["bit_order"] == "little_endian"
    assert len(result["result_meta"]["compile"]["cloud_measure_qubits"]) == num_qubits
    assert result["result_meta"]["task_id"]
    print(device, result["result_meta"]["task_id"], counts)
