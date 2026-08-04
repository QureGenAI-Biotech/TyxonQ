from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest
import torch

from tyxonq.applications.qml import (
    RiverONEVQCSpec,
    load_riverone_vqc,
    riverone_to_qasm2,
)


def _checkpoint_state(
    *, n_qubits: int = 2, n_layers: int = 1, vqc_index: int = 1
) -> dict[str, object]:
    state: dict[str, object] = {}
    value = 0.1
    for layer in range(n_layers):
        for wire in range(n_qubits):
            for gate in ("rx", "ry", "rz"):
                state[
                    f"vqcs.{vqc_index}.variational."
                    f"l{layer}_w{wire}.{gate}.params"
                ] = value
                value += 0.1
    return state


def _write_checkpoint(path: Path, state: dict[str, object]) -> Path:
    torch.save({"state": state}, path)
    return path


def _zero_spec(n_qubits: int = 2, n_layers: int = 1) -> RiverONEVQCSpec:
    return RiverONEVQCSpec(
        n_qubits=n_qubits,
        n_layers=n_layers,
        angles=[[[0.0, 0.0, 0.0] for _ in range(n_qubits)] for _ in range(n_layers)],
    )


def test_spec_rejects_non_finite_angles():
    with pytest.raises(ValueError, match="非有限角度"):
        RiverONEVQCSpec(
            n_qubits=2,
            n_layers=1,
            angles=[[[0.1, 0.2, float("nan")], [0.4, 0.5, 0.6]]],
        )


def test_load_riverone_vqc_directly_without_torchquantum(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    checkpoint = _write_checkpoint(tmp_path / "checkpoint.pt", _checkpoint_state())
    monkeypatch.setitem(sys.modules, "torchquantum", None)

    spec = load_riverone_vqc(checkpoint, vqc_index=1)

    assert spec.n_qubits == 2
    assert spec.n_layers == 1
    assert spec.reupload_every == 2
    actual = [value for layer in spec.angles for wire in layer for value in wire]
    assert actual == pytest.approx([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])


def test_load_riverone_vqc_requires_state_mapping(tmp_path: Path):
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({}, checkpoint)

    with pytest.raises(ValueError, match="必须包含字典字段 'state'"):
        load_riverone_vqc(checkpoint)


def test_load_riverone_vqc_rejects_unknown_index(tmp_path: Path):
    checkpoint = _write_checkpoint(
        tmp_path / "checkpoint.pt", _checkpoint_state(vqc_index=0)
    )

    with pytest.raises(IndexError, match="不存在 vqc_index=1"):
        load_riverone_vqc(checkpoint, vqc_index=1)


def test_load_riverone_vqc_rejects_missing_gate(tmp_path: Path):
    state = _checkpoint_state(vqc_index=0)
    del state["vqcs.0.variational.l0_w1.rz.params"]
    checkpoint = _write_checkpoint(tmp_path / "checkpoint.pt", state)

    with pytest.raises(ValueError, match="缺少 VQC 0 的 l0_w1.rz 参数"):
        load_riverone_vqc(checkpoint)


def test_load_riverone_vqc_rejects_non_scalar_parameter(tmp_path: Path):
    state = _checkpoint_state(vqc_index=0)
    state["vqcs.0.variational.l0_w0.rx.params"] = torch.tensor([0.1, 0.2])
    checkpoint = _write_checkpoint(tmp_path / "checkpoint.pt", state)

    with pytest.raises(ValueError, match="必须恰好包含一个角度"):
        load_riverone_vqc(checkpoint)


@pytest.mark.parametrize(
    ("amplitudes", "message"),
    [
        (np.array([], dtype=float), "不能为空"),
        (np.zeros(4), "全零"),
        (np.ones((2, 2)), "一维"),
        (np.ones(5), "最多包含 4"),
        (np.array([1.0, np.nan]), "非有限"),
    ],
)
def test_qasm2_rejects_invalid_amplitudes(amplitudes, message):
    with pytest.raises(ValueError, match=message):
        riverone_to_qasm2(_zero_spec(), amplitudes)


def test_reupload_discards_layers_before_last_state_overwrite():
    final_layer = [[0.2, -0.1, 0.3], [0.4, 0.5, -0.2]]
    zero_layer = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    changed_layer = [[1.2, -0.7, 0.8], [-0.4, 1.1, 0.9]]
    reference = RiverONEVQCSpec(
        n_qubits=2,
        n_layers=3,
        angles=[zero_layer, zero_layer, final_layer],
    )
    changed = RiverONEVQCSpec(
        n_qubits=2,
        n_layers=3,
        angles=[changed_layer, changed_layer, final_layer],
    )
    amplitudes = np.array([1.0, 2.0, 3.0, 4.0])

    assert riverone_to_qasm2(changed, amplitudes) == riverone_to_qasm2(
        reference, amplitudes
    )


def test_qasm2_xyz_has_expected_measurement_semantics():
    qiskit = pytest.importorskip("qiskit")
    from qiskit.quantum_info import Pauli, Statevector

    qasm_by_basis = riverone_to_qasm2(_zero_spec(), [2.0])
    expected = {"X": (0.0, 0.0), "Y": (0.0, 0.0), "Z": (1.0, 1.0)}

    assert tuple(qasm_by_basis) == ("X", "Y", "Z")
    for basis, source in qasm_by_basis.items():
        parsed = qiskit.qasm2.loads(source)
        assert parsed.num_qubits == 2
        assert parsed.num_clbits == 2
        assert parsed.count_ops().get("measure") == 2
        assert set(parsed.count_ops()) <= {
            "rx",
            "rz",
            "h",
            "cx",
            "cz",
            "measure",
            "barrier",
        }

        state = Statevector.from_instruction(
            parsed.remove_final_measurements(inplace=False)
        )
        actual = []
        for wire in range(2):
            pauli = ["I", "I"]
            pauli[1 - wire] = "Z"
            actual.append(float(np.real(state.expectation_value(Pauli("".join(pauli))))))
        assert actual == pytest.approx(expected[basis], abs=1e-7)


def test_real_eight_qubit_qasm2_is_parseable():
    qiskit = pytest.importorskip("qiskit")
    qasm_by_basis = riverone_to_qasm2(
        _zero_spec(n_qubits=8, n_layers=6),
        np.arange(1, 257, dtype=float),
    )

    for source in qasm_by_basis.values():
        circuit = qiskit.qasm2.loads(source)
        assert circuit.num_qubits == 8
        assert circuit.num_clbits == 8
        assert circuit.count_ops().get("measure") == 8
        assert "state_preparation" not in circuit.count_ops()
        assert "initialize" not in circuit.count_ops()
