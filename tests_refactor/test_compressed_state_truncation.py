from __future__ import annotations

from tyxonq.core.ir import Circuit
from tyxonq.devices.simulators.compressed_state import CompressedStateEngine


def test_compressed_state_engine_max_bond_option_smoke():
    eng = CompressedStateEngine(max_bond=1)
    circ = Circuit(num_qubits=3, ops=[("h", 0), ("cx", 0, 1), ("cx", 1, 2), ("measure_z", 2)])
    out = eng.run(circ, shots=0)
    assert "expectations" in out and "metadata" in out


