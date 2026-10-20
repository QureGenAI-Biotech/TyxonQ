"""通过态矢量或矩阵乘积态模拟本地线路，并计算局域可观测量。"""

import numpy as np

from .models import PAULIS


def quimb_circuit(circuit, max_bond=32, cutoff=1e-10):
    """直接将 TyxonQ 基础门送入 quimb MPS，不依赖 qiskit-quimb。"""
    import quimb.tensor as qtn

    out = qtn.CircuitMPS(circuit.num_qubits, gate_opts={"max_bond": max_bond, "cutoff": cutoff})
    for op in circuit.ops:
        name = op[0]
        if name in ("rx", "ry", "rz"):
            out.apply_gate(name.upper(), float(op[2]), op[1])
        elif name == "cx":
            out.apply_gate("CNOT", op[1], op[2])
        elif name in ("h", "x", "s", "sdg"):
            out.apply_gate("SDG" if name == "sdg" else name.upper(), op[1])
        else:
            raise ValueError(f"Unsupported spin circuit operation: {name}")
    return out


def mps_state(circuit, max_bond=32, cutoff=1e-10):
    """保留 MPS 表示并显式归一化；不会展开态矢量。"""
    state = quimb_circuit(circuit, max_bond, cutoff).psi
    state.normalize()
    return state


def statevector(circuit):
    """使用 TyxonQ 原生态矢量引擎独立执行绑定线路。"""
    from ...devices.simulators.statevector.engine import StatevectorEngine

    supported = {"h", "x", "s", "sdg", "cx", "rx", "ry", "rz"}
    if any(op[0] not in supported for op in circuit.ops):
        raise ValueError("Spin simulator only accepts bound unitary base gates")
    state = np.asarray(StatevectorEngine(backend_name="numpy").state(circuit), dtype=complex)
    return state / np.linalg.norm(state)


def pauli_expectations(state, n, axis="Z"):
    """从一次计算得到的态矢量，批量计算所有位点的单自旋 Pauli 期望值。"""
    axis = axis.upper()
    if axis not in PAULIS:
        raise ValueError("axis must be X/Y/Z")
    psi = np.asarray(state).reshape((2,) * n)
    values = []
    for q in range(n):
        rows = np.moveaxis(psi, q, 0).reshape(2, -1)
        values.append(float(np.vdot(rows, PAULIS[axis] @ rows).real))
    return np.array(values)


def simulate_observables(circuit, axis="Z", simulator="statevector", max_bond=32, cutoff=1e-10):
    """返回所有位点期望值；MPS 路径直接收缩局域算符。"""
    axis = axis.upper()
    if axis not in PAULIS:
        raise ValueError("axis must be X/Y/Z")
    if simulator == "statevector":
        return pauli_expectations(statevector(circuit), circuit.num_qubits, axis)
    if simulator == "quimb_mps":
        psi = mps_state(circuit, max_bond, cutoff)
        terms = {(q,): PAULIS[axis] for q in range(circuit.num_qubits)}
        values = psi.compute_local_expectation(terms, return_all=True)
        return np.array([complex(values[(q,)]).real for q in range(circuit.num_qubits)])
    raise ValueError("simulator must be 'statevector' or 'quimb_mps'")
