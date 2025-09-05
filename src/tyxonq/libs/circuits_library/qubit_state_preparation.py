from typing import List, Tuple, Optional
import numpy as np
import tyxonq as tq

from tyxonq.applications.chem.chem_libs.quantum_chem_library.ci_state_mapping import get_ci_strings, civector_to_statevector
from .utils import unpack_nelec


def get_init_circuit(
    n_qubits: int,
    n_elec_s,
    mode: str,
    init_state=None,
    givens_swap: bool = False,
    *,
    runtime: str = "numeric",
):
    """路由函数：根据 runtime 分流到设备/数值初始化。

    - runtime=="device": 返回门级 tq.Circuit（调用 get_device_init_circuit）
    - runtime=="numeric": 返回 numpy statevector（调用 get_numeric_init_circuit）
    """
    if runtime == "device":
        return get_device_init_circuit(
            n_qubits,
            n_elec_s,
            mode,
            givens_swap=givens_swap,
            init_circuit=init_state if isinstance(init_state, tq.Circuit) else None,
        )
    if runtime == "numeric":
        civector = init_state if (init_state is not None and not isinstance(init_state, tq.Circuit)) else None
        return get_numeric_init_circuit(
            n_qubits,
            n_elec_s,
            mode,
            civector=civector,
            givens_swap=givens_swap,
        )
    raise ValueError(f"Unsupported runtime: {runtime}")


#TODO 真正的大工程 线路继承 根据上一轮测量结果反馈新增线路门 
def get_device_init_circuit(
    n_qubits: int,
    n_elec_s,
    mode: str,
    *,
    givens_swap: bool = False,
    init_circuit: Optional[tq.Circuit] = None,
    state_recipe: Optional[dict] = None,
) -> tq.Circuit:
    """构建设备可执行的初态制备电路（门级）。

    支持三类来源的“继承/复用”：
    - init_circuit: 直接作为基底电路（上轮真机的已编译准备部分）。
    - state_recipe.bitstring: 用比特串（str 或 List[int]）制备计算基态。
    - state_recipe.ops: 以列表形式附加门操作（如 ("x", [q]), ("cry", [ctrl, tgt, theta]) 或 ("any", [q1,q2,unitary])）。

    其中 HCB 与 givens_swap=True 时，仅影响比特串的布置顺序。
    """

    def _apply_bitstring(circ: tq.Circuit, bit) -> None:
        if isinstance(bit, str):
            bits = [int(ch) for ch in bit]
        else:
            bits = list(bit)
        # bits[0] 代表最高位还是最低位与工程约定相关。下方遵循与 HF 构造一致的下标方向。
        for i, b in enumerate(reversed(bits)):
            if b:
                circ.X(i)

    def _apply_ops(circ: tq.Circuit, ops: list) -> None:
        for op in ops:
            if not op:
                continue
            name = str(op[0]).lower()
            args = op[1] if len(op) > 1 else []
            if name == "x":
                circ.X(int(args[0]))
            elif name in ("cnot", "cx"):
                circ.CNOT(int(args[0]), int(args[1]))
            elif name == "cz":
                circ.cz(int(args[0]), int(args[1]))
            elif name == "ry":
                circ.ry(int(args[0]), theta=float(args[1]))
            elif name == "cry":
                circ.cry(int(args[0]), int(args[1]), theta=float(args[2]))
            elif name == "any":
                # args: [q1, q2, unitary] 或 [q, unitary]
                if len(args) == 3 and isinstance(args[2], (list, np.ndarray)):
                    circ.any(int(args[0]), int(args[1]), unitary=args[2])
                elif len(args) == 2:
                    circ.any(int(args[0]), unitary=args[1])
            else:
                # 未知操作，忽略或抛错，这里选择忽略以增强兼容
                continue

    # 1) 基底电路：优先使用传入的已编译电路；否则构造 HF/bitstring
    circuit = init_circuit if init_circuit is not None else tq.Circuit(n_qubits)
    if init_circuit is None:
        na, nb = unpack_nelec(n_elec_s)
        if mode in ["fermion", "qubit"]:
            for i in range(nb):
                circuit.X(n_qubits - 1 - i)
            for i in range(na):
                circuit.X(n_qubits // 2 - 1 - i)
        else:
            assert mode == "hcb"
            if not givens_swap:
                for i in range(na):
                    circuit.X(n_qubits - 1 - i)
            else:
                for i in range(na):
                    circuit.X(i)

    # 2) 追加“继承”指令（bitstring / ops）
    if state_recipe:
        bit = state_recipe.get("bitstring")
        if bit is not None and init_circuit is None:
            _apply_bitstring(circuit, bit)
        ops = state_recipe.get("ops")
        if ops:
            _apply_ops(circuit, ops)

    return circuit


def get_numeric_init_circuit(
    n_qubits: int,
    n_elec_s,
    mode: str,
    *,
    civector: Optional[np.ndarray] = None,
    givens_swap: bool = False,
) -> np.ndarray:
    """Return a statevector (numpy array) for numeric simulators; no Circuit usage.

    - If civector is provided, map it to full statevector according to (mode, n_elec_s).
    - Else return HF/bitstring as a dense statevector.
    - For hcb with givens_swap=True, apply the index permutation consistent with gate order.
    """
    na, nb = unpack_nelec(n_elec_s)
    if civector is None:
        # build HF bitstring statevector
        state = np.zeros(1 << n_qubits, dtype=np.complex128)
        bit = [0] * n_qubits
        if mode in ["fermion", "qubit"]:
            for i in range(nb):
                bit[n_qubits - 1 - i] = 1
            for i in range(na):
                bit[n_qubits // 2 - 1 - i] = 1
        else:
            assert mode == "hcb"
            if not givens_swap:
                for i in range(na):
                    bit[n_qubits - 1 - i] = 1
            else:
                for i in range(na):
                    bit[i] = 1
        idx = 0
        for i, b in enumerate(reversed(bit)):
            if b:
                idx |= (1 << i)
        state[idx] = 1.0
        return state

    # civector provided
    ci_strings = get_ci_strings(n_qubits, n_elec_s, mode == "hcb")
    statevector = civector_to_statevector(civector, n_qubits, ci_strings)
    if mode == "hcb" and givens_swap:
        statevector = statevector.reshape([2] * n_qubits)
        new_idx = list(range(n_qubits - na, n_qubits)) + list(range(n_qubits - na))
        statevector = statevector.transpose(new_idx).ravel()
    return statevector


def get_gs_unitary(theta):
    a = [
        [1, 0, 0, 0],
        [0, -tq.backend.sin(theta), tq.backend.cos(theta), 0],
        [0, tq.backend.cos(theta), tq.backend.sin(theta), 0],
        [0, 0, 0, 1],
    ]
    return tq.backend.convert_to_tensor(a)


def get_gs_indices(no: int, nv: int) -> List[Tuple[int, int]]:
    layer1 = [np.array([no - 1, no])]
    for _ in range(nv - 1):
        layer1.append(layer1[-1] + 1)
    layer1 = np.array(layer1)
    ret = [layer1]
    for _ in range(no - 1):
        ret.append(ret[-1] - 1)
    ret = np.array(ret).reshape(-1, 2)
    assert len(ret) == no * nv
    return ret.tolist()


# https://arxiv.org/pdf/2002.00035.pdf
# the swapped qubit index represents molecule orbitals 0 to n_qubits - 1
# so we need to take negative of theta
#             ┌───┐        ┌──────┐
# q_0(MO 1) : ┤ X ├────────┤      ├─────────  MO 3
#             ├───┤┌──────┐│  GS  │┌──────┐
# q_1(MO 0) : ┤ X ├┤      ├┤ 3, 1 ├┤      ├─  MO 2
#             └───┘│  GS  │├──────┤│  GS  │
# q_2(MO 3) : ─────┤ 3, 0 ├┤      ├┤ 2, 1 ├─  MO 1
#                  └──────┘│  GS  │└──────┘
# q_3(MO 2) : ─────────────┤ 2, 0 ├─────────  MO 0
#                          └──────┘

def get_circuit_givens_swap(params, n_qubits: int, n_elec: int, init_state=None) -> tq.Circuit:
    circuit = get_init_circuit(n_qubits, n_elec, mode="hcb", init_state=init_state, givens_swap=True)
    gs_indices = get_gs_indices(n_elec // 2, n_qubits - n_elec // 2)
    for i, (j, k) in enumerate(gs_indices):
        theta = params[i]
        unitary = get_gs_unitary(theta)
        try:
            name = f"Givens-SWAP({theta:.4f})"
        except TypeError:
            name = "Givens-SWAP"
        circuit.any(j, k, unitary=unitary, name=name)
    return circuit
