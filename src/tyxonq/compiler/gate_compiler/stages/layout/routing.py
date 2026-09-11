"""布局 / 路由步骤：委托 ``qiskit.transpile`` 把线路映射到设备物理比特。

核心编译器「收集最先进功能」的一环：TyxonQ 自己不再重写一遍 SWAP 插入 / 布局搜索，
而是把这件事委托给 qiskit 的 transpiler（qiskit>=2.1.0 已是硬依赖，零新增依赖），
再把产物读回设备无关 IR。

设计要点
--------
1. **只在真需要时路由**：``device_rule``/``RoutingSpec`` 没有耦合图，或线路的两比特门
   在所选布局下已全部相邻 → **原样返回**（既有可跑路径零回归，如 HEA 2 比特在
   AGate-100 的 q0-q1 上）。
2. **在「本地下标空间」路由**：真机（如 AGate-100，101 比特）远大于线路比特数，若把
   整张设备耦合图交给 qiskit，产物会膨胀到设备比特数、计数键变成 101 位串。故先选定
   ``k`` 个物理比特（``initial_layout``），把耦合图**限制**到这些比特并重编号为
   ``0..k-1``，路由在这个小空间里做；物理映射作为运行期 ``initial_layout`` 交给设备
   驱动（lqcloud 语义：``backend.run(..., initial_layout=物理比特)``）。
3. **测量的逻辑序必须保住**：``to_lqcloud`` 等方言按 IR 中测量出现的顺序
   ``enumerate`` 分配经典比特，而 chem 的能量计算依赖「经典比特 i = 逻辑比特 i」。
   qiskit 路由会保持「经典比特 ↔ 逻辑比特」的关联但可能改变测量的书写顺序，因此这里
   按产物的 ``clbit → 物理比特`` 映射，以经典比特升序重排 ``measure_z``。
4. **门集合保守**：默认 ``optimization_level=0``（只布局 + 插 SWAP，不做门重写），
   产物门集合 ⊆ 输入门集合 ∪ {swap, 方向修正用的单比特门}。若自由路由仍然失败
   （输入含 qiskit 无翻转规则的 ``iswap``）或产物含 IR 词汇表外的门（如 ``u2``），
   退到保守门集合 ``_SAFE_BASIS_GATES`` 重试一次。
5. **产物必须可复现**：qiskit 的布局/SWAP 搜索默认带随机性（``seed_transpiler=None``），
   同一线路两次编译可能得到不同结果；故默认固定 ``RoutingSpec.seed=0``，否则真机结果
   与化学能量无法复现。

架构红线：源 IR 一字不改——本步骤返回**新**线路（``dataclasses.replace`` + 复制
``metadata``），物理布局等路由信息写进新线路的 ``metadata["routing"]``，并回填到调用方
传入的 ``device_rule["initial_layout"]``（供 ``Circuit.run()`` 作为运行期参数下发）。
"""

from __future__ import annotations

import warnings
from collections import deque
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, TYPE_CHECKING

from ..._circuit_utils import rebuild_circuit

if TYPE_CHECKING:  # pragma: no cover
    from tyxonq.core.ir import Circuit
    from tyxonq.devices import DeviceRule
    from ...specs import RoutingSpec


# 参与相邻性检查的两比特 op 词汇（measure_z / barrier 不参与）。
_TWO_QUBIT_OPS = frozenset(
    {
        "cx", "cnot", "cy", "cz", "swap", "iswap",
        "rxx", "ryy", "rzz", "crx", "cry", "crz",
        "fsim", "matchgate", "two_qubit_unitary",
    }
)

# ``from_qiskit`` 与各方言 lowering 都认识的保守门集合：仅在自由路由失败时兜底重试。
_SAFE_BASIS_GATES = ["cx", "cz", "swap", "h", "x", "rx", "ry", "rz", "measure"]


def _undirected_edges(coupling_map: Any) -> Set[Tuple[int, int]]:
    """耦合图 → 无向边集合（去重、去自环、丢弃非法项）。"""
    edges: Set[Tuple[int, int]] = set()
    if not isinstance(coupling_map, (list, tuple)):
        return edges
    for pair in coupling_map:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            continue
        try:
            q0, q1 = int(pair[0]), int(pair[1])
        except (TypeError, ValueError):
            continue
        if q0 == q1 or q0 < 0 or q1 < 0:
            continue
        edges.add((min(q0, q1), max(q0, q1)))
    return edges


def _adjacency(edges: Sequence[Tuple[int, int]]) -> Dict[int, Set[int]]:
    """无向边集合 → 邻接表。"""
    adj: Dict[int, Set[int]] = {}
    for q0, q1 in edges:
        adj.setdefault(q0, set()).add(q1)
        adj.setdefault(q1, set()).add(q0)
    return adj


def _induced_coupling_map(
    edges: Set[Tuple[int, int]], layout: Sequence[int]
) -> List[List[int]]:
    """把物理耦合图限制到 ``layout`` 选中的物理比特，并重编号为本地下标（双向边）。

    ``layout[j]`` 是本地比特 ``j`` 对应的物理比特；只有两端都被选中的边才保留。
    """
    position = {int(phys): local for local, phys in enumerate(layout)}
    local: List[List[int]] = []
    for q0, q1 in sorted(edges):
        if q0 in position and q1 in position:
            a, b = position[q0], position[q1]
            local.append([a, b])
            local.append([b, a])
    return local


def _covers_all_qubits(local_cm: Sequence[Sequence[int]], k: int) -> bool:
    """本地耦合图是否覆盖 ``0..k-1`` 全部下标（漏掉孤立比特会让 qiskit 缩小比特数）。"""
    seen = {int(q) for pair in local_cm for q in pair}
    return all(q in seen for q in range(k))


def _is_connected(local_cm: Sequence[Sequence[int]], k: int) -> bool:
    """本地耦合图是否连通（``k<=1`` 视为连通）。"""
    if k <= 1:
        return True
    adj = _adjacency({(int(a), int(b)) for a, b in local_cm})
    start = 0
    seen = {start}
    queue = deque([start])
    while queue:
        node = queue.popleft()
        for nxt in adj.get(node, ()):
            if nxt not in seen and nxt < k:
                seen.add(nxt)
                queue.append(nxt)
    return len(seen) == k


def _pick_layout(edges: Set[Tuple[int, int]], num_physical: int, k: int) -> List[int]:
    """无显式布局且恒等布局不可用时，从设备耦合图挑 ``k`` 个尽量连通的物理比特。

    种子按「度数降序、下标升序」尝试（高度数 = 连接性好、SWAP 少），从种子 BFS 扩张到
    ``k`` 个比特；返回按物理下标升序排列的布局，保证结果确定可复现。
    """
    adj = _adjacency(edges)
    nodes = sorted(adj.keys())
    if not nodes:
        return list(range(min(k, num_physical)))

    seeds = sorted(nodes, key=lambda n: (-len(adj[n]), n))
    chosen: List[int] = []
    visited: Set[int] = set()
    for seed in seeds:
        if seed in visited:
            continue
        queue = deque([seed])
        visited.add(seed)
        while queue and len(chosen) < k:
            node = queue.popleft()
            if node < num_physical:
                chosen.append(node)
            for nxt in sorted(adj.get(node, ())):
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append(nxt)
        if len(chosen) >= k:
            break
    return sorted(chosen[:k])


def _component_ids(local_cm: Sequence[Sequence[int]], k: int) -> List[int]:
    """本地下标 ``0..k-1`` 每个比特所属连通块的编号（孤立比特自成一块）。"""
    adj = _adjacency({(int(a), int(b)) for a, b in local_cm})
    comp = [-1] * k
    label = 0
    for start in range(k):
        if comp[start] != -1:
            continue
        comp[start] = label
        queue = deque([start])
        while queue:
            node = queue.popleft()
            for nxt in adj.get(node, ()):
                if 0 <= nxt < k and comp[nxt] == -1:
                    comp[nxt] = label
                    queue.append(nxt)
        label += 1
    return comp


def _crosses_components(
    ops: Sequence[Any], local_cm: Sequence[Sequence[int]], k: int
) -> bool:
    """线路是否存在**跨连通块**的两比特门（这种门永远无法用 SWAP 满足）。"""
    comp = _component_ids(local_cm, k)
    for q0, q1 in _two_qubit_pairs(ops):
        if q0 == q1 or not (0 <= q0 < k and 0 <= q1 < k):
            continue
        if comp[q0] != comp[q1]:
            return True
    return False


def _two_qubit_pairs(ops: Sequence[Any]) -> List[Tuple[int, int]]:
    """从 IR ops 里取出所有两比特门作用的比特对（含 ``unitary`` 这类矩阵 op）。"""
    pairs: List[Tuple[int, int]] = []
    for op in ops or []:
        if not isinstance(op, (list, tuple)) or len(op) < 3:
            continue
        name = str(op[0]).lower()
        if name in _TWO_QUBIT_OPS:
            pairs.append((int(op[1]), int(op[2])))
        elif name == "unitary" and len(op) == 4:
            # ("unitary", q0, q1, mat_key)：矩阵存在 _unitary_cache，比特在 op[1:3]
            pairs.append((int(op[1]), int(op[2])))
    return pairs


def _has_adjacency_conflict(
    ops: Sequence[Any], local_edges: Set[Tuple[int, int]]
) -> bool:
    """线路里是否存在**不相邻**的两比特门（无向比较）。"""
    for q0, q1 in _two_qubit_pairs(ops):
        if q0 == q1:
            continue
        if (min(q0, q1), max(q0, q1)) not in local_edges:
            return True
    return False


def _final_layout(qc_after: Any) -> Optional[List[int]]:
    """读 qiskit 产物的 final layout（``final_layout[逻辑比特] = 物理比特``）。

    仅作元信息与交叉校验用（测量逻辑序以经典比特映射为准，不依赖私有布局对象）；
    跨版本拿不到时返回 None。
    """
    layout = getattr(qc_after, "_layout", None)
    if layout is None:
        return None
    try:
        return [int(q) for q in layout.final_index_layout()]
    except Exception:  # pragma: no cover - 版本差异兜底
        return None


def _clbit_logical_map(qc_before: Any) -> Dict[int, int]:
    """路由前线路的「经典比特 → 逻辑比特」映射（由 ``to_qiskit`` 的测量顺序确定）。"""
    mapping: Dict[int, int] = {}
    for inst in getattr(qc_before, "data", []) or []:
        op = getattr(inst, "operation", None)
        if op is None or getattr(op, "name", "") != "measure":
            continue
        clbits = getattr(inst, "clbits", None)
        qubits = getattr(inst, "qubits", None)
        if not clbits or not qubits:
            continue
        mapping[qc_before.find_bit(clbits[0]).index] = qc_before.find_bit(qubits[0]).index
    return mapping


def _clbit_physical_map(qc_after: Any) -> Dict[int, int]:
    """路由产物的「经典比特 → 被测物理比特」映射。"""
    mapping: Dict[int, int] = {}
    for inst in getattr(qc_after, "data", []) or []:
        op = getattr(inst, "operation", None)
        if op is None or getattr(op, "name", "") != "measure":
            continue
        clbits = getattr(inst, "clbits", None)
        qubits = getattr(inst, "qubits", None)
        if not clbits or not qubits:
            continue
        mapping[qc_after.find_bit(clbits[0]).index] = qc_after.find_bit(qubits[0]).index
    return mapping


def _restore_measure_order(
    routed_ops: Sequence[Any], clbit_to_qubit: Dict[int, int]
) -> List[Any]:
    """按经典比特升序重排 ``measure_z``，使「经典比特 i = 路由前的第 i 个测量」。

    qiskit 路由保持「经典比特 ↔ 逻辑比特」的关联，但可能改变测量指令的书写顺序；而
    ``to_lqcloud`` 等方言按 IR 中测量的出现顺序分配经典比特，故这里以产物的
    ``clbit → 物理比特`` 映射为准重排，保证计数键的比特序与未路由时一致。
    """
    body = [op for op in routed_ops if str(op[0]).lower() != "measure_z"]
    n_measures = len(routed_ops) - len(body)
    if not clbit_to_qubit:
        # 产物没有测量（理论上不会：AutoMeasure 先于路由跑）→ 保持原样。
        return list(routed_ops)
    if len(clbit_to_qubit) != n_measures:
        warnings.warn(
            f"路由产物测量数({n_measures})与经典比特数({len(clbit_to_qubit)})不一致，"
            f"按经典比特映射重排，请核对计数比特序。",
            RuntimeWarning,
            stacklevel=2,
        )
    return body + [("measure_z", clbit_to_qubit[c]) for c in sorted(clbit_to_qubit)]


class Routing:
    """核心编译器的布局 / 路由步骤（``CompileStep``）。

    由 ``gate_compiler.steps.build_compile_steps`` 在 ``route`` 为真时装配
    （``route=False`` 时不追加本步骤，无占位）。构造签名::

        Routing(RoutingSpec(...), device_rule={"coupling_map": [...], "num_qubits": N})
    """

    name = "layout/routing"

    def __init__(
        self,
        spec: Optional["RoutingSpec"] = None,
        *,
        device_rule: Optional["DeviceRule"] = None,
    ) -> None:
        from ...specs import RoutingSpec, coerce_route

        self.spec = coerce_route(spec if spec is not None else True) or RoutingSpec()
        self.device_rule = device_rule

    # -- 内部：调 qiskit.transpile（带保守门集合兜底） -------------------------
    def _route_via_qiskit(
        self,
        circuit: "Circuit",
        local_cm: List[List[int]],
        basis_gates: Optional[Sequence[str]],
    ) -> Tuple[Any, Any, Any, int, Optional[List[str]]]:
        """IR →(to_qiskit) → qiskit.transpile →(from_qiskit) → 已路由 IR。

        Returns:
            ``(routed_ir, qc_before, qc_after, used_level, used_basis)``；``qc_before``
            用于反查「经典比特 → 逻辑比特」以还原测量逻辑序。

        Raises:
            RuntimeError: 所有尝试都失败（保留最后一个异常作为原因）。
        """
        from qiskit import transpile as qk_transpile

        from ...interop.qiskit_interop import from_qiskit, to_qiskit

        try:
            qc_before = to_qiskit(circuit, add_measures=True)
        except NotImplementedError as exc:
            raise NotImplementedError(
                f"路由需要先把 IR 转成 qiskit 线路，但存在不支持的 op：{exc}。"
                f"请先启用分解（decompose=True / DecomposeSpec(matchgate=True)）"
                f"把 unitary 等高层 op 降到原生门后再路由。"
            ) from exc

        level = int(self.spec.optimization_level)
        seed = getattr(self.spec, "seed", 0)
        attempts: List[Tuple[int, Optional[List[str]]]] = [
            (level, list(basis_gates) if basis_gates else None)
        ]
        # 兜底：自由路由报方向错误（iswap 无翻转规则）或产物含 IR 不认的门（u2 等）时，
        # 用保守门集合重试（产物一定能读回 IR，且各方言 lowering 都支持）。
        if attempts[0][1] is None:
            attempts.append((level, list(_SAFE_BASIS_GATES)))

        last_error: Optional[BaseException] = None
        for used_level, used_basis in attempts:
            try:
                qc_after = qk_transpile(
                    qc_before,
                    coupling_map=[list(pair) for pair in local_cm],
                    basis_gates=used_basis,
                    optimization_level=used_level,
                    seed_transpiler=seed,
                )
                routed = from_qiskit(qc_after)
            except Exception as exc:
                last_error = exc
                continue
            if used_basis is not None and used_basis != attempts[0][1]:
                warnings.warn(
                    f"qiskit 自由路由失败（{last_error}），已改用保守门集合 "
                    f"{used_basis} 重试完成路由。",
                    RuntimeWarning,
                    stacklevel=2,
                )
            return routed, qc_before, qc_after, used_level, used_basis

        raise RuntimeError(
            f"qiskit.transpile 路由失败（optimization_level={level}）：{last_error}"
        ) from last_error

    # -- CompileStep 协议 ----------------------------------------------------
    def apply(
        self,
        circuit: "Circuit",
        device_rule: Optional["DeviceRule"] = None,
        **opts: Any,
    ) -> "Circuit":
        """按耦合图对 ``circuit`` 做布局与路由；无需路由时原样返回。"""
        if str(getattr(self.spec, "algorithm", "qiskit")).lower() != "qiskit":
            raise NotImplementedError(
                f"路由算法暂只支持 'qiskit'，收到 {self.spec.algorithm!r}"
            )

        rule = device_rule if isinstance(device_rule, dict) else self.device_rule
        rule = rule or {}
        coupling_map = self.spec.coupling_map or rule.get("coupling_map")
        edges = _undirected_edges(coupling_map)
        if not edges:
            # 没有耦合图 = 视为全连接：不路由（也不改动 device_rule）。
            return circuit

        k = int(circuit.num_qubits)
        if k <= 1:
            return circuit
        num_physical = rule.get("num_qubits") or rule.get("max_qubits")
        num_physical = int(num_physical) if num_physical else (
            max(max(q0, q1) for q0, q1 in edges) + 1
        )
        if k > num_physical:
            raise ValueError(
                f"线路比特数 {k} 超过设备可用比特数 {num_physical}，无法路由。"
            )

        # 1) 选物理布局：显式 initial_layout 优先；否则恒等（与设备驱动默认一致，
        #    保证既有可跑路径零回归）。
        explicit_layout = self.spec.initial_layout or rule.get("initial_layout")
        if explicit_layout is not None:
            layout = [int(q) for q in explicit_layout]
            if len(layout) != k:
                raise ValueError(
                    f"initial_layout 长度 {len(layout)} 与线路比特数 {k} 不一致。"
                )
            if len(set(layout)) != k:
                raise ValueError(f"initial_layout 含重复物理比特: {layout}")
            if any(q < 0 or q >= num_physical for q in layout):
                raise ValueError(
                    f"initial_layout {layout} 超出设备比特范围 [0, {num_physical})。"
                )
        else:
            layout = list(range(k))

        local_cm = _induced_coupling_map(edges, layout)
        local_edges = _undirected_edges(local_cm)

        # 2) 相邻性检查：所选布局下已全部相邻 → 原样返回（不做任何改动）。
        if not _has_adjacency_conflict(circuit.ops, local_edges):
            return circuit

        # 3) 需要路由：恒等布局的诱导子图若残缺（孤立比特 / 不连通），自动挑一块
        #    连通的物理比特子图，否则 qiskit 会缩小比特数导致下标错位。
        if explicit_layout is None and not (
            _covers_all_qubits(local_cm, k) and _is_connected(local_cm, k)
        ):
            picked = _pick_layout(edges, num_physical, k)
            if len(picked) == k and picked != layout:
                layout = picked
                local_cm = _induced_coupling_map(edges, layout)
                local_edges = _undirected_edges(local_cm)
        if not _covers_all_qubits(local_cm, k):
            raise ValueError(
                f"设备耦合图在 {k} 个物理比特上无法构成完整可路由子图"
                f"（缺少比特或耦合边）；请显式指定 route=RoutingSpec(initial_layout=[...])。"
            )
        # 跨连通块的两比特门用 SWAP 永远满足不了；qiskit 对此只会抛晦涩的内部错误，
        # 故先拦住并给出可操作建议（改用同一连通块内的物理比特）。
        if _crosses_components(circuit.ops, local_cm, k):
            raise ValueError(
                f"设备耦合图在所选物理比特 {layout} 上不连通，而线路存在跨连通块的两比特门，"
                f"无法路由；请用 route=RoutingSpec(initial_layout=[...]) 指定同一连通块内的"
                f" {k} 个物理比特。"
            )

        routed, qc_before, qc_after, used_level, used_basis = self._route_via_qiskit(
            circuit, local_cm, opts.get("basis_gates")
        )
        if int(qc_after.num_qubits) != k:
            raise RuntimeError(
                f"qiskit 路由产物比特数 {qc_after.num_qubits} 与线路比特数 {k} 不一致，"
                f"拒绝返回下标可能错位的线路。"
            )

        clbit_to_qubit = _clbit_physical_map(qc_after)
        clbit_to_logical = _clbit_logical_map(qc_before)
        has_measures = any(
            str(op[0]).lower() == "measure_z"
            for op in circuit.ops
            if isinstance(op, (list, tuple)) and op
        )
        if has_measures:
            new_ops = _restore_measure_order(routed.ops, clbit_to_qubit)
        else:
            # 输入本来没有测量（标准流水线里 AutoMeasure 会先补上，只可能是单独调用
            # Routing）：保持「无测量」语义，不把 to_qiskit 自动补的测量带回来。
            new_ops = [
                op for op in routed.ops if str(op[0]).lower() != "measure_z"
            ]

        # 后置校验：产物必须真的相邻（防 qiskit 静默不路由 / 版本行为变化），
        # 否则交下去只会在真机上被 SDK 的相邻性校验拒掉，排查成本高得多。
        if _has_adjacency_conflict(new_ops, local_edges):
            bad = [
                (min(q0, q1), max(q0, q1))
                for q0, q1 in _two_qubit_pairs(new_ops)
                if q0 != q1 and (min(q0, q1), max(q0, q1)) not in local_edges
            ]
            raise RuntimeError(
                f"qiskit 路由产物仍存在不相邻的两比特门 {bad}（本地耦合图 {local_cm}）；"
                f"请尝试 route=RoutingSpec(optimization_level=1) 或显式指定 basis_gates。"
            )

        # 交叉校验：由经典比特映射推出的 logical→physical 应与 qiskit final layout 一致；
        # 不一致说明 qiskit 版本行为变了 → 显式告警（测量序仍以经典比特映射为准）。
        final_layout = _final_layout(qc_after)
        if final_layout is not None and clbit_to_qubit:
            mismatched = [
                (logical, clbit_to_qubit[clbit], final_layout[logical])
                for clbit, logical in sorted(clbit_to_logical.items())
                if clbit in clbit_to_qubit
                and logical < len(final_layout)
                and final_layout[logical] != clbit_to_qubit[clbit]
            ]
            if mismatched:
                warnings.warn(
                    f"qiskit final layout 与经典比特映射不一致（逻辑,经典比特映射,final layout）="
                    f"{mismatched}；已按经典比特映射还原测量序，请核对 qiskit 版本行为。",
                    RuntimeWarning,
                    stacklevel=2,
                )

        # metadata 必须复制后再写：源 IR 一字不改（架构红线）。
        metadata = dict(getattr(circuit, "metadata", {}) or {})
        metadata["routing"] = {
            "algorithm": "qiskit.transpile",
            "optimization_level": used_level,
            "seed": getattr(self.spec, "seed", 0),
            "basis_gates": used_basis,
            "initial_layout": list(layout),
            "final_layout": final_layout,
            "num_physical_qubits": num_physical,
            "local_coupling_map": [list(pair) for pair in local_cm],
            "swaps_inserted": sum(
                1 for op in new_ops if str(op[0]).lower() == "swap"
            ),
        }
        new_circuit = rebuild_circuit(circuit, ops=new_ops, metadata=metadata)

        # 回填物理布局：Circuit.run() 据此把它作为运行期 initial_layout 下发给驱动
        # （lqcloud 的 physical_qubits → SDK initial_layout）。
        if isinstance(device_rule, dict):
            device_rule["initial_layout"] = list(layout)
        elif isinstance(self.device_rule, dict):
            self.device_rule["initial_layout"] = list(layout)
        return new_circuit


__all__ = ["Routing"]
