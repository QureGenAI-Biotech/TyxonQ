# 布局 / 路由步骤（Routing）单测：委托 qiskit.transpile 把线路映射到设备物理比特。
# 覆盖「只在真需要时路由」「本地下标空间路由」「测量逻辑序保持」「产物可复现」
# 「源 IR 一字不改」五条设计约束，以及布局挑选、参数校验与失败降级路径。
# 全部离线运行：耦合图直接由 device_rule 提供，不触发任何拓扑发现或网络请求。
from __future__ import annotations

import warnings

import pytest

from tyxonq.core.ir import Circuit
from tyxonq.compiler.api import compile as compile_api
from tyxonq.compiler.gate_compiler.specs import RoutingSpec
from tyxonq.compiler.gate_compiler.stages.layout import routing as routing_mod
from tyxonq.compiler.gate_compiler.stages.layout.routing import Routing
from tyxonq.devices.simulators.statevector import StatevectorEngine


# 4 比特「全连接少一条边 (0,3)」：足以逼出 SWAP，又小到能穷举断言。
CM4 = [[a, b] for a in range(4) for b in range(4)
       if a != b and (a, b) not in {(0, 3), (3, 0)}]
EDGES4 = {(min(a, b), max(a, b)) for a, b in CM4}

# 需要路由的线路：cx(0,3) 落在缺失的那条边上。
NEEDS_ROUTING = [
    ("h", 0), ("cx", 0, 1), ("cx", 1, 2), ("cx", 2, 3), ("cx", 0, 3),
    ("measure_z", 0), ("measure_z", 1), ("measure_z", 2), ("measure_z", 3),
]


def _circuit(ops=None, num_qubits=4):
    return Circuit(num_qubits=num_qubits, ops=list(ops if ops is not None else NEEDS_ROUTING))


def _two_qubit_pairs(ops):
    return [(op[1], op[2]) for op in ops if op[0] in {"cx", "cnot", "cz", "swap", "iswap"}]


def _all_adjacent(ops, edges=EDGES4):
    return all((min(a, b), max(a, b)) in edges for a, b in _two_qubit_pairs(ops))


def _measured(ops):
    return [op[1] for op in ops if op[0] == "measure_z"]


# ---------------------------------------------------------------------------
# 1. 只在真需要时路由
# ---------------------------------------------------------------------------
def test_no_coupling_map_returns_same_object():
    # 没有耦合图 = 视为全连接：原样返回同一对象，且不改动 device_rule
    c = _circuit()
    rule: dict = {}
    assert Routing().apply(c, device_rule=rule) is c
    assert rule == {}


def test_already_adjacent_returns_same_object():
    # 所选布局下全部相邻 → 不路由、不回填 initial_layout（既有可跑路径零回归）
    c = Circuit(num_qubits=3, ops=[("cx", 0, 1), ("cx", 1, 2), ("measure_z", 0), ("measure_z", 2)])
    rule = {"coupling_map": [[0, 1], [1, 0], [1, 2], [2, 1]], "num_qubits": 3}
    assert Routing().apply(c, device_rule=rule) is c
    assert "initial_layout" not in rule


def test_single_qubit_circuit_skips_routing():
    c = Circuit(num_qubits=1, ops=[("h", 0), ("measure_z", 0)])
    assert Routing().apply(c, device_rule={"coupling_map": CM4, "num_qubits": 4}) is c


# ---------------------------------------------------------------------------
# 2. 路由正确性（结构）
# ---------------------------------------------------------------------------
def test_routing_inserts_swaps_and_satisfies_adjacency():
    out = Routing().apply(_circuit(), device_rule={"coupling_map": CM4, "num_qubits": 4})
    assert any(op[0] == "swap" for op in out.ops)
    assert _all_adjacent(out.ops)


def test_routing_preserves_qubit_count_and_source_ir():
    c = _circuit()
    ops_before, meta_before = list(c.ops), dict(c.metadata or {})
    out = Routing().apply(c, device_rule={"coupling_map": CM4, "num_qubits": 4})
    # 产物停留在「本地下标空间」，不膨胀到设备比特数
    assert out.num_qubits == 4
    # 架构红线：源 IR 一字不改
    assert c.ops == ops_before
    assert (c.metadata or {}) == meta_before
    assert out is not c


def test_routing_writes_metadata_and_backfills_device_rule():
    rule = {"coupling_map": CM4, "num_qubits": 4}
    out = Routing().apply(_circuit(), device_rule=rule)
    meta = (out.metadata or {})["routing"]
    assert meta["algorithm"] == "qiskit.transpile"
    assert meta["optimization_level"] == 0
    assert meta["initial_layout"] == [0, 1, 2, 3]
    assert meta["num_physical_qubits"] == 4
    assert meta["swaps_inserted"] >= 1
    assert "final_layout" in meta and "seed" in meta
    assert {(min(a), max(a)) for a in meta["local_coupling_map"]} == EDGES4
    # 物理布局回填给调用方，供 Circuit.run() 作为运行期 initial_layout 下发
    assert rule["initial_layout"] == [0, 1, 2, 3]


# ---------------------------------------------------------------------------
# 3. 测量逻辑序（chem 能量依赖「经典比特 i = 逻辑比特 i」）
# ---------------------------------------------------------------------------
def test_measurement_logical_order_is_restored():
    # 第 i 个经典比特必须测「持有逻辑比特 i 的物理比特」，即 routed_meas[i] == final_layout[i]
    c = _circuit()
    src_meas = _measured(c.ops)
    out = Routing().apply(c, device_rule={"coupling_map": CM4, "num_qubits": 4})
    final_layout = out.metadata["routing"]["final_layout"]
    routed_meas = _measured(out.ops)
    assert len(routed_meas) == len(src_meas)
    assert routed_meas == [final_layout[q] for q in src_meas]


def test_partial_measurement_keeps_clbit_association():
    # 只测部分比特时，经典比特序仍须跟随逻辑比特（而非物理下标升序）
    c = Circuit(num_qubits=4, ops=[("h", 0), ("cx", 0, 3), ("measure_z", 3), ("measure_z", 0)])
    out = Routing().apply(c, device_rule={"coupling_map": CM4, "num_qubits": 4})
    final_layout = out.metadata["routing"]["final_layout"]
    assert _measured(out.ops) == [final_layout[3], final_layout[0]]


def test_routing_is_physically_equivalent_under_logical_permutation():
    # 端到端物理等价：模拟器按物理下标出 bitstr，故须按「物理→逻辑」置换后比较。
    # 线路故意用非对称旋转，使「置换弄错」能被键集比较直接捕获。
    c = Circuit(num_qubits=4, ops=[
        ("rx", 0, 0.7), ("cx", 0, 1), ("ry", 1, 1.1), ("cx", 1, 2),
        ("cx", 2, 3), ("cx", 0, 3),
        ("measure_z", 0), ("measure_z", 1), ("measure_z", 2), ("measure_z", 3),
    ])
    out = Routing().apply(c, device_rule={"coupling_map": CM4, "num_qubits": 4})
    src_meas, routed_meas = _measured(c.ops), _measured(out.ops)
    logical_of_phys = {p: l for l, p in zip(src_meas, routed_meas)}

    eng = StatevectorEngine()
    res_src = eng.run(c, shots=20000)["result"]
    res_routed = eng.run(out, shots=20000)["result"]

    def to_logical(counts, mapping, n):
        agg: dict = {}
        for bits, cnt in counts.items():
            logical = ["0"] * n
            for pos, ch in enumerate(bits):
                if pos in mapping:
                    logical[mapping[pos]] = ch
            agg["".join(logical)] = agg.get("".join(logical), 0) + cnt
        return agg

    p_src = to_logical(res_src, {i: i for i in range(len(src_meas))}, len(src_meas))
    p_rt = to_logical(res_routed, logical_of_phys, len(routed_meas))
    assert set(p_src) == set(p_rt)
    tot_s, tot_r = sum(p_src.values()), sum(p_rt.values())
    maxdiff = max(abs(p_src[k] / tot_s - p_rt.get(k, 0) / tot_r) for k in p_src)
    assert maxdiff < 0.02, f"路由前后概率分布偏差过大: {maxdiff}"


def test_input_without_measures_stays_without_measures():
    # 单独调用 Routing（标准流水线里 AutoMeasure 会先补测量）：不得把自动补的测量带回来
    c = Circuit(num_qubits=4, ops=[("h", 0), ("cx", 0, 1), ("cx", 1, 2), ("cx", 2, 3), ("cx", 0, 3)])
    out = Routing().apply(c, device_rule={"coupling_map": CM4, "num_qubits": 4})
    assert not any(op[0] == "measure_z" for op in out.ops)
    assert _all_adjacent(out.ops)


# ---------------------------------------------------------------------------
# 4. 产物可复现
# ---------------------------------------------------------------------------
def test_routing_is_deterministic_with_default_seed():
    # qiskit 的布局/SWAP 搜索默认随机；seed=0 保证同一线路反复编译产物逐 op 一致
    outs = [Routing().apply(_circuit(), device_rule={"coupling_map": CM4, "num_qubits": 4}).ops
            for _ in range(3)]
    assert outs[0] == outs[1] == outs[2]


def test_seed_none_still_produces_valid_routing():
    # 显式 seed=None 恢复随机搜索，但产物仍须合法（全部相邻）
    out = Routing(RoutingSpec(seed=None)).apply(
        _circuit(), device_rule={"coupling_map": CM4, "num_qubits": 4})
    assert _all_adjacent(out.ops)


def test_explicit_seed_is_recorded_in_metadata():
    out = Routing(RoutingSpec(seed=42)).apply(
        _circuit(), device_rule={"coupling_map": CM4, "num_qubits": 4})
    assert out.metadata["routing"]["seed"] == 42


# ---------------------------------------------------------------------------
# 5. 布局挑选（真机耦合图远大于线路、且低比特区常是线性链）
# ---------------------------------------------------------------------------
def test_explicit_initial_layout_routes_in_local_index_space():
    # 物理比特 10..13：产物仍是 k 比特本地下标，物理映射只进 metadata / device_rule
    phys_cm = [[10 + a, 10 + b] for a in range(4) for b in range(4)
               if a != b and (a, b) not in {(0, 3), (3, 0)}]
    rule = {"coupling_map": phys_cm, "num_qubits": 101, "initial_layout": [10, 11, 12, 13]}
    out = Routing().apply(_circuit(), device_rule=rule)
    assert out.num_qubits == 4
    assert rule["initial_layout"] == [10, 11, 12, 13]
    assert out.metadata["routing"]["initial_layout"] == [10, 11, 12, 13]
    assert out.metadata["routing"]["num_physical_qubits"] == 101


def test_linear_chain_topology_routes_with_swaps():
    # 真机低比特区常见形态：8 比特线性链承载 4 比特线路（cx(0,3) 跨度为 3）
    chain = [[i, i + 1] for i in range(7)]
    chain += [[b, a] for a, b in list(chain)]
    out = Routing().apply(_circuit(), device_rule={"coupling_map": chain, "num_qubits": 8})
    assert out.num_qubits == 4
    local_edges = {(min(a), max(a)) for a in out.metadata["routing"]["local_coupling_map"]}
    assert local_edges == {(0, 1), (1, 2), (2, 3)}
    assert _all_adjacent(out.ops, local_edges)
    assert out.metadata["routing"]["swaps_inserted"] >= 1


def test_broken_identity_layout_triggers_auto_pick():
    # 恒等布局 [0,1,2,3] 的诱导子图残缺（比特 0 只连 5，孤立）时，应自动改挑一块
    # 连通子图，否则 qiskit 会缩小比特数导致下标错位。
    edges = [(1, 2), (2, 3), (3, 4), (4, 5), (0, 5)]
    cm = [[a, b] for a, b in edges] + [[b, a] for a, b in edges]
    c = Circuit(num_qubits=4, ops=[("h", 0), ("cx", 0, 3), ("measure_z", 0), ("measure_z", 3)])
    out = Routing().apply(c, device_rule={"coupling_map": cm, "num_qubits": 6})
    layout = out.metadata["routing"]["initial_layout"]
    assert layout != [0, 1, 2, 3]
    assert len(layout) == 4 and len(set(layout)) == 4
    local_edges = {(min(a), max(a)) for a in out.metadata["routing"]["local_coupling_map"]}
    assert routing_mod._is_connected([[a, b] for a, b in local_edges], 4)
    assert _all_adjacent(out.ops, local_edges)


def test_unroutable_topology_raises_actionable_error():
    # 设备拆成两个互不相连的块（真机双环拓扑常见），而 cx(0,1) 跨块 →
    # 必须拦住并给出可操作建议，而不是把 qiskit 的晦涩内部错误抛给用户。
    isolated = [[0, 4], [4, 0], [1, 5], [5, 1]]
    c = Circuit(num_qubits=4, ops=[("cx", 0, 1), ("measure_z", 0)])
    with pytest.raises(ValueError, match="不连通"):
        Routing().apply(c, device_rule={"coupling_map": isolated, "num_qubits": 8})


# ---------------------------------------------------------------------------
# 6. 参数校验与失败路径
# ---------------------------------------------------------------------------
def test_circuit_larger_than_device_raises():
    c = Circuit(num_qubits=8, ops=[("cx", 0, 7), ("measure_z", 0)])
    with pytest.raises(ValueError, match="超过设备可用比特数"):
        Routing().apply(c, device_rule={"coupling_map": CM4, "num_qubits": 4})


@pytest.mark.parametrize("layout,match", [
    ([0, 1, 2], "长度"),
    ([0, 1, 1, 2], "重复"),
    ([0, 1, 2, 99], "超出设备比特范围"),
])
def test_invalid_initial_layout_raises(layout, match):
    rule = {"coupling_map": CM4, "num_qubits": 4, "initial_layout": layout}
    with pytest.raises(ValueError, match=match):
        Routing().apply(_circuit(), device_rule=rule)


def test_spec_coupling_map_takes_precedence_over_device_rule():
    # RoutingSpec.coupling_map 显式给定 → 优先于 device_rule
    out = Routing(RoutingSpec(coupling_map=CM4)).apply(_circuit(), device_rule={})
    assert any(op[0] == "swap" for op in out.ops)


def test_unsupported_algorithm_raises():
    with pytest.raises(NotImplementedError, match="路由算法"):
        Routing(RoutingSpec(algorithm="sabre-custom")).apply(
            _circuit(), device_rule={"coupling_map": CM4, "num_qubits": 4})


def test_high_level_unitary_op_asks_for_decomposition_first():
    # unitary 这类矩阵 op 无法直接进 qiskit：报错须提示先启用分解
    c = Circuit(num_qubits=4, ops=[("h", 0), ("unitary", 0, 3, "mat_key"), ("measure_z", 0)])
    with pytest.raises(NotImplementedError, match="分解"):
        Routing().apply(c, device_rule={"coupling_map": CM4, "num_qubits": 4})


def test_iswap_is_routed_despite_missing_direction_rule():
    # qiskit 对 iswap 无方向翻转规则：lvl=0 自由路由应插 SWAP 绕开，而非直接失败
    c = Circuit(num_qubits=4, ops=[("h", 0), ("iswap", 0, 3), ("measure_z", 0), ("measure_z", 3)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out = Routing().apply(c, device_rule={"coupling_map": CM4, "num_qubits": 4})
    assert _all_adjacent(out.ops)
    assert out.metadata["routing"]["swaps_inserted"] >= 1


def test_controlled_rotations_are_routed_not_rejected():
    # crx/cry/crz 是核心不分解、但 routing._TWO_QUBIT_OPS 与 to_lqcloud 都承认的具名双比特门。
    # 路由适配器（to_qiskit/from_qiskit）必须原生承接，否则真机（route=True）会在 to_qiskit 处
    # raise NotImplementedError——UCCSD 的 cry 正是这样只在真机暴露、离线（route=False）却通过。
    c = Circuit(num_qubits=4, ops=[
        ("h", 0), ("cry", 0, 3, 0.7), ("crx", 1, 2, 0.3), ("crz", 2, 0, -0.4),
        ("measure_z", 0), ("measure_z", 3),
    ])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out = Routing().apply(c, device_rule={"coupling_map": CM4, "num_qubits": 4})
    # cry(0,3) 落在 CM4 缺失的那条边上，路由后所有两比特门（cry 本身，或它被降级成的
    # cx/ry 等）都必须落到耦合边上——用 routing 自己的 _two_qubit_pairs（含 crx/cry/crz）核验。
    pairs = routing_mod._two_qubit_pairs(out.ops)
    assert pairs
    assert all((min(a, b), max(a, b)) in EDGES4 for a, b in pairs)


def test_free_routing_failure_falls_back_to_safe_basis(monkeypatch):
    # 未显式指定 basis 时，自由路由失败 → 用保守门集合重试一次并告警
    import qiskit

    real_transpile = qiskit.transpile
    seen: list = []

    def fake_transpile(circ, **kwargs):
        seen.append(kwargs.get("basis_gates"))
        if kwargs.get("basis_gates") is None:
            raise RuntimeError("模拟：自由路由无法处理该门集合")
        return real_transpile(circ, **kwargs)

    monkeypatch.setattr(qiskit, "transpile", fake_transpile)
    with pytest.warns(RuntimeWarning, match="保守门集合"):
        out = Routing().apply(_circuit(), device_rule={"coupling_map": CM4, "num_qubits": 4})

    assert seen == [None, routing_mod._SAFE_BASIS_GATES]
    assert out.metadata["routing"]["basis_gates"] == routing_mod._SAFE_BASIS_GATES
    assert _all_adjacent(out.ops)


def test_all_routing_attempts_failing_raises_runtime_error(monkeypatch):
    # 连保守门集合也失败 → RuntimeError（保留最后一个异常作为原因）
    import qiskit

    def always_fail(circ, **kwargs):
        raise RuntimeError("模拟：transpiler 彻底失败")

    monkeypatch.setattr(qiskit, "transpile", always_fail)
    with pytest.raises(RuntimeError, match="路由失败"):
        Routing().apply(_circuit(), device_rule={"coupling_map": CM4, "num_qubits": 4})


def test_explicit_unsupported_basis_gates_raises_without_substitution():
    # 用户显式指定的门集合无法完成时，不得静默换成别的门集合（那是用户的意图）
    with pytest.raises(RuntimeError, match="路由失败"):
        Routing().apply(_circuit(), basis_gates=["cx", "totally_fake_gate"],
                        device_rule={"coupling_map": CM4, "num_qubits": 4})


def test_unitary_cache_is_preserved():
    # _unitary_cache 不是 dataclass 字段，replace() 不会带过来，必须显式保留
    c = _circuit()
    c._unitary_cache = {"mat_key": [[1, 0], [0, 1]]}
    out = Routing().apply(c, device_rule={"coupling_map": CM4, "num_qubits": 4})
    assert out._unitary_cache == {"mat_key": [[1, 0], [0, 1]]}


# ---------------------------------------------------------------------------
# 7. 内部工具函数
# ---------------------------------------------------------------------------
def test_undirected_edges_dedups_and_drops_invalid():
    # 双向边去重、自环 / 负下标 / 非序列项 / 长度不对的项全部丢弃
    edges = routing_mod._undirected_edges([[3, 1], [1, 3], [2, 2], [-1, 0], "bad", [0, 1, 2]])
    assert edges == {(1, 3)}


def test_undirected_edges_handles_non_list_input():
    assert routing_mod._undirected_edges(None) == set()
    assert routing_mod._undirected_edges({"a": 1}) == set()


def test_induced_coupling_map_renumbers_and_emits_both_directions():
    local = routing_mod._induced_coupling_map({(10, 11), (11, 12), (30, 31)}, [12, 11, 10])
    assert sorted(map(tuple, local)) == sorted([(0, 1), (1, 0), (1, 2), (2, 1)])


def test_covers_all_qubits_and_connectivity():
    assert routing_mod._covers_all_qubits([[0, 1], [1, 0], [1, 2], [2, 1]], 3)
    assert not routing_mod._covers_all_qubits([[0, 1], [1, 0]], 3)
    assert routing_mod._is_connected([[0, 1], [1, 0], [1, 2], [2, 1]], 3)
    assert not routing_mod._is_connected([[0, 1], [1, 0], [2, 3], [3, 2]], 4)
    assert routing_mod._is_connected([], 1)


def test_pick_layout_is_deterministic_and_prefers_high_degree():
    # 星形图：中心 0 度数最高，应作为种子被选中；结果按物理下标升序、可复现
    star = {(0, 1), (0, 2), (0, 3), (0, 4), (1, 2)}
    first = routing_mod._pick_layout(star, 5, 3)
    assert first == routing_mod._pick_layout(star, 5, 3)
    assert len(first) == 3 and len(set(first)) == 3
    assert 0 in first
    assert first == sorted(first)


def test_component_ids_labels_isolated_qubits_separately():
    # 链 0-1-2 + 孤立 3：应得到两个连通块，孤立比特自成一块
    comp = routing_mod._component_ids([[0, 1], [1, 0], [1, 2], [2, 1]], 4)
    assert comp[0] == comp[1] == comp[2]
    assert comp[3] != comp[0]


def test_crosses_components_detects_unroutable_two_qubit_gate():
    chain = [[0, 1], [1, 0], [1, 2], [2, 1]]
    assert routing_mod._crosses_components([("cx", 0, 3)], chain, 4)
    assert not routing_mod._crosses_components([("cx", 0, 2)], chain, 4)
    # 单比特门与测量不参与连通块判定
    assert not routing_mod._crosses_components([("h", 3), ("measure_z", 3)], chain, 4)


# ---------------------------------------------------------------------------
# 8. 与编译流水线 / Circuit.compile 的集成
# ---------------------------------------------------------------------------
def test_route_true_adds_routing_step_to_pipeline():
    c = _circuit()
    rule = {"coupling_map": CM4, "num_qubits": 4}
    res = compile_api(c, platform="tyxonq", output="ir", route=True, device_rule=rule)
    assert "layout/routing" in res["metadata"]["steps"]
    compiled = res["compiled_source"]
    assert _all_adjacent(compiled.ops)
    assert res["circuit"] is c          # 源线路原封不动
    assert res["metadata"]["device_rule"]["initial_layout"] == [0, 1, 2, 3]


def test_route_false_does_not_add_routing_step():
    # route=False 时不追加步骤（无 NoOp 占位），含非相邻门的线路原样保留
    res = compile_api(_circuit(), platform="tyxonq", output="ir",
                      device_rule={"coupling_map": CM4, "num_qubits": 4})
    assert "layout/routing" not in res["metadata"]["steps"]
    assert not any(op[0] == "swap" for op in res["compiled_source"].ops)
    assert ("cx", 0, 3) in res["compiled_source"].ops


def test_routing_spec_object_is_accepted_by_compile():
    res = compile_api(
        _circuit(), platform="tyxonq", output="ir",
        route=RoutingSpec(coupling_map=CM4, optimization_level=0, seed=7),
    )
    assert res["compiled_source"].metadata["routing"]["seed"] == 7
    assert _all_adjacent(res["compiled_source"].ops)


def test_circuit_compile_caches_routed_source():
    # Circuit.compile() 返回 self，并把已路由产物存到 _compiled_source（供 run() 直接用）
    c = _circuit()
    returned = c.compile(platform="tyxonq", output="ir", route=True,
                         device_rule={"coupling_map": CM4, "num_qubits": 4})
    assert returned is c
    assert _all_adjacent(c._compiled_source.ops)
    # 源 IR 仍然一字未改
    assert c.ops == NEEDS_ROUTING


def test_routing_runs_after_decompose_in_pipeline_order():
    # 固定步骤顺序：分解必须跑在路由之前，否则 unitary 这类矩阵 op 进不了 qiskit
    res = compile_api(_circuit(), platform="tyxonq", output="ir", route=True,
                      decompose=True, device_rule={"coupling_map": CM4, "num_qubits": 4})
    steps = res["metadata"]["steps"]
    assert "layout/routing" in steps
    decompose_idx = [i for i, s in enumerate(steps) if "decompose" in s]
    assert decompose_idx and max(decompose_idx) < steps.index("layout/routing")
