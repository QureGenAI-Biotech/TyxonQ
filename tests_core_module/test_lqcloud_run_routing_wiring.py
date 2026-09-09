# Circuit.run() 的 lqcloud 硬件感知接线单测：提交前发现设备拓扑（1 天 TTL 缓存）→
# 交给核心编译器做布局 / SWAP → 把路由选定的物理比特作为运行期 initial_layout 下发。
# 重点验证「拓扑发现绝不能阻断提交」这条离线韧性约束。
# 全部用假 get_device_rule + 假 compile_api + 假 devices.base.run，不联网、不提交作业。
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from tyxonq.core.ir import Circuit
from tyxonq.core.ir import circuit as circuit_mod
from tyxonq.devices import base as device_base
from tyxonq.devices.hardware import topology as devtopo

CM4 = [[a, b] for a in range(4) for b in range(4)
       if a != b and (a, b) not in {(0, 3), (3, 0)}]

BELLOPS = [("h", 0), ("cx", 0, 3), ("measure_z", 0), ("measure_z", 3)]


def _circuit(**kwargs):
    return Circuit(num_qubits=4, ops=list(BELLOPS), **kwargs)


@pytest.fixture
def harness(monkeypatch):
    """拦下拓扑发现、编译入口与设备提交，记录各自收到的参数。"""
    calls = {"discovered": [], "compile": [], "run": []}

    def fake_get_device_rule(provider, device, token=None, **opts):
        calls["discovered"].append({"provider": provider, "device": device, "token": token})
        return {"provider": "lqcloud", "device": device, "coupling_map": [list(p) for p in CM4],
                "num_qubits": 4, "max_qubits": 4, "fetched_at": "2026-09-09T00:00:00+00:00"}

    def fake_compile_api(circ, **kwargs):
        calls["compile"].append(kwargs)
        return {"compiled_source": "FAKE_LQCLOUD_SOURCE", "metadata": {}}

    def fake_run(**kwargs):
        calls["run"].append(kwargs)
        return []

    monkeypatch.setattr(devtopo, "get_device_rule", fake_get_device_rule)
    monkeypatch.setattr(circuit_mod, "compile_api", fake_compile_api)
    monkeypatch.setattr(device_base, "run", fake_run)
    return calls


# ---------------------------------------------------------------------------
# 1. 拓扑发现 → device_rule → 编译
# ---------------------------------------------------------------------------
def test_run_discovers_topology_and_enables_routing(harness):
    _circuit().run(provider="lqcloud", device="AGate-100", shots=8, token="tk")

    assert harness["discovered"] == [{"provider": "lqcloud", "device": "AGate-100", "token": "tk"}]
    kwargs = harness["compile"][0]
    assert kwargs["platform"] == "lqcloud" and kwargs["output"] == "lqcloud"
    assert kwargs["route"] is True
    assert kwargs["device_rule"]["coupling_map"] == CM4
    assert kwargs["device_rule"]["num_qubits"] == 4
    # 提交的是编译产物，而非源线路
    assert harness["run"][0]["source"] == "FAKE_LQCLOUD_SOURCE"


def test_run_does_not_route_when_device_declares_no_topology(harness, monkeypatch):
    # 设备未声明耦合图（空列表）→ 视为全连接，保持纯直译
    monkeypatch.setattr(devtopo, "get_device_rule",
                        lambda provider, device, token=None, **o: {"coupling_map": []})
    _circuit().run(provider="lqcloud", device="AGate-100", shots=8)

    kwargs = harness["compile"][0]
    assert kwargs["route"] is False
    # 规则本身仍透传（保留 num_qubits / fetched_at 等信息）；空耦合图由 Routing
    # 自己判定为「全连接 → 不路由」，无需在 run() 里提前丢弃。
    assert kwargs["device_rule"] == {"coupling_map": []}


def test_run_skips_discovery_when_device_rule_already_supplied(harness):
    # 调用方已给出耦合图（如离线复现真机实验）→ 不再联网发现
    supplied = {"coupling_map": [[0, 1], [1, 0]], "num_qubits": 4}
    _circuit(compile_options={"device_rule": dict(supplied)}).run(
        provider="lqcloud", device="AGate-100", shots=8)

    assert harness["discovered"] == []
    assert harness["compile"][0]["device_rule"]["coupling_map"] == [[0, 1], [1, 0]]


# ---------------------------------------------------------------------------
# 2. 离线韧性：拓扑发现失败绝不阻断提交
# ---------------------------------------------------------------------------
def test_topology_discovery_failure_warns_and_still_submits(harness, monkeypatch):
    def boom(provider, device, token=None, **opts):
        raise RuntimeError("no network")

    monkeypatch.setattr(devtopo, "get_device_rule", boom)
    with pytest.warns(RuntimeWarning, match="拓扑发现失败"):
        _circuit().run(provider="lqcloud", device="AGate-100", shots=8)

    # 仍然按直译提交，且不带 device_rule
    assert harness["compile"][0]["route"] is False
    assert harness["compile"][0]["device_rule"] is None
    assert harness["run"][0]["source"] == "FAKE_LQCLOUD_SOURCE"


def test_discovery_returning_empty_rule_keeps_direct_translation(harness, monkeypatch):
    # get_device_rule 自身已把失败降级成空字典（不抛），run() 同样不得路由
    monkeypatch.setattr(devtopo, "get_device_rule", lambda *a, **k: {})
    _circuit().run(provider="lqcloud", device="AGate-100", shots=8)

    assert harness["compile"][0]["route"] is False
    assert harness["compile"][0]["device_rule"] is None
    assert len(harness["run"]) == 1


# ---------------------------------------------------------------------------
# 3. 物理比特布局：用户显式指定 vs 路由选定
# ---------------------------------------------------------------------------
def test_user_physical_qubits_become_initial_layout(harness):
    # 用户显式给的 physical_qubits 即运行期 initial_layout，路由以它为初始布局
    _circuit().run(provider="lqcloud", device="AGate-100", shots=8,
                   physical_qubits=[10, 11, 12, 13])

    assert harness["compile"][0]["device_rule"]["initial_layout"] == [10, 11, 12, 13]
    # 并原样透传给设备层
    assert harness["run"][0]["physical_qubits"] == [10, 11, 12, 13]


def test_routed_layout_is_forwarded_as_runtime_physical_qubits(harness, monkeypatch):
    # Routing 步骤会把选定的物理布局回填进 device_rule；run() 须据此下发 physical_qubits
    def compile_and_backfill(circ, **kwargs):
        rule = kwargs.get("device_rule")
        if isinstance(rule, dict):
            rule["initial_layout"] = [5, 6, 7, 8]
        harness["compile"].append(kwargs)
        return {"compiled_source": "FAKE_LQCLOUD_SOURCE", "metadata": {}}

    monkeypatch.setattr(circuit_mod, "compile_api", compile_and_backfill)
    _circuit().run(provider="lqcloud", device="AGate-100", shots=8)

    assert harness["run"][0]["physical_qubits"] == [5, 6, 7, 8]


def test_user_physical_qubits_are_not_overwritten_by_routing(harness, monkeypatch):
    # 用户显式指定优先：路由回填不得覆盖用户给的物理比特
    def compile_and_backfill(circ, **kwargs):
        rule = kwargs.get("device_rule")
        if isinstance(rule, dict):
            rule["initial_layout"] = [5, 6, 7, 8]
        harness["compile"].append(kwargs)
        return {"compiled_source": "FAKE_LQCLOUD_SOURCE", "metadata": {}}

    monkeypatch.setattr(circuit_mod, "compile_api", compile_and_backfill)
    _circuit().run(provider="lqcloud", device="AGate-100", shots=8,
                   physical_qubits=[10, 11, 12, 13])

    assert harness["run"][0]["physical_qubits"] == [10, 11, 12, 13]


def test_no_physical_qubits_when_routing_did_not_pick_a_layout(harness):
    # 未路由（或路由未改布局）时不得凭空塞 physical_qubits，保持既有直译行为
    _circuit().run(provider="lqcloud", device="AGate-100", shots=8)
    assert "physical_qubits" not in harness["run"][0]


# ---------------------------------------------------------------------------
# 4. device_rule / route 不得经 options 重复传入
# ---------------------------------------------------------------------------
def test_device_rule_and_route_are_popped_from_options(harness):
    # GateCompiler.compile 把 device_rule 当独立参数，经 options 重复传会冲突
    _circuit(compile_options={"device_rule": {"coupling_map": CM4}, "route": True,
                              "basis_gates": ["rx", "cz"]}).run(
        provider="lqcloud", device="AGate-100", shots=8)

    options = harness["compile"][0]["options"]
    assert "device_rule" not in options
    assert "route" not in options
    # 其余编译选项照常透传
    assert options.get("basis_gates") == ["rx", "cz"]


def test_source_ir_is_untouched_by_run(harness):
    # 架构红线：run() 走完全程后源 IR 仍一字未改
    c = _circuit()
    ops_before = list(c.ops)
    c.run(provider="lqcloud", device="AGate-100", shots=8)
    assert c.ops == ops_before
    assert c.num_qubits == 4


def test_shots_token_and_device_reach_device_layer(harness):
    # shots / token / device 应原样抵达设备层
    _circuit().run(provider="lqcloud", device="AGate-100", shots=2048, token="tk")
    submitted = harness["run"][0]
    assert submitted["shots"] == 2048
    assert submitted["device"] == "AGate-100"
    assert submitted["provider"] == "lqcloud"
    assert submitted["token"] == "tk"


def test_harness_mock_is_not_leaked_into_production_path():
    # 反向哨兵：未打桩时 devices.base.run 必须是真函数（防止 fixture 意外全局生效）
    assert not isinstance(device_base.run, MagicMock)
