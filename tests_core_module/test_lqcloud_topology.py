# LQCloud 设备拓扑抓取单测：耦合图归一（服务端边列表不对称 → 补全双向）、比特数与
# 原生门集合的防御式字段探测、落盘 raw 的大字段裁剪，以及跨 provider 的 fetch 分派。
# 全部用假 SDK + 假 api_key 解析，**不产生任何真实网络请求**（真机探查只在手工验证时做）。
from __future__ import annotations

import pytest

from tyxonq.compiler.platforms.lqcloud import topology as lqtopo
from tyxonq.devices.hardware import topology as devtopo
from tyxonq.devices.hardware.lqcloud import driver as lqdriver


# AGate-100 现网 config 的最小仿真：qubits=101、边列表不对称、含两个大字段。
def _agatelike_config(coupling_map=None, **overrides):
    config = {
        "name": "AGate-100",
        "qubits": 101,
        "status": "active",
        "native_gates": None,
        "topology": {
            "type": "grid",
            "rows": 7,
            "cols": 15,
            "coupling_map": coupling_map if coupling_map is not None else [[0, 1], [1, 2], [1, 0]],
            "qubit_coordinates": {str(i): [i, 0] for i in range(101)},
        },
        "properties": {"qubit_metrics": {"t1": [1.0] * 101}, "coupler_metrics": {}},
    }
    config.update(overrides)
    return config


class _FakeBackend:
    def __init__(self, config):
        self.config = config


class _FakeProvider:
    """仿真 lqcloud.LQCloudProvider：只记录构造参数与被请求的 backend 名。"""

    config: object = None
    instances: list = []

    def __init__(self, api_key=None, interactive=True):
        self.api_key = api_key
        self.interactive = interactive
        self.requested = None
        type(self).instances.append(self)

    def get_backend(self, name):
        self.requested = name
        return _FakeBackend(type(self).config)


@pytest.fixture
def fake_sdk(monkeypatch):
    """把驱动的 SDK 导入与 api_key 解析换成假的（get_topology 在函数内延迟导入）。"""
    _FakeProvider.instances = []
    _FakeProvider.config = _agatelike_config()
    monkeypatch.setattr(lqdriver, "_sdk_classes",
                        lambda: (_FakeProvider, object, object))
    monkeypatch.setattr(lqdriver, "_resolve_api_key", lambda token, opts: "fake-api-key")
    return _FakeProvider


# ---------------------------------------------------------------------------
# 1. 耦合图归一
# ---------------------------------------------------------------------------
def test_normalize_coupling_map_completes_both_directions():
    # 服务端边列表不保证对称（AGate-100 是 157 条，奇数），物理耦合本身无向 → 补全双向
    out = lqtopo._normalize_coupling_map([[0, 1], [1, 2]])
    assert out == [[0, 1], [1, 0], [1, 2], [2, 1]]


def test_normalize_coupling_map_dedups_and_sorts():
    assert lqtopo._normalize_coupling_map([[3, 1], [1, 3], [3, 1]]) == [[1, 3], [3, 1]]


def test_normalize_coupling_map_drops_invalid_entries():
    # 自环 / 负下标 / 长度不对 / 非数值 / 非序列项一律丢弃
    assert lqtopo._normalize_coupling_map(
        [[2, 2], [-1, 0], [0, 1, 2], ["a", "b"], "bad", None]
    ) == []


def test_normalize_coupling_map_handles_non_list_input():
    assert lqtopo._normalize_coupling_map(None) == []
    assert lqtopo._normalize_coupling_map({"coupling": [[0, 1]]}) == []


def test_normalize_coupling_map_accepts_tuples_and_strings():
    # SDK 若返回 tuple 或字符串下标也应能吃下（防御式解析）
    assert lqtopo._normalize_coupling_map([(0, 1), ["1", "2"]]) == [
        [0, 1], [1, 0], [1, 2], [2, 1]
    ]


# ---------------------------------------------------------------------------
# 2. 比特数 / 原生门集合的防御式探测
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("key", ["qubits", "num_qubits", "n_qubits", "qubit_count"])
def test_pick_num_qubits_accepts_known_field_names(key):
    assert lqtopo._pick_num_qubits({key: 101}, []) == 101


def test_pick_num_qubits_accepts_digit_string():
    assert lqtopo._pick_num_qubits({"qubits": " 101 "}, []) == 101


def test_pick_num_qubits_ignores_bool_and_nonpositive():
    # bool 是 int 的子类，必须显式排除；0 / 负数视为缺失
    assert lqtopo._pick_num_qubits({"qubits": True}, [[0, 1], [1, 0]]) == 2
    assert lqtopo._pick_num_qubits({"qubits": 0}, [[0, 5], [5, 0]]) == 6


def test_pick_num_qubits_falls_back_to_coupling_map():
    assert lqtopo._pick_num_qubits({}, [[3, 7], [7, 3]]) == 8
    assert lqtopo._pick_num_qubits({}, []) is None


def test_pick_native_gates_variants():
    # 现网设备多为 None（服务端未声明）
    assert lqtopo._pick_native_gates({"native_gates": None}, {}) is None
    assert lqtopo._pick_native_gates({"native_gates": ["rx", "cz"]}, {}) == ["rx", "cz"]
    assert lqtopo._pick_native_gates({}, {"basis_gates": ("h", "cx")}) == ["h", "cx"]
    assert lqtopo._pick_native_gates({}, {"gates": {"rx": {}, "cz": {}}}) == ["rx", "cz"]
    # 空容器视为未声明
    assert lqtopo._pick_native_gates({"native_gates": []}, {}) is None


# ---------------------------------------------------------------------------
# 3. get_topology 端到端（假 SDK）
# ---------------------------------------------------------------------------
def test_get_topology_returns_normalized_rule(fake_sdk):
    rule = lqtopo.get_topology("AGate-100")

    assert rule["provider"] == "lqcloud"
    assert rule["device"] == "AGate-100"
    assert rule["num_qubits"] == 101
    assert rule["status"] == "active"
    assert rule["native_gates"] is None
    assert rule["coupling_map"] == [[0, 1], [1, 0], [1, 2], [2, 1]]
    # 只读发现：构造 provider 时关掉交互式登录，且只请求了一个 backend
    assert len(fake_sdk.instances) == 1
    assert fake_sdk.instances[0].interactive is False
    assert fake_sdk.instances[0].api_key == "fake-api-key"
    assert fake_sdk.instances[0].requested == "AGate-100"


def test_get_topology_strips_provider_prefix(fake_sdk):
    assert lqtopo.get_topology("lqcloud::AGate-100")["device"] == "AGate-100"
    assert fake_sdk.instances[0].requested == "AGate-100"


def test_get_topology_raw_drops_bulky_fields(fake_sdk):
    # raw 仅供缓存文件排查用：必须裁掉 properties 与 qubit_coordinates 两个大字段，
    # 否则单设备缓存会从几 KB 涨到 30 KB+
    raw = lqtopo.get_topology("AGate-100")["raw"]
    assert "properties" not in raw
    assert "qubit_coordinates" not in raw["topology"]
    # 有用信息保留
    assert raw["topology"]["coupling_map"] == [[0, 1], [1, 2], [1, 0]]
    assert raw["qubits"] == 101


def test_get_topology_without_coupling_map_yields_empty_list(fake_sdk):
    # 设备未声明耦合图 → 空列表（调用方据此判定「全连接 / 无需路由」）
    fake_sdk.config = _agatelike_config(coupling_map=None, topology={"type": "all-to-all"})
    rule = lqtopo.get_topology("AGate-100")
    assert rule["coupling_map"] == []
    assert rule["num_qubits"] == 101


def test_get_topology_requires_device_name(fake_sdk):
    with pytest.raises(ValueError, match="设备名"):
        lqtopo.get_topology("")


def test_get_topology_rejects_unexpected_config_type(fake_sdk):
    fake_sdk.config = ["not", "a", "dict"]
    with pytest.raises(RuntimeError, match="非预期的 config 类型"):
        lqtopo.get_topology("AGate-100")


def test_get_topology_propagates_sdk_errors(fake_sdk, monkeypatch):
    # SDK 报错（设备不存在 / 凭证失效）须原样上抛，由上层拓扑缓存统一兜底降级
    def boom(token, opts):
        raise RuntimeError("缺少 LQCloud API key")

    monkeypatch.setattr(lqdriver, "_resolve_api_key", boom)
    with pytest.raises(RuntimeError, match="API key"):
        lqtopo.get_topology("AGate-100")


def test_get_topology_passes_api_key_option_through(fake_sdk, monkeypatch):
    # opts 里的 api_key= 应交给驱动的解析链（而不是被本模块自己吞掉）
    seen = {}

    def spy(token, opts):
        seen.update(opts)
        return "resolved"

    monkeypatch.setattr(lqdriver, "_resolve_api_key", spy)
    lqtopo.get_topology("AGate-100", api_key="user-key")
    assert seen.get("api_key") == "user-key"
    assert fake_sdk.instances[0].api_key == "resolved"


# ---------------------------------------------------------------------------
# 4. 跨 provider 的 fetch 分派
# ---------------------------------------------------------------------------
def test_fetch_topology_dispatches_to_lqcloud(fake_sdk):
    rule = devtopo.fetch_topology("lqcloud", "AGate-100")
    assert rule["coupling_map"] == [[0, 1], [1, 0], [1, 2], [2, 1]]


def test_fetch_topology_dispatch_is_case_insensitive(fake_sdk):
    assert devtopo.fetch_topology("LQCloud", "AGate-100")["provider"] == "lqcloud"


@pytest.mark.parametrize("provider", ["guodun", "quafu", "qcos", "tyxonq", "ibm", ""])
def test_fetch_topology_returns_none_for_unimplemented_providers(provider):
    # 未实现拓扑发现的 provider 返回 None（= 不路由），而不是抛异常
    assert devtopo.fetch_topology(provider, "SomeDevice") is None


# ---------------------------------------------------------------------------
# 5. 与 TTL 缓存的衔接
# ---------------------------------------------------------------------------
def test_get_device_rule_uses_lqcloud_fetcher_and_caches(tmp_path, monkeypatch, fake_sdk):
    # 完整链路：get_device_rule → fetch_topology → lqcloud.get_topology → 写盘 → 内存缓存
    monkeypatch.setenv(devtopo.CACHE_DIR_ENV, str(tmp_path))
    devtopo._MEMORY_CACHE.clear()
    try:
        rule = devtopo.get_device_rule("lqcloud", "AGate-100")
        assert rule["coupling_map"] == [[0, 1], [1, 0], [1, 2], [2, 1]]
        assert rule["num_qubits"] == 101 and rule["max_qubits"] == 101
        assert len(fake_sdk.instances) == 1

        # 第二次命中会话内存缓存：不再构造 provider
        assert devtopo.get_device_rule("lqcloud", "AGate-100")["num_qubits"] == 101
        assert len(fake_sdk.instances) == 1

        # 缓存文件里存的是归一后的双向耦合图，且不含被裁掉的大字段
        import json
        entry = json.loads(
            devtopo.cache_path("lqcloud").read_text(encoding="utf-8")
        )["AGate-100"]
        assert entry["coupling_map"] == [[0, 1], [1, 0], [1, 2], [2, 1]]
        assert "properties" not in entry["raw"]
    finally:
        devtopo._MEMORY_CACHE.clear()
