# 设备拓扑发现 + 持久化 TTL 缓存单测：缺失→抓取写盘、新鲜→不联网、过期→刷新、
# 会话内存缓存、抓取失败降级沿用陈旧缓存、彻底拿不到→空 DeviceRule（不中断编译）、
# 未实现 provider、force_refresh、设备名归一、损坏缓存文件、清缓存。
# 全部用 TYXONQ_TOPOLOGY_CACHE_DIR 重定向到 tmp_path + monkeypatch fetch_topology，
# 不产生任何真实网络请求。
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from tyxonq.devices.hardware import topology as topo


@pytest.fixture
def cache_env(tmp_path, monkeypatch):
    """把缓存目录指向 tmp_path，并清空会话内存缓存（用例之间必须互不干扰）。"""
    monkeypatch.setenv(topo.CACHE_DIR_ENV, str(tmp_path))
    topo._MEMORY_CACHE.clear()
    yield tmp_path
    topo._MEMORY_CACHE.clear()


def _iso(days_ago: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat(
        timespec="seconds"
    )


def _topology(num_qubits: int = 5, tag: str = "v1") -> dict:
    """一份假的抓取结果（形状与 platforms/lqcloud/topology.get_topology 一致）。"""
    return {
        "provider": "lqcloud",
        "device": "FakeDevice",
        "num_qubits": num_qubits,
        "coupling_map": [[0, 1], [1, 0], [1, 2], [2, 1]],
        "native_gates": ["rx", "rz", "cz"],
        "status": "active",
        "raw": {"tag": tag},
    }


class _FetchSpy:
    """记录调用次数的假 fetch_topology。"""

    def __init__(self, result=None, error: Exception | None = None):
        self.calls = 0
        self.result = result
        self.error = error

    def __call__(self, provider, device, token=None, **opts):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.result


def test_missing_cache_fetches_and_persists(cache_env, monkeypatch):
    # 缓存文件不存在 → 联网抓取一次 → 写盘 → 返回可用 DeviceRule
    spy = _FetchSpy(result=_topology())
    monkeypatch.setattr(topo, "fetch_topology", spy)

    rule = topo.get_device_rule("lqcloud", "FakeDevice")

    assert spy.calls == 1
    assert rule["coupling_map"] == [[0, 1], [1, 0], [1, 2], [2, 1]]
    assert rule["num_qubits"] == 5 and rule["max_qubits"] == 5
    assert rule["native_gates"] == ["rx", "rz", "cz"]
    assert rule["provider"] == "lqcloud" and rule["device"] == "FakeDevice"
    # raw 不进 DeviceRule：避免编译 metadata 被大字段撑爆
    assert "raw" not in rule

    path = topo.cache_path("lqcloud")
    assert path.is_file()
    on_disk = json.loads(path.read_text(encoding="utf-8"))["FakeDevice"]
    assert on_disk["num_qubits"] == 5
    assert on_disk["raw"] == {"tag": "v1"}
    assert on_disk["fetched_at"]


def test_fresh_disk_cache_skips_fetch(cache_env, monkeypatch):
    # 磁盘缓存未过期（<1 天）→ 完全不联网
    spy = _FetchSpy(result=_topology(tag="should-not-be-used"))
    monkeypatch.setattr(topo, "fetch_topology", spy)
    path = topo.cache_path("lqcloud")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"FakeDevice": {**_topology(tag="cached"), "fetched_at": _iso(0.1)}}),
        encoding="utf-8",
    )

    rule = topo.get_device_rule("lqcloud", "FakeDevice")

    assert spy.calls == 0
    assert rule["coupling_map"] == [[0, 1], [1, 0], [1, 2], [2, 1]]
    # 用的是磁盘上的旧值，而非抓取结果
    assert rule["num_qubits"] == 5


def test_stale_disk_cache_triggers_refresh(cache_env, monkeypatch):
    # 磁盘缓存超过 1 天 → 刷新并覆盖写盘
    spy = _FetchSpy(result=_topology(num_qubits=9, tag="v2"))
    monkeypatch.setattr(topo, "fetch_topology", spy)
    path = topo.cache_path("lqcloud")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"FakeDevice": {**_topology(), "fetched_at": _iso(2.0)}}),
        encoding="utf-8",
    )

    rule = topo.get_device_rule("lqcloud", "FakeDevice")

    assert spy.calls == 1
    assert rule["num_qubits"] == 9
    refreshed = json.loads(path.read_text(encoding="utf-8"))["FakeDevice"]
    assert refreshed["raw"] == {"tag": "v2"}
    # 刷新后时间戳应是"刚刚"
    assert topo._age_days(refreshed) < 0.01


def test_custom_max_age_days_is_honoured(cache_env, monkeypatch):
    # max_age_days 可收紧：0.5 天时，1 小时前的缓存仍然新鲜
    spy = _FetchSpy(result=_topology())
    monkeypatch.setattr(topo, "fetch_topology", spy)
    path = topo.cache_path("lqcloud")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"FakeDevice": {**_topology(), "fetched_at": _iso(1 / 24)}}),
        encoding="utf-8",
    )

    assert topo.get_device_rule("lqcloud", "FakeDevice", max_age_days=0.5)
    assert spy.calls == 0
    # 收紧到 1 小时以内则必须刷新
    assert topo.get_device_rule("lqcloud", "FakeDevice", max_age_days=1 / 48)
    assert spy.calls == 1


def test_memory_cache_avoids_disk_and_network(cache_env, monkeypatch):
    # 会话内存缓存：命中后即使磁盘文件被删也不再抓取
    spy = _FetchSpy(result=_topology())
    monkeypatch.setattr(topo, "fetch_topology", spy)

    first = topo.get_device_rule("lqcloud", "FakeDevice")
    topo.cache_path("lqcloud").unlink()
    second = topo.get_device_rule("lqcloud", "FakeDevice")

    assert spy.calls == 1
    assert first == second


def test_memory_cache_returns_copy_not_shared_dict(cache_env, monkeypatch):
    # 返回值必须是副本：调用方回填 initial_layout / 原地改 coupling_map 都不能污染缓存
    # （Routing 步骤会往传入的 device_rule 里写 initial_layout）。
    spy = _FetchSpy(result=_topology())
    monkeypatch.setattr(topo, "fetch_topology", spy)

    rule = topo.get_device_rule("lqcloud", "FakeDevice")
    rule["initial_layout"] = [3, 4, 5]
    rule["coupling_map"].append([9, 9])

    again = topo.get_device_rule("lqcloud", "FakeDevice")
    assert "initial_layout" not in again
    assert again["coupling_map"] == [[0, 1], [1, 0], [1, 2], [2, 1]]


def test_fetch_failure_falls_back_to_stale_cache(cache_env, monkeypatch):
    # 抓取失败但有陈旧缓存 → 降级沿用 + RuntimeWarning（离线仍可路由）
    spy = _FetchSpy(error=RuntimeError("network down"))
    monkeypatch.setattr(topo, "fetch_topology", spy)
    path = topo.cache_path("lqcloud")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"FakeDevice": {**_topology(), "fetched_at": _iso(10.0)}}),
        encoding="utf-8",
    )

    with pytest.warns(RuntimeWarning, match="沿用过期缓存"):
        rule = topo.get_device_rule("lqcloud", "FakeDevice")

    assert spy.calls == 1
    assert rule["coupling_map"] == [[0, 1], [1, 0], [1, 2], [2, 1]]


def test_fetch_failure_without_cache_returns_empty_rule(cache_env, monkeypatch):
    # 既无缓存又抓取失败 → 空 DeviceRule + 告警，绝不抛异常中断编译
    spy = _FetchSpy(error=RuntimeError("no credentials"))
    monkeypatch.setattr(topo, "fetch_topology", spy)

    with pytest.warns(RuntimeWarning, match="无法获取"):
        rule = topo.get_device_rule("lqcloud", "FakeDevice")

    assert rule == {}
    assert not topo.cache_path("lqcloud").exists()


def test_unimplemented_provider_returns_empty_rule(cache_env):
    # 未实现拓扑发现的 provider：不联网、返回空规则（编译器据此跳过路由）
    with pytest.warns(RuntimeWarning, match="暂未实现拓扑发现"):
        assert topo.get_device_rule("guodun", "SomeDevice") == {}


def test_empty_provider_or_device_returns_empty(cache_env):
    # 参数缺失时静默返回空规则（不告警：调用方本来就没指定设备）
    assert topo.get_device_rule("", "FakeDevice") == {}
    assert topo.get_device_rule("lqcloud", "") == {}


def test_force_refresh_bypasses_both_caches(cache_env, monkeypatch):
    # force_refresh：跳过内存与磁盘缓存，强制重新抓取
    spy = _FetchSpy(result=_topology(tag="v1"))
    monkeypatch.setattr(topo, "fetch_topology", spy)
    topo.get_device_rule("lqcloud", "FakeDevice")
    assert spy.calls == 1

    spy.result = _topology(num_qubits=7, tag="v2")
    rule = topo.get_device_rule("lqcloud", "FakeDevice", force_refresh=True)

    assert spy.calls == 2
    assert rule["num_qubits"] == 7


def test_device_name_with_provider_prefix_is_normalized(cache_env, monkeypatch):
    # "lqcloud::FakeDevice" 与 "FakeDevice" 命中同一条缓存
    spy = _FetchSpy(result=_topology())
    monkeypatch.setattr(topo, "fetch_topology", spy)

    prefixed = topo.get_device_rule("lqcloud", "lqcloud::FakeDevice")
    plain = topo.get_device_rule("lqcloud", "FakeDevice")

    assert spy.calls == 1
    assert prefixed == plain
    assert list(json.loads(topo.cache_path("lqcloud").read_text(encoding="utf-8"))) == [
        "FakeDevice"
    ]


def test_corrupt_cache_file_is_treated_as_missing(cache_env, monkeypatch):
    # 缓存文件损坏（半截 JSON）→ 按无缓存处理并重新抓取，不抛异常
    spy = _FetchSpy(result=_topology())
    monkeypatch.setattr(topo, "fetch_topology", spy)
    path = topo.cache_path("lqcloud")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"FakeDevice": {"coupling_map": [[0, 1]', encoding="utf-8")

    rule = topo.get_device_rule("lqcloud", "FakeDevice")

    assert spy.calls == 1
    assert rule["num_qubits"] == 5
    # 重新写盘后文件应是合法 JSON
    assert json.loads(path.read_text(encoding="utf-8"))["FakeDevice"]["num_qubits"] == 5


def test_missing_fetched_at_is_treated_as_expired(cache_env, monkeypatch):
    # 没有时间戳的条目视为过期 → 刷新
    spy = _FetchSpy(result=_topology(tag="v2"))
    monkeypatch.setattr(topo, "fetch_topology", spy)
    path = topo.cache_path("lqcloud")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"FakeDevice": {k: v for k, v in _topology().items() if k != "raw"}}),
        encoding="utf-8",
    )

    topo.get_device_rule("lqcloud", "FakeDevice")
    assert spy.calls == 1


def test_clear_cache_removes_memory_and_disk(cache_env, monkeypatch):
    # 清缓存：内存必清，disk=True 时删掉缓存文件
    spy = _FetchSpy(result=_topology())
    monkeypatch.setattr(topo, "fetch_topology", spy)
    topo.get_device_rule("lqcloud", "FakeDevice")
    assert topo.cache_path("lqcloud").is_file()

    topo.clear_topology_cache()

    assert not topo.cache_path("lqcloud").exists()
    assert topo._MEMORY_CACHE == {}
    # 清完后下一次调用必须重新抓取
    topo.get_device_rule("lqcloud", "FakeDevice")
    assert spy.calls == 2


def test_clear_cache_single_device_keeps_others(cache_env, monkeypatch):
    # 只清单个设备：同 provider 的其他设备条目保留
    spy = _FetchSpy(result=_topology())
    monkeypatch.setattr(topo, "fetch_topology", spy)
    topo.get_device_rule("lqcloud", "DeviceA")
    spy.result = {**_topology(), "device": "DeviceB"}
    topo.get_device_rule("lqcloud", "DeviceB")

    topo.clear_topology_cache(provider="lqcloud", device="DeviceA")

    remaining = json.loads(topo.cache_path("lqcloud").read_text(encoding="utf-8"))
    assert list(remaining) == ["DeviceB"]
    assert ("lqcloud", "DeviceA") not in topo._MEMORY_CACHE
    assert ("lqcloud", "DeviceB") in topo._MEMORY_CACHE


def test_readonly_cache_dir_does_not_break_compile(cache_env, monkeypatch):
    # 缓存目录不可写（只读 site-packages 场景）→ 只告警，本次仍返回可用规则
    spy = _FetchSpy(result=_topology())
    monkeypatch.setattr(topo, "fetch_topology", spy)
    monkeypatch.setattr(topo, "_save_cache", lambda provider, data: False)

    rule = topo.get_device_rule("lqcloud", "FakeDevice")

    assert rule["coupling_map"] == [[0, 1], [1, 0], [1, 2], [2, 1]]
    # 写盘失败不影响内存缓存生效
    topo.get_device_rule("lqcloud", "FakeDevice")
    assert spy.calls == 1
