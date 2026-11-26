"""SpinQ 拓扑发现与凭证透传测试。"""

from __future__ import annotations

from tyxonq.compiler.platforms.spinq import topology
from tyxonq.devices.hardware import topology as device_topology
from tyxonq.devices.hardware.spinq import driver


class _Platform:
    def __init__(self, config=None):
        self.config = config or {}
        self.machine_count = 2

    def available(self):
        return True


class _Backend:
    def __init__(self, platform):
        self.platform = platform

    def get_platform(self, name):
        return self.platform


def test_coupling_map_is_normalized(monkeypatch):
    platform = _Platform({"qubits": 3, "topology": {"coupling_map": [[0, 1], [1, 2]]}})
    monkeypatch.setattr(driver, "_resolve_credentials", lambda opts: {"username": "u", "keyfile": "k", "host": None})
    monkeypatch.setattr(driver, "_create_backend", lambda **kwargs: _Backend(platform))
    rule = topology.get_topology("spinq::triangulum_vp")
    assert rule["coupling_map"] == [[0, 1], [1, 0], [1, 2], [2, 1]]
    assert rule["num_qubits"] == 3
    assert rule["status"] == "active"


def test_official_platform_without_topology_stays_unrouted(monkeypatch):
    monkeypatch.setattr(driver, "_resolve_credentials", lambda opts: {"username": "u", "keyfile": "k", "host": None})
    monkeypatch.setattr(driver, "_create_backend", lambda **kwargs: _Backend(_Platform()))
    rule = topology.get_topology("gemini_vp")
    assert rule["coupling_map"] == []
    assert rule["num_qubits"] == 2
    assert rule["raw"] == {
        "machine_count": 2, "available": True, "active_qubits": None, "display_name": None
    }


def test_real_platform_fields_and_sqc_alias(monkeypatch):
    platform = _Platform()
    platform.max_bitnum = 25
    platform.coupling_map = [[0, 1], [1, 2]]
    platform.active_qubits = [0, 1, 2]
    platform.name = "SQC-25"
    monkeypatch.setattr(driver, "_resolve_credentials", lambda opts: {"username": "u", "keyfile": "k", "host": None})
    monkeypatch.setattr(driver, "_create_backend", lambda **kwargs: _Backend(platform))
    rule = topology.get_topology("SQC-25")
    assert rule["device"] == "sqc_25_vp"
    assert rule["num_qubits"] == 25
    assert rule["coupling_map"] == [[0, 1], [1, 0], [1, 2], [2, 1]]
    assert rule["raw"]["active_qubits"] == [0, 1, 2]


def test_get_device_rule_forwards_spinq_credentials(monkeypatch, tmp_path):
    calls = []

    def fake_fetch(provider, device, token=None, **opts):
        calls.append((provider, device, opts))
        return {"num_qubits": 2, "coupling_map": []}

    monkeypatch.setenv("TYXONQ_TOPOLOGY_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(device_topology, "fetch_topology", fake_fetch)
    device_topology.clear_topology_cache(provider="spinq", disk=False)
    rule = device_topology.get_device_rule(
        "spinq", "gemini_vp", username="user", keyfile="key", host="host", force_refresh=True
    )
    assert calls == [("spinq", "gemini_vp", {"username": "user", "keyfile": "key", "host": "host"})]
    assert rule["coupling_map"] == []
