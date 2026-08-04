from __future__ import annotations

import pytest

from tyxonq.devices.hardware.tyxonq import driver


class _FakeResponse:
    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict[str, object]:
        return {"id": "mock-task", "status": "submitted", "success": True}


def test_homebrew_s3_sends_qasm2_payload(monkeypatch: pytest.MonkeyPatch):
    """离线验证 homebrew_s3 客户端发送合同。"""

    received: dict[str, object] = {}

    def fake_post(url, *, json, headers, timeout):
        received.update(
            url=url,
            payload=json,
            headers=headers,
            timeout=timeout,
        )
        return _FakeResponse()

    monkeypatch.setattr(driver.requests, "post", fake_post)
    qasms = [
        f"OPENQASM 2.0; qreg q[1]; creg c[1]; // {basis}"
        for basis in ("X", "Y", "Z")
    ]

    tasks = driver.submit_task(
        "homebrew_s3",
        token="test-token",
        source=qasms,
        shots=1024,
    )

    assert received["url"].endswith("/api/v1/tasks/submit_task")
    assert received["headers"] == {"Authorization": "Bearer test-token"}
    assert received["timeout"] == 30
    assert received["payload"] == [
        {
            "device": "homebrew_s3",
            "shots": 1024,
            "source": qasm,
            "version": "1",
            "lang": "OPENQASM",
        }
        for qasm in qasms
    ]
    assert [task.id for task in tasks] == ["mock-task"]


@pytest.mark.parametrize(
    "source",
    [
        "OPENQASM 2.0; qreg q[1];",
        "OPENQASM 2.0; creg c[1];",
    ],
)
def test_homebrew_s3_rejects_qasm_without_registers(source: str):
    with pytest.raises(ValueError, match="homebrew_s3 source must be valid QASM2"):
        driver.submit_task(
            "homebrew_s3",
            token="test-token",
            source=source,
        )
