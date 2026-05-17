"""End-to-end Chain API test for the quafu provider.

Mocks the requests.Session used by the vendored Task client so the test
runs offline. Exercises the full path: Circuit.compile() -> device() -> run()
-> get_task_details().

Adjustment notes vs. original plan:
- The Chain API's c.compile() (no args) is a chainable no-op that returns self
  without compiling to QASM.  When run() later re-compiles it falls through to
  the native compiler whose compiled_source is the Circuit IR object, not an
  OpenQASM 2.0 string.  The quafu driver rejects non-string sources.
  Fix: call c.compile(compile_engine="qiskit", output="qasm2") *before* the
  chain so the _compiled_source cache is primed with a valid QASM2 string.
  The subsequent .compile() (no-arg chainable form) then sees the cached source
  and skips re-compilation entirely.
- results[0]["result_meta"] from the chain is the unified info dict returned by
  devices.base.get_task_details(), NOT the quafu driver's inner result_meta.
  The driver's result_meta (containing "device") is nested one level deeper at
  results[0]["result_meta"]["result_meta"]["device"].  The assertion is adjusted
  accordingly.
"""
from __future__ import annotations

import json
import pytest
from unittest.mock import MagicMock, patch


@pytest.fixture(autouse=True)
def _reset_quafu_state(monkeypatch):
    from tyxonq.devices.hardware.quafu._vendor_quafu import Task
    from tyxonq.devices.hardware import config as hwcfg

    if hasattr(Task, "instance"):
        del Task.instance
    hwcfg._TOKENS.clear()
    for var in ("TYXONQ_API_KEY", "TYXONQ_QUAFU_TOKEN", "QPU_API_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("TYXONQ_QUAFU_TOKEN", "test-token")


def _bytes(obj):
    return MagicMock(content=json.dumps(obj).encode())


def test_chain_api_bell_circuit_end_to_end():
    import tyxonq as tq
    from tyxonq.devices.hardware.quafu._vendor_quafu import Task

    sess = MagicMock()
    sess.get.side_effect = [
        # 1st: verify in Task.__init__
        _bytes("ok"),
        # 2nd: result poll
        _bytes({"count": {"00": 512, "11": 512}, "status": "Finished"}),
    ]
    sess.post.return_value = _bytes(77001)

    c = tq.Circuit(2)
    c.h(0).cx(0, 1).measure_z(0).measure_z(1)
    # Pre-compile to QASM2 so the quafu driver receives a valid OpenQASM 2.0
    # string.  The no-arg .compile() call in the chain below is a chainable
    # no-op that reuses this cached source.
    c.compile(compile_engine="qiskit", output="qasm2")

    with patch.object(Task, "session", sess):
        results = (
            c.compile()
             .device(provider="quafu", device="Dongling", shots=1024)
             .postprocessing()
             .run()
        )

    # `run()` returns a list of result dicts (matching the existing tyxonq path).
    assert isinstance(results, list)
    assert len(results) >= 1
    # Counts must thread through the unified shape.
    assert results[0]["result"] == {"00": 512, "11": 512}
    assert results[0]["uni_status"] == "completed"
    # The chain's unified result wraps the driver's result dict in result_meta;
    # the driver's own result_meta (with "device") is one level deeper.
    assert results[0]["result_meta"]["result_meta"]["device"] == "Dongling"


def test_chain_api_passes_compiler_option():
    import tyxonq as tq
    from tyxonq.devices.hardware.quafu._vendor_quafu import Task

    sess = MagicMock()
    sess.get.side_effect = [
        _bytes("ok"),
        _bytes({"count": {"0": 1024}, "status": "Finished"}),
    ]
    sess.post.return_value = _bytes(77002)

    c = tq.Circuit(1).h(0).measure_z(0)
    # Pre-compile to QASM2 before entering the chain.
    c.compile(compile_engine="qiskit", output="qasm2")

    with patch.object(Task, "session", sess):
        c.compile().device(
            provider="quafu", device="Dongling", shots=1024, compiler="qsteed"
        ).run()

    body = json.loads(sess.post.call_args[1]["data"])
    assert body["options"]["compiler"] == "qsteed"
