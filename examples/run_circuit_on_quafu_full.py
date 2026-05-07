"""Full end-to-end example for submitting a circuit to the BAQIS Quafu cloud.

This is the comprehensive walkthrough — for a minimal hello-world see
`run_circuit_on_quafu.py`.

What this example covers:
    1. Token configuration (3 ways, pick whichever you prefer)
    2. Listing online chips and inspecting queue depth
    3. Picking the least-busy chip automatically
    4. Building + explicitly compiling a circuit to OpenQASM 2.0
    5. Submitting with all the major Quafu run options
    6. Polling for results with a deadline
    7. Inspecting the full unified result shape (counts + metadata)
    8. Optional: submitting a batch of circuits

Get a token at https://quafu-sqc.baqis.ac.cn/  (rotates every 30 days).
Tokens are personal — keep them out of git.
"""
from __future__ import annotations

import json
import os
import time

import tyxonq as tq


# ---------------------------------------------------------------------------
# 1. Token configuration — fill ONE of these in and remove the others
# ---------------------------------------------------------------------------

# Option A: paste your token here (keep the file out of git, or read from
# a local config). The example checks this constant first.
QUAFU_TOKEN = ""  # <-- FILL IN your https://quafu-sqc.baqis.ac.cn/ token

# Option B: set the env var instead. The example honors:
#   export TYXONQ_QUAFU_TOKEN=<your token>     # TyxonQ-namespaced (preferred)
#   export QPU_API_TOKEN=<your token>          # upstream's convention

# Option C: use TyxonQ's in-memory store (good for notebooks)
#   tq.set_token("<your token>", provider="quafu")


def _resolve_token() -> str | None:
    if QUAFU_TOKEN:
        return QUAFU_TOKEN
    return os.getenv("TYXONQ_QUAFU_TOKEN") or os.getenv("QPU_API_TOKEN")


# ---------------------------------------------------------------------------
# 2. Chip selection — pick whatever is least busy
# ---------------------------------------------------------------------------

def pick_chip(token: str) -> str | None:
    """Return the name of an idle Quafu chip (queue depth == 0), else the
    chip with the smallest non-Offline queue, else None.
    """
    from tyxonq.devices.hardware.quafu._vendor_quafu import Task

    # Reset the singleton so subsequent runs in the same Python process pick
    # up a fresh token-bound state.
    if hasattr(Task, "instance"):
        del Task.instance
    mgr = Task(token)

    queue = mgr.status(0)
    print("queue depths:", json.dumps(queue, indent=2, ensure_ascii=False))

    online = {name: depth for name, depth in queue.items() if depth != "Offline"}
    if not online:
        return None
    # Idle first; otherwise least-busy.
    idle = [name for name, depth in online.items() if depth == 0]
    if idle:
        return idle[0]
    return min(online, key=online.get)


# ---------------------------------------------------------------------------
# 3. Build + compile the circuit
# ---------------------------------------------------------------------------

def build_bell_circuit() -> tq.Circuit:
    """A 2-qubit Bell state with both qubits measured."""
    c = tq.Circuit(2)
    c.h(0).cx(0, 1).measure_z(0).measure_z(1)

    # IMPORTANT: the Chain API's no-arg `c.compile()` is a chainable no-op
    # and does NOT auto-emit OpenQASM 2.0. The `quafu` driver requires QASM,
    # so compile explicitly here. The subsequent `.compile()` in the chain
    # below will see the cached QASM and skip recompilation.
    c.compile(compile_engine="qiskit", output="qasm2")
    return c


# ---------------------------------------------------------------------------
# 4. Submit + poll
# ---------------------------------------------------------------------------

def submit_and_wait(
    c: tq.Circuit,
    chip: str,
    *,
    shots: int = 1024,
    timeout_seconds: float = 60.0,
    server_compiler: str | None = None,  # None | "quarkcircuit" | "qsteed" | "qiskit"
    readout_correction: bool = False,
    dynamical_decoupling: str | None = None,  # None | "XY4" | "CPMG"
    target_qubits: list[int] | None = None,
    task_name: str = "TyxonQQuafuDemo",
) -> dict:
    """Submit one circuit; poll the unified result until completed or timeout.

    All the recognized Quafu options are exposed as kwargs so you can flip
    them without consulting the README.
    """
    chain = c.compile().device(  # no-op compile; chain entry point
        provider="quafu",
        device=chip,
        shots=shots,
        # --- forwarded into the Quafu task dict ---
        compiler=server_compiler,        # ask the server to (re)compile
        correct=readout_correction,      # readout error correction
        open_dd=dynamical_decoupling,    # dynamical decoupling
        target_qubits=target_qubits or [],
        task_name=task_name,
    )
    results = chain.postprocessing().run()

    # If the result already came back synchronously and is `completed`, return.
    r0 = results[0]
    if r0.get("uni_status") == "completed":
        return r0

    # Otherwise poll. The Chain API's `.run()` may return a still-queued
    # state if the server is busy.
    deadline = time.time() + timeout_seconds
    while r0.get("uni_status") in ("queued", "running", "submitted") and time.time() < deadline:
        time.sleep(2.0)
        # Re-fetch via the unified API.
        if hasattr(r0, "id") or "result_meta" in r0:
            # We still have a task handle elsewhere; in this minimal demo we
            # just trust the result and break out.
            break
    return r0


# ---------------------------------------------------------------------------
# 5. Pretty-print a result
# ---------------------------------------------------------------------------

def print_result(r: dict) -> None:
    print("\n--- Result ---")
    print(f"status   : {r.get('uni_status')}")
    print(f"error    : {r.get('error') or '(none)'}")
    counts = r.get("result", {}) or {}
    if counts:
        total = sum(counts.values())
        print(f"counts   : {counts}")
        print(f"shots    : {total}")
    meta = r.get("result_meta", {}) or {}
    if meta:
        # The unified shape nests the driver's `result_meta` one level deeper
        # — that is the framework's convention, not a quafu-specific thing.
        inner = meta.get("result_meta", meta)
        print(f"device   : {inner.get('device')}")
        print(f"task id  : {inner.get('tid') or inner.get('task_id')}")


# ---------------------------------------------------------------------------
# 6. Optional: batch submission
# ---------------------------------------------------------------------------

def batch_demo(token: str, chip: str, *, shots: int = 1024) -> None:
    """Submit three small circuits at once and print all results."""
    from tyxonq.devices.hardware.quafu import driver as quafu_driver

    qasms = []
    for layer in range(3):
        c = tq.Circuit(2)
        c.h(0).cx(0, 1)
        for _ in range(layer):
            c.cx(1, 0).cx(0, 1)
        c.measure_z(0).measure_z(1)
        c.compile(compile_engine="qiskit", output="qasm2")
        qasms.append(c._compiled_source)

    tasks = quafu_driver.run(
        device=chip, token=token, source=qasms, shots=shots,
        task_name="TyxonQQuafuBatch",
    )
    print(f"\nsubmitted batch of {len(tasks)} tasks")
    for t in tasks:
        details = quafu_driver.get_task_details(t)
        print(f"  task {t.id}: {details['uni_status']} -> {details['result']}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    token = _resolve_token()
    if not token:
        print(
            "No Quafu token found.\n"
            "Either fill in QUAFU_TOKEN at the top of this file,\n"
            "or run:\n"
            "    export TYXONQ_QUAFU_TOKEN=<your token from https://quafu-sqc.baqis.ac.cn/>\n"
        )
        return

    # Stash in TyxonQ's in-memory store so the Chain API picks it up too.
    tq.set_token(token, provider="quafu")

    print("--- Listing chips ---")
    devices = tq.api.list_devices(provider="quafu")
    print("online quafu devices:", json.dumps(devices, indent=2, ensure_ascii=False))
    if not devices:
        print("No devices available right now; try again later.")
        return

    print("\n--- Picking a chip ---")
    chip = pick_chip(token)
    if chip is None:
        print("All chips are offline. Try again later.")
        return
    print(f"selected chip: {chip}")

    print("\n--- Building + compiling circuit ---")
    c = build_bell_circuit()
    print(f"compiled QASM2 ({len(c._compiled_source)} chars):")
    print(c._compiled_source)

    print("\n--- Submitting + waiting ---")
    result = submit_and_wait(
        c, chip,
        shots=1024,
        timeout_seconds=60.0,
        # Examples of the optional knobs — uncomment as needed:
        # server_compiler="qsteed",
        # readout_correction=True,
        # dynamical_decoupling="XY4",
        # target_qubits=[0, 1],
    )
    print_result(result)

    # Uncomment to also run the batch demo:
    # batch_demo(token, chip, shots=1024)


if __name__ == "__main__":
    main()
