"""
QCOS cloud task demo: submit a Bell-state circuit to a China Mobile WuYue
device through the TyxonQ `qcos` provider.

Setup:
    1. Create a Python 3.11 env.
    2. Download the unified WuYue SDK whl from the ecloud console
       (编程框架本地部署 / "Deploy SDK locally") and install:
           pip install wuyue-1.0-py3-none-any.whl
    3. Reinstall tyxonq (e.g. `pip install -e .`) so it picks up the new SDK.
    4. Export your credentials:
           export QCOS_ACCESS_KEY=...
           export QCOS_SECRET_KEY=...
       (Or pass `access_key=` / `secret_key=` directly to `run()`.)

Note: The new wuyue==1.0 whl pins `qiskit==1.4.3`. Use a dedicated env if you
need a newer qiskit elsewhere.
"""

from __future__ import annotations

import json
import os

import tyxonq as tq


def main() -> None:
    if not os.getenv("QCOS_ACCESS_KEY") or not os.getenv("QCOS_SECRET_KEY"):
        raise SystemExit(
            "Set QCOS_ACCESS_KEY and QCOS_SECRET_KEY (from the China Mobile "
            "ecloud console) before running this example."
        )

    devices = tq.api.list_devices(provider="qcos") if hasattr(tq, "api") else []
    print("known qcos devices:", json.dumps(devices, indent=2, ensure_ascii=False))

    c = tq.Circuit(2)
    c.h(0).cx(0, 1).measure_z(0).measure_z(1)

    # Submit synchronously: timeout > 0 makes the WuYue Runner poll until
    # the task finishes (or until timeout seconds elapse, max 3600).
    results = c.run(
        provider="qcos",
        device="WuYue-QPUSim-FullAmpSim",
        shots=100,
        timeout=100,
        wait_async_result=True,
    )

    print("result:", json.dumps(results, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
