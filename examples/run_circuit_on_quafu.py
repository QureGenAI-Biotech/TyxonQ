"""Submit a 2-qubit Bell state to the BAQIS Quafu superconducting cloud.

Get a token at https://quafu-sqc.baqis.ac.cn/ (rotates every 30 days), then:

    export TYXONQ_QUAFU_TOKEN=<your token>
    python examples/run_circuit_on_quafu.py

Or pass it inline via tq.set_token(token, provider="quafu").

Note: explicit `c.compile(compile_engine="qiskit", output="qasm2")` is required.
The Chain API's no-arg `.compile()` is a no-op; without the explicit call the
quafu driver receives an IR Circuit object instead of a QASM string.
"""
from __future__ import annotations

import json
import os

import tyxonq as tq


def main() -> None:
    if "TYXONQ_QUAFU_TOKEN" not in os.environ:
        print(
            "Set TYXONQ_QUAFU_TOKEN first. "
            "Get a token at https://quafu-sqc.baqis.ac.cn/."
        )
        return

    devices = tq.api.list_devices(provider="quafu")
    print("online quafu devices:", json.dumps(devices, indent=2, ensure_ascii=False))
    if not devices:
        print("No devices available right now; try again later.")
        return

    chip = devices[0].split("::")[-1]
    print(f"using chip: {chip}")

    c = tq.Circuit(2)
    c.h(0).cx(0, 1).measure_z(0).measure_z(1)
    # Explicit pre-compile to QASM2 — see module docstring above.
    c.compile(compile_engine="qiskit", output="qasm2")

    results = (
        c.compile()  # no-op; sees the cached QASM2 from above
         .device(provider="quafu", device=chip, shots=1024)
         .postprocessing()
         .run()
    )

    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
