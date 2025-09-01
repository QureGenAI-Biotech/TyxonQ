"""
Cloud task demo using TyxonQ cloud API:

- Legacy-style: submit OPENQASM source via facade and poll task
- Chain-style: build Circuit → compile(qiskit→qasm2) → submit → poll
"""

from __future__ import annotations

import os
import json
import time
import tyxonq as tq
import getpass


def bell_qasm() -> str:
    return (
        "OPENQASM 2.0;\n"
        "include \"qelib1.inc\";\n"
        "qreg q[2];\n"
        "creg c[2];\n"
        "h q[0];\n"
        "cx q[0],q[1];\n"
        "measure q[0] -> c[0];\n"
        "measure q[1] -> c[1];\n"
    )


def main():
    # token = os.getenv("TYXONQ_API_KEY")
    token = getpass.getpass("Enter your token: ")
    if token:
        tq.set_token(token, provider="tyxonq", device="homebrew_s2")
    # list devices
    devs = tq.api.list_devices(provider="tyxonq") if hasattr(tq, "api") else []
    print("devices:", json.dumps(devs, indent=2, ensure_ascii=False))

    # # --- Legacy-style: submit OPENQASM source ---
    res_legacy = tq.api.submit_task(provider="tyxonq", device="homebrew_s2", source=bell_qasm(), shots=100)
    legacy_tasks = res_legacy if isinstance(res_legacy, list) else [res_legacy]
    print("submitted (legacy):", legacy_tasks)

    time.sleep(5)
    for t in legacy_tasks:
        try:
            details = tq.api.get_task_details(t)
            print("legacy task details:", json.dumps(details, indent=2, ensure_ascii=False))
        except Exception as e:
            print("legacy detail error:", e)

    # --- Chain-style: Circuit → compile(qasm2) → submit ---
    c = tq.Circuit(2)
    c.h(0).cx(0, 1).measure_z(0).measure_z(1)

    #the tyxonq will auto comiple the circuit with the cloud qasm requirement
    #could let the machine wait for async result or just get back the reuslt 
    res_chain = c.compile().device(provider="tyxonq", device="homebrew_s2", shots=100).postprocessing().run(wait_async_result=True)
    print("submitted (chain):", res_chain)



if __name__ == "__main__":
    main()
