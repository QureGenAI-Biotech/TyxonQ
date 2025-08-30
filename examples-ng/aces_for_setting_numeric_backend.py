"""
Minimal ACE: demonstrate setting the global numeric backend and running a tiny circuit.

Usage:
  - python examples-ng/aces_for_setting_numeric_backend.py
  - Or import and call demo() / demo_with(backend_name)
"""

import tyxonq as tq


def _bell_circuit():
    c = tq.Circuit(2).H(0).CX(0, 1)
    c.measure_z(0).measure_z(1)
    return c


def demo_with(backend_name: str = "numpy"):
    tq.set_backend(backend_name)
    c = _bell_circuit()
    # Chain-style configuration with postprocessing specified in the chain
    results = (
        c.compile()
         .device(provider="local", device="statevector", shots=0)
         .postprocessing(method=None)
         .run()
    )
    print(f"[{backend_name}] results:", results)
    return results


def demo():
    out = []
    out.append(demo_with("numpy"))
    try:
        out.append(demo_with("pytorch"))
    except Exception:
        pass
    try:
        import cupynumeric  # noqa: F401
        out.append(demo_with("cupynumeric"))
    except Exception:
        pass
    return out


if __name__ == "__main__":
    demo()


