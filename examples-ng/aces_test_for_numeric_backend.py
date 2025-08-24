import tyxonq as tq


def _build_sample_circuit():
    c = tq.Circuit(2)
    c.H(0)
    c.CNOT(0, 1)
    # measurements can be auto-added by Circuit.run, but add explicitly here
    c.measure_z(0)
    c.measure_z(1)
    return c


def demo_set_numeric_backend(backend_name: str = "numpy"):
    # Set global numerics backend (numpy | pytorch | cupynumeric)
    tq.set_backend(backend_name)

    c = _build_sample_circuit()
    # Prefer IR on local simulators; shots=0 for fast deterministic expectations
    tasks = c.run(provider="local", device="statevector", shots=0)
    for t in tasks:
        print("Result:", t.details())
    return tasks


if __name__ == "__main__":
    # Example: switch between backends if available
    demo_set_numeric_backend("pytorch")
    print('pytorch backend avaiable')
    demo_set_numeric_backend("numpy")
    print('numpy backend avaiable')
    try:
        import cupynumeric
        demo_set_numeric_backend("cupynumeric")
        print('cupynumeric backend avaiable')
    except:
        pass


