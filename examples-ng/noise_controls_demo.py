import tyxonq as tq


def demo_noise_controls():
    # Show defaults
    status_before = tq.is_noise_enabled()
    cfg_before = tq.get_noise_config()
    print("Noise enabled:", status_before)
    print("Noise config:", cfg_before)

    # Enable global simulator noise (example: small depolarizing)
    tq.enable_noise(True, {"type": "depolarizing", "p": 0.01})

    # Build a tiny circuit
    c = tq.Circuit(2)
    c.H(0)
    c.CNOT(0, 1)
    c.measure_z(0)
    c.measure_z(1)

    # Prefer IR on local simulators; pass per-call noise to override if desired
    tasks = c.run(
        provider="local",
        device="statevector",
        shots=0,
        use_noise=True,
        noise={"type": "depolarizing", "p": 0.02},
    )

    # `devices.base.run` returns a list of task-like objects (even for single run)
    for t in tasks:
        print("Result:", t.details())


if __name__ == "__main__":
    demo_noise_controls()


