import tyxonq as tq


def demo_noise_controls():
    # Show defaults (before enabling)
    print("Noise enabled (before):", tq.is_noise_enabled())
    print("Noise config (before):", tq.get_noise_config())

    # Enable global simulator noise (example: depolarizing)
    tq.enable_noise(True, {"type": "depolarizing", "p": 0.05})
    try:
        print("Noise enabled (after):", tq.is_noise_enabled())
        print("Noise config (after):", tq.get_noise_config())

        # Build a circuit that yields non-zero Z expectation and is touched by a gate
        # RZ leaves Z expectation at +1 ideally, but triggers attenuation under noise
        c = tq.Circuit(1)
        c.rz(0, theta=0.5)
        c.measure_z(0)

        # Run without noise (device-level override)
        res_clean = (
            c.device(provider="local", device="statevector", shots=0, use_noise=False)
             .postprocessing(method=None)
             .run()
        )
        print("Result (no noise):", res_clean)

        # Run with noise (device-level)
        res_noisy = (
            c.device(
                provider="local",
                device="statevector",
                shots=0,
                use_noise=True,
                noise={"type": "depolarizing", "p": 0.05},
            )
             .postprocessing(method=None)
             .run()
        )
        print("Result (with noise):", res_noisy)
    finally:
        # 复原全局噪声开关：enable_noise 改写的是 devices.base 的模块级全局态，
        # 若不关闭会泄漏到同进程后续测试（如 tests_mol_valid 的 UCC 设备路径），
        # 导致其能量被去极化噪声污染。
        tq.enable_noise(False)


if __name__ == "__main__":
    demo_noise_controls()


