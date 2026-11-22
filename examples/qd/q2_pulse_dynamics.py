"""Q2：PulseProgram 波形演化及 lqcloud 方言翻译。

仅核心依赖；预期运行时间 < 2 秒。
默认只跑数值档并离线构造 waves；真机提交模板保留在注释中，不会自动执行。
"""

import numpy as np

from tyxonq.applications.chem.dynamics.models.spin_lattice import heisenberg_chain
from tyxonq.applications.chem.dynamics.pulse_dynamics import evolve_with_pulses
from tyxonq.compiler.platforms.lqcloud.lqwave_adapter import program_to_waves


def main():
    model = heisenberg_chain(2, j_xy=0.5, j_z=0.0, h=0.2)
    program = model.to_pulse_program(duration=32)
    program.set_device_params(
        qubit_freq=[5.0e9, 5.1e9], anharmonicity=[-330e6, -330e6]
    )

    numeric = evolve_with_pulses(program, provider=None, backend="numpy")
    state = np.asarray(numeric["state"])
    print(f"mode={numeric['mode']}, state shape={state.shape}")
    print(f"state norm={np.linalg.norm(state):.6f}")

    waves = program_to_waves(program)
    print("offline lqcloud channels:", sorted(waves))

    # 真机模板（消耗云资源，必须逐次获得用户确认后再取消注释）：
    # result = evolve_with_pulses(
    #     program, provider="lqcloud", device="AGate-100", shots=1024,
    #     result_format="counts", readout_correction=False,
    # )
    # print(result["task"])


if __name__ == "__main__":
    main()
