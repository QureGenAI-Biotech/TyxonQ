"""Q6：Tully 模型上的 Ehrenfest 与 FSSH 非绝热动力学。

仅核心依赖；预期运行时间约 2 秒。
验收锚点是解析 PES/NAC、能量守恒与可复现跳跃统计，而非实验数据。
"""

from tyxonq.applications.chem.dynamics.ehrenfest import run_ehrenfest
from tyxonq.applications.chem.dynamics.models.pes import tully_model
from tyxonq.applications.chem.dynamics.surface_hopping import run_fssh


def main():
    pes = tully_model("A")
    ehrenfest = run_ehrenfest(pes, x0=-5.0, v0=1.0, dt=0.01, n_steps=1000)
    drift = abs(ehrenfest.energies[-1] - ehrenfest.energies[0])
    print(f"Ehrenfest energy drift: {drift:.3e} Hartree")
    print("final adiabatic populations:", ehrenfest.adiabatic_populations[-1])

    hops = 0
    frustrated = 0
    for seed in range(8):
        trajectory = run_fssh(
            pes, -5.0, 1.0, 0, 0.01, 1000, seed=seed, decoherence=True
        )
        hops += sum(not hop["frustrated"] for hop in trajectory.hops)
        frustrated += sum(hop["frustrated"] for hop in trajectory.hops)
    print(f"FSSH ensemble: 8 trajectories, successful hops={hops}, frustrated={frustrated}")


if __name__ == "__main__":
    main()
