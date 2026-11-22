"""Q7：Tully 绝热面通过 ASE Calculator 接入 VelocityVerlet。

依赖 ASE；预期运行时间 < 1 秒。
ASE 缺失时打印安装提示并正常退出；本示例不访问网络或真机。
"""

import importlib.util

import numpy as np


def main():
    if importlib.util.find_spec("ase") is None:
        print("ASE 未安装；请先安装 ase 后运行本示例。")
        return

    from ase import Atoms
    from ase.md.verlet import VelocityVerlet

    from tyxonq.applications.chem.dynamics.models.pes import tully_model
    from tyxonq.applications.chem.interfaces.dynamics_md import NonadiabaticASECalculator

    atoms = Atoms("H", positions=[[-2.0, 0.0, 0.0]])
    atoms.calc = NonadiabaticASECalculator(tully_model("A"), surface=0)
    print(f"initial energy: {atoms.get_potential_energy():.8f} eV")
    print("initial forces (eV/Å):", atoms.get_forces())

    dynamics = VelocityVerlet(atoms, timestep=0.01, logfile=None)
    initial_total = atoms.get_potential_energy() + atoms.get_kinetic_energy()
    dynamics.run(20)
    final_total = atoms.get_potential_energy() + atoms.get_kinetic_energy()
    print(f"final position (Å): {atoms.positions[0, 0]:.8f}")
    print(f"20-step total-energy drift: {abs(final_total - initial_total):.3e} eV")
    assert np.isfinite(atoms.positions).all()


if __name__ == "__main__":
    main()
