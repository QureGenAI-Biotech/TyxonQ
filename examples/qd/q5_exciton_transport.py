"""Q5：Frenkel 激子布居、MSD、扩散长度与迁移率标量管线。

仅核心依赖；预期运行时间 < 2 秒。
封闭无耗散链的 D 仅作诊断；实验迁移率对照按当前验收范围延后。
"""

import numpy as np

from tyxonq.applications.chem.dynamics.exciton_dynamics import (
    diffusion_length,
    diffusivity_from_msd,
    mobility_from_diffusivity,
    population_trajectory,
    rabi_frequency_two_site,
)
from tyxonq.applications.chem.dynamics.models.exciton import frenkel_chain


def main():
    hopping = 0.3
    model = frenkel_chain(5, hopping=hopping)
    times = np.linspace(0.0, 4.0, 41)
    result = population_trajectory(model, times, provider=None)
    populations = result["populations"]
    transport = diffusion_length(populations, times, tau=4.0, site0=result["site0"])
    diffusivity = diffusivity_from_msd(populations, times, site0=result["site0"])
    mobility = mobility_from_diffusivity(diffusivity, temperature_K=300.0)

    print(f"population sum at final time: {populations[-1].sum():.12f}")
    print(f"two-site resonant Rabi frequency: {rabi_frequency_two_site(hopping):.6f}")
    print(f"L_D (lattice sites): {transport['L_D']:.6f}")
    print(f"diagnostic D: {diffusivity:.6f}")
    print(f"Einstein mobility (atomic-unit model convention): {mobility:.6e}")


if __name__ == "__main__":
    main()
