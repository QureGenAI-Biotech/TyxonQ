"""Q4：XXZ 自旋链的 S(q,ω) 数值精确档。

仅核心依赖；预期运行时间 < 2 秒。
M1 只对照小体系 ED；实验 FMR 与 M2 真机定标不在本示例中执行。
"""

import numpy as np

from tyxonq.applications.chem.dynamics.models.spin_lattice import heisenberg_chain
from tyxonq.applications.chem.dynamics.sqw import compute_sqw


def main():
    model = heisenberg_chain(4, j_xy=0.5, j_z=0.0)
    times = np.linspace(0.0, 20.0, 201)
    result = compute_sqw(
        model, times, ops=("X",), n_momentum=8, n_frequency=300
    )
    print(f"schema: {result['schema']}")
    print("q        strongest ω    intensity")
    for iq, q in enumerate(result["q"]):
        iw = np.argmax(result["Sqw"][:, iq])
        print(f"{q:7.4f}   {result['omega'][iw]:10.4f}   {result['Sqw'][iw, iq]:9.4f}")


if __name__ == "__main__":
    main()
