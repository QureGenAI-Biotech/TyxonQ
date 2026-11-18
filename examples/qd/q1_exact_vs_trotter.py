"""Q1：单比特精确演化与 Trotter 线路档对照。

仅核心依赖；预期运行时间 < 2 秒。
解析金标准：H=ωX/2、|0⟩ 初态时 ⟨Z(t)⟩=cos(ωt)。
"""

import numpy as np

from tyxonq.applications.chem.dynamics.circuits import trotter_expectation_trajectory
from tyxonq.applications.chem.dynamics.exact import expectation_trajectory


def main():
    omega = 0.7
    ham = [(omega / 2, [("X", 0)])]
    observable = [(1.0, [("Z", 0)])]
    psi0 = np.array([1.0, 0.0], dtype=complex)
    times = np.linspace(0.0, 2.0, 9)

    exact = expectation_trajectory(ham, observable, psi0, times, 1)
    trotter = trotter_expectation_trajectory(
        ham, observable, times, 1, dt=0.02, provider="simulator", shots=2048
    )
    analytic = np.cos(omega * times)
    print("time     analytic    exact      Trotter")
    for t, ref, num, circ in zip(times, analytic, exact, trotter):
        print(f"{t:5.2f}   {ref:+.6f}   {num:+.6f}   {circ:+.6f}")
    print(f"exact vs analytic max error: {np.max(np.abs(exact - analytic)):.3e}")
    print("线路档残差同时包含 Trotter 误差与有限 shots 采样噪声。")


if __name__ == "__main__":
    main()
