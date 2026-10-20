"""示例专用模型、分子和参考态；不进入可复用算法层。"""

import numpy as np

from tyxonq.applications.chem.algorithms.skqd import prepare_slater_reference, rotate_integrals


def near_fermi_reference(norb, nelec):
    """教程的近费米面网络：每种自旋各七个粒子数守恒旋转。

    按 IBM SKQD 教程 prepare_initial_state 的连边顺序重新实现。
    原始来源：迁移/qiskit/notebook/sample-based-krylov-quantum-diagonalization.ipynb。
    本实现改用 TyxonQ 的 q0 为矩阵最高位约定，保留相同 XXPlusYY 作用。
    """
    if norb < 8 or any(k < 3 or k + 3 >= norb for k in nelec):
        raise ValueError("seven-gate reference requires at least 3 occupied and 4 empty modes")
    circuit = prepare_slater_reference(norb, nelec)
    cosine = sine = np.sqrt(0.5)
    gate = np.eye(4)
    # Qiskit 的矩阵按 q1,q0 排列；交换中间两个基矢后，非对角元符号对调。
    gate[1:3, 1:3] = [[cosine, -sine], [sine, cosine]]
    for spin, occupied in enumerate(nelec):
        pairs = [(j, j + 1) for i in range(3) for j in range(occupied - i - 1, occupied + i, 2)]
        pairs.append((pairs[-1][1], pairs[-1][1] + 1))
        for p, q in pairs:
            circuit.unitary(spin * norb + p, spin * norb + q, matrix=gate)
    return circuit


def build_problem(config):
    """返回完整活性空间积分和能量常数；不在算法层固定具体化学体系。"""
    model = config["model"]
    kind, n = model["kind"], int(model["norb"])
    nelec = tuple(model["nelec"])
    h, g, c = np.zeros((n, n)), np.zeros((n,) * 4), np.eye(n)
    offset, blocks = 0.0, []
    if kind == "siam":
        for p in range(n - 1):
            h[p, p + 1] = h[p + 1, p] = -model["hopping"]
        h[0, 1] = h[1, 0] = -model["hybridization"]
        h[0, 0], g[0, 0, 0, 0] = -model["onsite"] / 2, model["onsite"]
        _, bath = np.linalg.eigh(h[1:, 1:])
        c[1:, 1:] = bath
        impurity = (n - 1) // 2
        order = list(range(1, impurity + 1)) + [0] + list(range(impurity + 1, n))
        c = c[:, order]
        h, g = rotate_integrals(h, g, c)
        neighbors = sorted((p for p in range(n) if p != impurity), key=lambda p: abs(h[p, p]))[:3]
        blocks = [[impurity] + neighbors]
    elif kind == "four_impurity":
        # 四个 impurity 在正方形顶点，每个带独立的、已对角化的 bath。
        levels = np.asarray(model["bath_energies"], float)
        if levels.ndim != 1 or n != 4 * (len(levels) + 1):
            raise ValueError("four_impurity requires norb=4*(len(bath_energies)+1)")
        if np.any(np.abs(levels) > model["bandwidth"] / 2):
            raise ValueError("bath energies must lie inside the specified band")
        size, impurities = len(levels) + 1, []
        for site in range(4):
            start = site * size
            center = start + len(levels) // 2
            impurities.append(center)
            bath_modes = [p for p in range(start, start + size) if p != center]
            for p, energy in zip(bath_modes, levels):
                h[p, p] = energy
                h[p, center] = h[center, p] = model["hybridization"] * np.sqrt((model["bandwidth"] / 2) ** 2 - energy ** 2)
            h[center, center], g[center, center, center, center] = -model["onsite"] / 2, model["onsite"]
            nearest = sorted(bath_modes, key=lambda p: abs(h[p, p]))[:2]
            blocks.append([center] + nearest)
        for i in range(4):
            for j in range(i + 1, 4):
                coupling = model["diagonal_hopping"] if (i - j) % 2 == 0 else model["hopping"]
                h[impurities[i], impurities[j]] = h[impurities[j], impurities[i]] = coupling
    elif kind == "molecule":
        from pyscf import ao2mo, gto, mcscf, scf

        molecule = gto.M(atom=model["geometry"], basis=model["basis"], unit="Angstrom",
                         charge=model.get("charge", 0), spin=0, verbose=0)
        mean_field = scf.RHF(molecule).run(conv_tol=1e-11)
        if not mean_field.converged:
            raise RuntimeError("RHF did not converge")
        cas = mcscf.CASCI(mean_field, n, nelec)
        cas.ncore = int(model.get("frozen_core", 0))
        if 2 * cas.ncore + sum(nelec) != molecule.nelectron:
            raise ValueError("frozen core plus active electrons must match molecule")
        h, offset = cas.get_h1cas()
        g = ao2mo.restore(1, cas.get_h2cas(), n)
    elif kind == "fcidump":
        from pyscf import ao2mo
        from pyscf.tools import fcidump

        data = fcidump.read(model["path"], verbose=False)
        if (data["NORB"] != n or data["NELEC"] != sum(nelec)
                or data.get("MS2", 0) != nelec[0] - nelec[1]):
            raise ValueError("FCIDUMP active space does not match config")
        h, g, offset = data["H1"], ao2mo.restore(1, data["H2"], n), data.get("ECORE", 0.0)
    else:
        raise ValueError(f"unknown model kind: {kind}")
    return {"h1e": h, "eri": g, "norb": n, "nelec": np.array(nelec),
            "energy_offset": float(offset), "original_rotation": c,
            "interaction_rotation": np.eye(n)}, blocks
