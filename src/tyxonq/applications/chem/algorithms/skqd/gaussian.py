"""固定深度 Gaussian 编译：单粒子矩阵拟合与解析梯度。"""

from dataclasses import dataclass
from operator import index

import numpy as np
from scipy.optimize import minimize

from ...chem_libs.fermionic_circuits import two_mode_fock_matrix


def _givens(theta, phi):
    c, s, phase = np.cos(theta), np.sin(theta), np.exp(1j * phi)
    matrix = np.array([[c, -phase * s], [phase.conjugate() * s, c]])
    dtheta = np.array([[-s, -phase * c], [phase.conjugate() * c, -s]])
    dphi = np.array([[0, -1j * phase * s], [-1j * phase.conjugate() * s, 0]])
    return matrix, dtheta, dphi


def _network_and_gradient(parameters, target, pairs):
    """前缀/后缀乘积复用计算 dG/dθ；优化的不是基态能量。"""
    n = len(target)
    phases = np.exp(1j * parameters[:n])
    factors = [np.diag(phases)]
    derivatives = []
    for j, (p, q) in enumerate(pairs):
        small, dtheta, dphi = _givens(*parameters[n + 2 * j:n + 2 * j + 2])
        factor = np.eye(n, dtype=complex)
        factor[np.ix_([p, q], [p, q])] = small
        factors.append(factor)
        derivatives.append((dtheta, dphi))
    prefixes = [np.eye(n, dtype=complex)]
    for factor in factors:
        prefixes.append(factor @ prefixes[-1])
    suffixes = [None] * len(factors)
    suffix = np.eye(n, dtype=complex)
    for j in reversed(range(len(factors))):
        suffixes[j] = suffix
        suffix = suffix @ factors[j]
    unitary = prefixes[-1]
    diff = unitary - target
    gradient = np.empty_like(parameters)
    for p in range(n):
        d = np.zeros((n, n), complex)
        d[p, p] = 1j * phases[p]
        gradient[p] = 2 * np.vdot(diff, suffixes[0] @ d).real / n
    for j, (p, q) in enumerate(pairs):
        for axis, small in enumerate(derivatives[j]):
            d = np.zeros((n, n), complex)
            d[np.ix_([p, q], [p, q])] = small
            du = suffixes[j + 1] @ d @ prefixes[j + 1]
            gradient[n + 2 * j + axis] = 2 * np.vdot(diff, du).real / n
    return float(np.vdot(diff, diff).real / n), gradient, unitary


@dataclass
class GaussianFit:
    target: np.ndarray
    unitary: np.ndarray
    parameters: np.ndarray
    pairs: tuple
    depth: int
    error: float
    initial_error: float
    converged: bool
    iterations: int
    message: str

    def append_to(self, circuit, modes):
        """直接追加所拟合的层，不把拟合矩阵再次分解为更深的精确线路。"""
        modes = tuple(modes)
        n = len(self.target)
        if len(modes) != n or any(b != a + 1 for a, b in zip(modes, modes[1:])):
            raise ValueError("GaussianFit requires consecutive modes matching target")
        for mode, angle in zip(modes, self.parameters[:n]):
            circuit.rz(mode, float(angle))
        for j, (p, q) in enumerate(self.pairs):
            matrix, _, _ = _givens(*self.parameters[n + 2 * j:n + 2 * j + 2])
            circuit.unitary(modes[p], modes[q], matrix=two_mode_fock_matrix(matrix))


def fit_gaussian_evolution(target, *, depth, max_iterations=2000, seed=24, tolerance=1e-10):
    """以 ||G-U||F²/n 为目标，用 L-BFGS-B 拟合交替奇偶相邻门网络。"""
    target = np.asarray(target, complex)
    n = len(target)
    if target.shape != (n, n) or not np.allclose(target.conj().T @ target, np.eye(n), atol=1e-10, rtol=0):
        raise ValueError("target must be unitary")
    depth, max_iterations = index(depth), index(max_iterations)
    if depth < 1 or max_iterations < 1 or tolerance <= 0:
        raise ValueError("depth, max_iterations and tolerance must be positive")
    pairs = tuple((p, p + 1) for layer in range(depth) for p in range(layer % 2, n - 1, 2))
    rng = np.random.default_rng(seed)
    initial = np.r_[np.angle(np.diag(target)), rng.normal(0, 0.02, 2 * len(pairs))]
    # 随机相位打破 θ=0 时 φ 的零梯度停滞；固定种子便于复核。
    initial[n + 1::2] = rng.uniform(-np.pi, np.pi, len(pairs))
    initial_error = _network_and_gradient(initial, target, pairs)[0]
    result = minimize(lambda x: _network_and_gradient(x, target, pairs)[:2], initial,
                      method="L-BFGS-B", jac=True,
                      options={"maxiter": max_iterations, "gtol": tolerance, "ftol": 1e-14})
    error, _, unitary = _network_and_gradient(result.x, target, pairs)
    return GaussianFit(target.copy(), unitary, result.x, pairs, depth, error, initial_error,
                       bool(result.success), int(result.nit), str(result.message))
