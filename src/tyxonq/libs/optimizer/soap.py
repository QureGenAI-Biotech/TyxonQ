from __future__ import annotations

import numpy as np
from scipy.optimize import OptimizeResult
from typing import Callable, Tuple, Any


def soap(
    fun: Callable[[np.ndarray, Any], float],
    x0: np.ndarray,
    args: Tuple[Any, ...] = (),
    u: float = 0.1,
    maxfev: int = 2000,
    atol: float = 1e-3,
    callback: Callable[[np.ndarray], None] | None = None,
    ret_traj: bool = False,
    **kwargs: Any,
) -> OptimizeResult:
    """Sequential Optimization with Approximate Parabola (SOAP) optimizer.

    SOAP is a lightweight, gradient-free optimization algorithm specifically designed
    for quantum variational algorithms. It combines parabolic line search with
    adaptive direction selection, making it robust to measurement noise and
    suitable for optimizing noisy objective functions common in quantum computing.

    Algorithm Overview:
        SOAP operates by performing sequential one-dimensional optimizations along
        carefully selected directions. For each direction, it fits a parabolic model
        to function evaluations and moves to the predicted minimum. The algorithm
        dynamically adapts its search directions based on optimization progress,
        occasionally testing promising average directions from recent iterations.

    Key Features:
        - **Gradient-free**: No gradient information required
        - **Noise-robust**: Effective with noisy function evaluations
        - **Adaptive directions**: Learns good search directions during optimization
        - **Parabolic modeling**: Uses second-order approximations for efficiency
        - **SciPy compatible**: Integrates seamlessly with scipy.optimize

    Args:
        fun (Callable): Objective function to minimize. Should accept a parameter
            vector and optional arguments, returning a scalar value.
        x0 (ndarray): Initial parameter guess. The optimization starts from this point.
        args (Tuple, optional): Additional arguments to pass to the objective function.
            Default is empty tuple.
        u (float, optional): Step size parameter controlling the scale of line searches.
            Larger values explore more aggressively but may overshoot minima.
            Default 0.1.
        maxfev (int, optional): Maximum number of function evaluations allowed.
            Algorithm terminates when this limit is reached. Default 2000.
        atol (float, optional): Absolute tolerance for convergence detection.
            Algorithm stops when average function value improvement over recent
            iterations falls below this threshold. Default 1e-3.
        callback (Callable, optional): Optional callback function called after each
            iteration with the current parameter vector. Useful for monitoring
            progress or logging. Default None.
        ret_traj (bool, optional): If True, include the full optimization trajectory
            in the result. Useful for analysis but increases memory usage. Default False.
        **kwargs: Additional keyword arguments (ignored for compatibility).

    Returns:
        OptimizeResult: Optimization result object containing:
            - x (ndarray): Final optimized parameter vector
            - fun (float): Final objective function value
            - nit (int): Number of iterations performed
            - nfev (int): Total number of function evaluations
            - success (bool): Always True (algorithm-specific)
            - fun_list (ndarray): Function values at each iteration
            - nfev_list (ndarray): Cumulative function evaluations at each iteration
            - trajectory (ndarray, optional): Full parameter trajectory (if ret_traj=True)

    Examples:
        >>> # Basic usage with a simple quadratic function
        >>> def quadratic(x):
        ...     return np.sum((x - 1)**2)  # Minimum at x = [1, 1, ...]
        >>> x0 = np.array([0.0, 0.0])
        >>> result = soap(quadratic, x0)
        >>> print(f"Optimized parameters: {result.x}")
        >>> print(f"Final value: {result.fun:.6f}")
        >>> print(f"Converged in {result.nit} iterations")
        
        >>> # VQE energy minimization with SOAP
        >>> def vqe_energy(params, circuit_func, hamiltonian):
        ...     circuit = circuit_func(params)
        ...     return circuit.expectation_value(hamiltonian, shots=1024)
        >>> 
        >>> # Initial guess for variational parameters
        >>> theta0 = np.random.random(n_params) * 0.1
        >>> 
        >>> # Optimize with custom tolerance and callback
        >>> def progress_callback(x):
        ...     print(f"Current energy: {vqe_energy(x, circuit_func, hamiltonian):.6f}")
        >>> 
        >>> result = soap(
        ...     vqe_energy,
        ...     theta0,
        ...     args=(circuit_func, hamiltonian),
        ...     u=0.05,  # Smaller steps for quantum circuits
        ...     maxfev=1000,
        ...     atol=1e-4,
        ...     callback=progress_callback,
        ...     ret_traj=True  # Keep trajectory for analysis
        ... )
        >>> 
        >>> optimal_energy = result.fun
        >>> optimal_params = result.x
        >>> convergence_history = result.fun_list
        
        >>> # Integration with scipy.optimize.minimize
        >>> from scipy.optimize import minimize
        >>> result = minimize(vqe_energy, theta0, method=soap, 
        ...                   args=(circuit_func, hamiltonian),
        ...                   options={'u': 0.05, 'maxfev': 1000})

    Algorithm Details:
        **Direction Selection**:
            1. Initially optimizes along coordinate directions (sorted by parameter magnitude)
            2. After cycling through all directions, tests "average directions" from recent progress
            3. Uses Powell's criterion to decide whether to add promising new directions

        **Line Search**:
            1. Evaluates function at {-u, 0, +u} along the current direction
            2. If minimum is at boundary, extends search to {-4u, -u, 0, +u} or {-u, 0, +u, +4u}
            3. Fits parabola to the evaluations and moves to predicted minimum
            4. Falls back to best sampled point if parabolic prediction is unreliable

        **Convergence**:
            - Stops when average improvement over recent iterations < atol
            - Or when maximum function evaluations (maxfev) is reached
            - Uses rolling window of length 2×n_params for convergence assessment

    Performance Notes:
        - **Function evaluation budget**: Typically uses 3-7 evaluations per iteration
        - **Scaling**: Performance degrades gracefully with parameter count
        - **Noise tolerance**: Robust to shot noise and measurement errors
        - **Memory usage**: O(n_params) storage, O(n_iterations) if ret_traj=True

    Comparison with Other Optimizers:
        - **vs. COBYLA**: Better convergence for smooth functions, fewer evaluations
        - **vs. Nelder-Mead**: More systematic direction selection, better for high dimensions
        - **vs. L-BFGS-B**: No gradient required, better for noisy functions
        - **vs. SPSA**: Deterministic behavior, more predictable convergence

    See Also:
        scipy.optimize.minimize: Standard optimization interface.
        tyxonq.libs.optimizer.interop: Integration utilities for scipy.
        tyxonq.applications.chem.algorithms: VQE and other variational algorithms.

    References:
        SOAP algorithm design is inspired by Powell's method and parabolic interpolation
        techniques, adapted for the specific challenges of quantum variational optimization.
    """

    nfev = 0
    nit = 0

    def _fun(_x: np.ndarray) -> float:
        nonlocal nfev
        nfev += 1
        return float(fun(_x, *args))

    trajectory = [x0.copy()]
    vec_list = []
    metric = np.abs(x0)
    for i in np.argsort(metric)[::-1]:
        vec = np.zeros_like(x0)
        vec[i] = 1
        vec_list.append(vec)
    vec_list_copy = vec_list.copy()

    e_list = [_fun(trajectory[-1])]
    nfev_list = [nfev]
    offset_list = []
    diff_list = []
    scale = float(u)

    while nfev < int(maxfev):
        if len(vec_list) != 0:
            vec = vec_list[0]
            vec_list = vec_list[1:]
        else:
            vec_list = vec_list_copy.copy()
            if len(trajectory) < len(vec_list_copy):
                continue
            p0 = trajectory[-1 - len(vec_list_copy)]
            f0 = e_list[-1 - len(vec_list_copy)]
            pn = trajectory[-1]
            fn = e_list[-1]
            fe = _fun(2 * pn - p0)
            if fe > f0:  # not promising
                continue
            average_direction = pn - p0
            if np.allclose(average_direction, 0):
                continue
            average_direction = average_direction / np.linalg.norm(average_direction)
            replace_idx = int(np.argmax(np.abs(diff_list[-len(vec_list_copy) :]))) if diff_list else 0
            df = float(np.abs(diff_list[-len(vec_list_copy) :][replace_idx])) if diff_list else 0.0
            if 2 * (f0 - 2 * fn + fe) * (f0 - fn - df) ** 2 > (f0 - fe) ** 2 * df:
                continue
            if vec_list:
                del vec_list[replace_idx]
            vec_list = [average_direction] + vec_list
            vec_list_copy = vec_list.copy()
            continue

        vec_normed = vec / np.linalg.norm(vec)
        x = [-scale, 0.0, scale]
        es = [None, e_list[-1], None]
        for j in [0, -1]:
            es[j] = _fun(trajectory[-1] + x[j] * vec_normed)
        if np.argmin(es) == 0:
            x = [-4 * scale, -scale, 0.0, scale]
            es = [None, es[0], es[1], es[2]]
            es[0] = _fun(trajectory[-1] + x[0] * vec_normed)
        elif np.argmin(es) == 2:
            x = [-scale, 0.0, scale, 4 * scale]
            es = [es[0], es[1], es[2], None]
            es[-1] = _fun(trajectory[-1] + x[-1] * vec_normed)
        a, b, c = np.polyfit(x, es, 2)
        if np.argmin(es) not in [0, 3]:
            offset = b / (2 * a)
        else:
            offset = -x[int(np.argmin(es))]
        offset_list.append(offset)
        trajectory.append(trajectory[-1] - offset * vec_normed)
        e_list.append(_fun(trajectory[-1]))
        diff_list.append(e_list[-1] - e_list[-2])
        nfev_list.append(nfev)

        if callback is not None:
            callback(np.copy(trajectory[-1]))

        nit += 1
        if len(e_list) > 2 * len(x0):
            if np.mean(e_list[-2 * len(x0) : -len(x0)]) - np.mean(e_list[-len(x0) :]) < float(atol):
                break

    y = _fun(trajectory[-1])
    res = OptimizeResult(
        fun=y,
        x=trajectory[-1],
        nit=nit,
        nfev=nfev,
        fun_list=np.array(e_list),
        nfev_list=np.array(nfev_list),
        success=True,
    )
    if ret_traj:
        res["trajectory"] = np.array(trajectory)
    return res


__all__ = ["soap"]


