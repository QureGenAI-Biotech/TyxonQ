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
    """Sequential Optimization with Approximate Parabola (SOAP).

    Lightweight SciPy-compatible optimizer.
    Returns OptimizeResult with fields: x, fun, nit, nfev, fun_list, nfev_list, (optional) trajectory.
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


