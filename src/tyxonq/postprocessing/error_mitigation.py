from __future__ import annotations

from typing import Any, Callable, Optional, Sequence, Tuple, Union, Dict, List
import numpy as np


def apply_zne(
    circuit: Any,
    executor: Callable[[Union[Any, Sequence[Any]]], Any],
    factory: Optional[Any] = None,
    scale_noise: Optional[Callable[[Any, float], Any]] = None,
    num_to_average: int = 1,
    **kws: Any,
) -> Any:
    """Minimal ZNE placeholder: average executor outputs.

    This is a skeleton; full ZNE with mitiq will be integrated later.
    """

    vals: List[float] = []
    for _ in range(max(1, int(num_to_average))):
        vals.append(float(executor(circuit)))
    return float(np.mean(vals))


def apply_dd(
    circuit: Any,
    executor: Callable[[Any], Any],
    rule: Optional[Union[Callable[[int], Any], List[str]]] = None,
    rule_args: Optional[Dict[str, Any]] = None,
    num_trials: int = 1,
    full_output: bool = False,
    ignore_idle_qubit: bool = True,
    fulldd: bool = False,
    iscount: bool = False,
) -> Union[float, Tuple[float, List[Any]], Dict[str, float], Tuple[Dict[str, float], List[Any]]]:
    """Minimal DD placeholder: average executor outputs without modifying circuit."""

    vals: List[Any] = []
    for _ in range(max(1, int(num_trials))):
        vals.append(executor(circuit))
    if iscount:
        # average dictionaries elementwise
        keys = set().union(*[d.keys() for d in vals])  # type: ignore
        out: Dict[str, float] = {}
        n = len(vals)
        for k in keys:
            out[k] = float(np.mean([float(v.get(k, 0)) for v in vals]))  # type: ignore
        return (out, [circuit] * len(vals)) if full_output else out
    else:
        mean_val = float(np.mean([float(v) for v in vals]))
        return (mean_val, [circuit] * len(vals)) if full_output else mean_val


def apply_rc(
    circuit: Any,
    executor: Callable[[Any], Any],
    num_to_average: int = 1,
    simplify: bool = True,
    iscount: bool = False,
    **kws: Any,
) -> Tuple[Any, List[Any]]:
    """Minimal RC placeholder: call executor multiple times and average outputs."""

    vals: List[Any] = []
    for _ in range(max(1, int(num_to_average))):
        vals.append(executor(circuit))
    if iscount:
        keys = set().union(*[d.keys() for d in vals])  # type: ignore
        out: Dict[str, float] = {}
        n = len(vals)
        for k in keys:
            out[k] = float(np.mean([float(v.get(k, 0)) for v in vals]))  # type: ignore
        return out, [circuit] * len(vals)
    else:
        return float(np.mean([float(v) for v in vals])), [circuit] * len(vals)


__all__ = ["apply_zne", "apply_dd", "apply_rc"]


