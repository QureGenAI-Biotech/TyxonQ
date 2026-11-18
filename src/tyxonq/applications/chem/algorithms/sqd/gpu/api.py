"""Single public entry point for SQD selected-CI solves.

Use ``backend='gpu'``/``'auto'`` for CUDA SBD → Metal SBD → PyTorch CPU,
or ``backend='cpu'`` for PyTorch CPU. An explicitly requested but unavailable
accelerator falls back to CPU with a warning; solver and numerical failures do not.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from tyxonq.numerics.accelerator import BackendUnavailableError, resolve_sqd_backend


def _default_device(backend: str) -> str:
    return {"sbd": "cuda:0", "metal": "metal:0", "pytorch": "cpu"}[backend]


def solve_sci_gpu(
    ci_strings: tuple[np.ndarray, np.ndarray],
    h1e: np.ndarray,
    eri: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
    *,
    backend: str = "auto",
    device: str | None = None,
    **options: Any,
) -> tuple[Any, dict[str, Any]]:
    """Solve using hardware-only CUDA SBD, Metal SBD, or PyTorch CPU dispatch."""
    resolution = resolve_sqd_backend(backend, device)
    selected = resolution.resolved
    if selected == "sbd":
        from .sbd_solver import solve_sci_sbd
        solve = solve_sci_sbd
    elif selected == "metal":
        from .sbd_solver import solve_sci_metal
        solve = solve_sci_metal
    else:
        from .solver import solve_sci_cuda
        solve = solve_sci_cuda
    result, report = solve(
        ci_strings, h1e, eri, norb, nelec, device_text=resolution.device, **options
    )
    report = dict(report)
    report["backend"] = selected
    report["requested_backend"] = backend
    report["resolved_backend"] = selected
    report["selection_reason"] = resolution.selection_reason
    report["precision_mode"] = resolution.precision_mode
    report["used_cpu_fallback"] = resolution.used_cpu_fallback
    report.setdefault("rdm_backend", resolution.rdm_backend)
    return result, report


def make_gpu_sci_solver(
    *,
    backend: str = "auto",
    device_ids: Any = None,
    report_callback: Callable[[dict[str, Any]], None] | None = None,
    worker_timeout: float = 600,
    **solver_options: Any,
) -> Callable[..., Any]:
    """Return a batch solver ``(batches, h1e, eri, norb, nelec) -> results``."""
    from .batch import _make_batch_solver

    resolution = resolve_sqd_backend(backend)
    selected = resolution.resolved
    options = dict(solver_options)
    if device_ids is None:
        options.setdefault("device_text", resolution.device)

    def resolved_report(record: dict[str, Any]) -> None:
        report = dict(record["report"])
        report.update(requested_backend=backend, resolved_backend=selected,
                      backend=selected, selection_reason=resolution.selection_reason,
                      precision_mode=resolution.precision_mode,
                      used_cpu_fallback=resolution.used_cpu_fallback)
        report.setdefault("rdm_backend", resolution.rdm_backend)
        if report_callback is not None:
            report_callback({**record, "report": report})

    return _make_batch_solver(
        selected, resolved_report if report_callback is not None else None,
        device_ids, int(worker_timeout), options
    )


__all__ = ["BackendUnavailableError", "make_gpu_sci_solver", "solve_sci_gpu"]
