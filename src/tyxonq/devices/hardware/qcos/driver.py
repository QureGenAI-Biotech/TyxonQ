"""QCOS driver for TyxonQ - connects to 移动云量子真机 via wuyue_plugin.

Submits Qiskit QuantumCircuit objects to the WuYue cloud platform using
wuyue_plugin.runner.Runner. No local QCOS Docker required.

Usage:
    from tyxonq import Circuit

    c = Circuit(2)
    c.h(0).cx(0, 1).measure_z(0).measure_z(1)

    results = c.run(
        provider="qcos",
        device="WuYue-QPUSim-FullAmpSim",
        shots=100,
        access_key="...",
        secret_key="...",
        sdk_code="...",
        timeout=100,
        wait_async_result=True,
    )
    print(results[0]["result"])
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union

logger = logging.getLogger(__name__)

_license_initialized = False


def _ensure_license(access_key: str, secret_key: str, sdk_code: str) -> None:
    """Initialize WuYue license once."""
    global _license_initialized
    if _license_initialized:
        return
    from wuyue.ecloudsdkcore.license.license import License
    License.init_license(sdk_code, access_key, secret_key)
    _license_initialized = True


def _get_credentials(opts: Dict[str, Any]) -> tuple:
    """Extract access_key, secret_key, sdk_code from opts or env vars.

    Returns:
        (access_key, secret_key, sdk_code)

    Raises:
        ValueError: If any credential is missing.
    """
    access_key = opts.pop("access_key", None) or os.getenv("QCOS_ACCESS_KEY")
    secret_key = opts.pop("secret_key", None) or os.getenv("QCOS_SECRET_KEY")
    sdk_code = opts.pop("sdk_code", None) or os.getenv("QCOS_SDK_CODE")

    missing = []
    if not access_key:
        missing.append("access_key (or QCOS_ACCESS_KEY env var)")
    if not secret_key:
        missing.append("secret_key (or QCOS_SECRET_KEY env var)")
    if not sdk_code:
        missing.append("sdk_code (or QCOS_SDK_CODE env var)")
    if missing:
        raise ValueError(f"QCOS credentials missing: {', '.join(missing)}")

    return access_key, secret_key, sdk_code


@dataclass
class QCOSTask:
    """Task wrapper for WuYue cloud submissions."""

    id: str
    device: str
    status: str = "queued"
    wuyue_result: Any = field(default=None, repr=False)
    _runner: Any = field(default=None, repr=False)
    async_result: bool = True


def run(
    device: str,
    token: Optional[str] = None,
    *,
    source: Any = None,
    shots: Union[int, Sequence[int]] = 1024,
    **opts: Any,
) -> List[QCOSTask]:
    """Submit a Qiskit QuantumCircuit to WuYue cloud.

    Args:
        device: Device identifier (e.g., "WuYue-QPUSim-FullAmpSim").
        token: Unused, kept for interface compatibility.
        source: A Qiskit QuantumCircuit object (converted from TyxonQ IR
                by base.py before reaching here).
        shots: Number of measurement shots.
        **opts: Additional options including:
            - access_key, secret_key, sdk_code: Authentication credentials.
            - timeout: Wait timeout in seconds (default 100, 0 for async).
            - auto_retry: Enable automatic retry (default True).
            - task_name: Custom task name.
            - calculate_type: Device-specific parameter.

    Returns:
        List containing a single QCOSTask.
    """
    from wuyue_plugin.runner import Runner

    if source is None:
        raise ValueError("QCOS driver requires a circuit (source parameter)")

    # Strip provider prefix if present
    backend = device.split("::")[-1] if "::" in device else device

    # Get credentials and initialize license
    access_key, secret_key, sdk_code = _get_credentials(opts)
    _ensure_license(access_key, secret_key, sdk_code)

    # Handle batch shots
    if isinstance(shots, (list, tuple)):
        shots = int(shots[0])
    else:
        shots = int(shots)

    # Extract runner options
    auto_retry = opts.pop("auto_retry", True)
    timeout = opts.pop("timeout", 100)
    task_name = opts.pop("task_name", None)

    # Build runner config from remaining opts
    runner_kwargs = {}
    if task_name is not None:
        runner_kwargs["task_name"] = task_name
    # Pass through WuYue-specific options
    for key in ("calculate_type", "mapping_flag", "noise_type",
                "circuit_optimization", "qubit_mapping",
                "gate_decomposition", "is_amend"):
        if key in opts:
            runner_kwargs[key] = opts.pop(key)

    runner = Runner(access_key, secret_key, auto_retry)

    qc = source
    qubits = qc.num_qubits

    result = runner.run(
        qc=qc,
        shots=shots,
        qubits=qubits,
        device_id=backend,
        timeout=timeout,
        **runner_kwargs,
    )

    task = QCOSTask(
        id=result.get_task_id(),
        device=backend,
        status="completed" if result.get_success() else result.get_status(),
        wuyue_result=result,
        _runner=runner,
        async_result=(timeout == 0),
    )

    return [task]


def get_task_details(task: QCOSTask, token: Optional[str] = None) -> Dict[str, Any]:
    """Get task status and results.

    Args:
        task: QCOSTask object.
        token: Unused.

    Returns:
        Unified result dictionary.
    """
    result = task.wuyue_result

    # If we already have a completed result
    if result is not None and result.get_success():
        return {
            "result": result.get_counts() or {},
            "result_meta": {
                "task_id": result.get_task_id(),
                "device": result.get_device_id(),
                "prob": result.get_prob(),
                "amps": result.get_amps(),
                "raw": result.get_raw_result(),
            },
            "uni_status": "completed",
            "error": "",
        }

    # For async tasks or incomplete results, poll via runner
    if task._runner is not None and task.id:
        try:
            response = task._runner.get_task_result(task.id)
            status_code = response.data.task_status

            status_map = {
                1: "submitted",
                2: "queued",
                3: "running",
                4: "waiting",
                5: "completed",
                6: "failed",
            }
            uni_status = status_map.get(status_code, "unknown")

            counts = {}
            if status_code == 5 and response.data.out_counts:
                import ast
                counts = ast.literal_eval(response.data.out_counts)

            return {
                "result": counts,
                "result_meta": {
                    "task_id": task.id,
                    "device": task.device,
                    "status_code": status_code,
                },
                "uni_status": uni_status,
                "error": "" if status_code != 6 else "Task failed",
            }
        except Exception as e:
            logger.warning(f"Failed to poll task {task.id}: {e}")
            return {
                "result": {},
                "result_meta": {"task_id": task.id},
                "uni_status": "error",
                "error": str(e),
            }

    # Fallback: return whatever we have
    status = "unknown"
    if result is not None:
        status = "failed" if not result.get_success() else "completed"

    return {
        "result": (result.get_counts() if result else {}) or {},
        "result_meta": {"task_id": task.id, "device": task.device},
        "uni_status": status,
        "error": "" if status != "failed" else (result.get_status() if result else ""),
    }


def list_devices(token: Optional[str] = None, **kws: Any) -> List[str]:
    """List available WuYue cloud devices.

    Requires access_key and secret_key in kws or env vars.

    Returns:
        List of device names with "qcos::" prefix.
    """
    try:
        access_key = kws.pop("access_key", None) or os.getenv("QCOS_ACCESS_KEY")
        secret_key = kws.pop("secret_key", None) or os.getenv("QCOS_SECRET_KEY")
        if not access_key or not secret_key:
            logger.warning("QCOS credentials not set, cannot list devices")
            return []

        from wuyue_plugin.runner import Runner
        runner = Runner(access_key, secret_key)
        devices = runner.get_eng_list(**kws)
        return [f"qcos::{dev_id}" for dev_id in devices.values()]
    except Exception as e:
        logger.warning(f"Failed to list QCOS devices: {e}")
        return []
