from __future__ import annotations

from typing import Any, Dict, List, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from . import Device
    from tyxonq.core.ir import Circuit


def device_job_plan(device: "Device", plan: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a segmented shot plan and aggregate results.

    Parameters:
        device: Device implementing `run`.
        plan: Dict with keys:
            - circuit: IR circuit to execute
            - segments: list of segments, each with a positive integer `shots`

    Returns:
        Aggregated result dict with keys:
            - expectations: summed expectations across segments
            - metadata: includes per-segment records and total shots
    """

    circuit: "Circuit" = plan["circuit"]
    segments: List[Dict[str, Any]] = plan.get("segments", [])
    per_segment: List[Dict[str, Any]] = []
    total_shots = 0
    agg_expectations: Dict[str, float] = {}

    for seg in segments:
        shots = int(seg.get("shots", 0))
        total_shots += shots
        res = device.run(circuit, shots=shots)
        per_segment.append({
            "shots": shots,
            "basis": seg.get("basis"),
            "wires": seg.get("wires"),
            "basis_map": seg.get("basis_map", {}),
            "metadata": res.get("metadata", {}),
        })
        for key, val in (res.get("expectations") or {}).items():
            agg_expectations[key] = agg_expectations.get(key, 0.0) + float(val)

    return {
        "expectations": agg_expectations,
        "metadata": {"per_segment": per_segment, "total_shots": total_shots},
    }


