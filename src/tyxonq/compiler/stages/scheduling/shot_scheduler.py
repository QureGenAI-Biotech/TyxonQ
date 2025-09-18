from __future__ import annotations

"""Shot scheduler: bridge measurement grouping to executable shot segments.

Why this exists
---------------
- Provide a single place to turn measurement grouping metadata into a concrete
  execution plan (segments), decoupling scheduling from device executors.
- Unify two common modes: explicit shot vectors from users and automatic
  distribution based on grouping weights.

What it does
------------
- Reads `Circuit.metadata["measurement_groups"]` (produced by the measurement
  rewrite stage) and assigns per-group shot budgets.
- Emits a plan with segments that carry `shots`, `basis`, `wires`, and
  `basis_map` to inform downstream executors of measurement settings reuse.

Why this is better than ad‑hoc scheduling
----------------------------------------
- Explicit and testable: plan is data, easy to inspect and unit test.
- Decoupled: devices/session only consume the plan, no embedded policy logic.
- Extensible: cost models (weights, rounding policy) can evolve without
  touching executors or groupers.
"""

# TODO(hardware-optimization): Device-aware scheduling
# - Adapt shot distribution using `DeviceRule` and calibration metadata:
#   * supports_batch / max_shots_per_job / queue policy
#   * basis-change overhead and readout-reset time per device
#   * parallel execution lanes / multiplexing constraints
#   * per-basis or per-wire setting-change cost models
# - Provide vendor-specific policies via plug-ins (e.g., ibm, braket) while
#   keeping this module policy-agnostic by accepting a cost function callback.
# - Feed back runtime metrics (segment durations, error bars) to refine weights.

from typing import Any, Dict, List, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from tyxonq.core.ir import Circuit
    from tyxonq.devices import DeviceRule


def schedule(circuit: "Circuit", shot_plan: List[int] | None = None, *, total_shots: int | None = None, device_rule: "DeviceRule" | None = None) -> Dict[str, Any]:
    """Create an execution plan (segments) for measurement execution.

    Modes
    -----
    - Explicit: When `shot_plan` is provided, returns one segment per entry.
    - Group-based: Otherwise, consumes measurement grouping metadata and
      distributes `total_shots` using `estimated_shots_per_group` as weights.

    Notes
    -----
    - Each segment carries measurement context (basis, wires, basis_map), which
      allows executors to reuse settings and batch runs efficiently.
    - Rounding is handled deterministically; the remainder is assigned to the
      last segment to preserve `sum(shots) == total_shots`.
    """

    segments: List[Dict[str, Any]] = []
    if shot_plan is not None:
        segments = [{"shots": int(s)} for s in shot_plan]
    else:
        groups = (circuit.metadata or {}).get("measurement_groups", [])  # type: ignore[union-attr]
        if groups and total_shots is not None:
            weights = [max(1, g.get("estimated_shots_per_group", 1)) for g in groups]
            wsum = sum(weights)
            rem = int(total_shots)
            for i, (g, w) in enumerate(zip(groups, weights)):
                shots_i = int(round(total_shots * (w / wsum))) if i < len(groups) - 1 else rem
                rem -= shots_i
                segments.append({
                    "shots": max(0, shots_i),
                    "basis": g.get("basis"),
                    "wires": g.get("wires"),
                    "basis_map": g.get("basis_map", {}),
                    "group_index": i,
                })
        else:
            # No groups: allocate all total_shots into a single segment (or 0)
            if total_shots is not None:
                segments = [{"shots": int(total_shots)}]
            else:
                segments = [{"shots": 0}]

    # Respect device constraints (basic): split segments by max_shots_per_job
    max_per_job = 0
    try:
        max_per_job = int((device_rule or {}).get("max_shots_per_job", 0))  # type: ignore[call-arg]
    except Exception:
        max_per_job = 0
    if max_per_job and max_per_job > 0:
        split_segments: List[Dict[str, Any]] = []
        for seg in segments:
            shots_left = int(seg.get("shots", 0))
            if shots_left <= 0:
                split_segments.append({**seg})
                continue
            while shots_left > 0:
                take = min(max_per_job, shots_left)
                split = dict(seg)
                split["shots"] = take
                split_segments.append(split)
                shots_left -= take
        segments = split_segments

    # Optional: assign batch ids if device supports batching
    try:
        supports_batch = bool((device_rule or {}).get("supports_batch", False))  # type: ignore[call-arg]
        max_segments_per_batch = int((device_rule or {}).get("max_segments_per_batch", 0))  # type: ignore[call-arg]
    except Exception:
        supports_batch = False
        max_segments_per_batch = 0
    if supports_batch and max_segments_per_batch and max_segments_per_batch > 0:
        for idx, seg in enumerate(segments):
            seg["batch_id"] = idx // max_segments_per_batch

    return {"circuit": circuit, "segments": segments}


class ShotSchedulerPass:
    """A pass that attaches a shot plan to compilation metadata.

    Skeleton behavior: no circuit changes, but validates/normalizes provided
    `shot_plan` option.
    """

    def execute_plan(self, circuit: "Circuit", device_rule: "DeviceRule", **opts) -> "Circuit":
        plan = opts.get("shot_plan")
        if plan is not None:
            if not isinstance(plan, list) or not all(isinstance(x, int) and x > 0 for x in plan):
                raise ValueError("shot_plan must be a list of positive integers")
        return circuit


