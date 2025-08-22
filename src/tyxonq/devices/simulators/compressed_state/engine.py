from __future__ import annotations

from typing import Any, Dict, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from ....core.ir import Circuit


class CompressedStateEngine:
    name = "compressed_state"
    capabilities = {"supports_shots": True}

    def run(self, circuit: "Circuit", shots: int | None = None, **kwargs: Any) -> Dict[str, Any]:
        shots = int(shots or 0)
        expectations: Dict[str, float] = {}
        for op in circuit.ops:
            if isinstance(op, (list, tuple)) and op and op[0] == "measure_z":
                q = int(op[1])
                expectations[f"Z{q}"] = expectations.get(f"Z{q}", 0.0) + float(shots)
        return {"expectations": expectations, "metadata": {"shots": shots}}

    def expval(self, circuit: "Circuit", obs: Any, **kwargs: Any) -> float:
        return 0.0


