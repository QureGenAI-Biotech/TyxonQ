from __future__ import annotations

from typing import Any, Dict, Optional
from fastapi import FastAPI
from pydantic import BaseModel

from tyxonq.applications.chem.classical_chem_cloud.server import cpu_chem
from tyxonq.applications.chem.classical_chem_cloud.server  import gpu_chem


class MoleculeData(BaseModel):
    atom: str
    basis: str = "sto-3g"
    charge: int = 0
    spin: int = 0
    unit: str = "Angstrom"


class ClassicalRequest(BaseModel):
    method: str
    molecule_data: MoleculeData
    active_space: Optional[tuple[int, int]] = None
    active_orbital_indices: Optional[list[int]] = None
    method_options: Optional[dict] = None
    classical_device: str = "auto"
    verbose: bool = False


app = FastAPI()


def _route_backend(payload: Dict[str, Any]) -> Dict[str, Any]:
    dev = str(payload.get("classical_device", "auto")).lower()
    if dev == "gpu" or (dev == "auto" and gpu_chem.gpu_available):
        try:
            return gpu_chem.compute(payload)
        except Exception:
            # fallback to CPU
            pass
    return cpu_chem.compute(payload)


@app.post("/classical/compute")
def classical_compute(req: ClassicalRequest):
    payload: Dict[str, Any] = req.model_dump()
    return _route_backend(payload)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8009)

    #pip install fastapi uvicorn pydantic
    #pip install pyscf
    #pip install gpu4pyscf-cuda12x 
    #pip install cutensor-cu12
