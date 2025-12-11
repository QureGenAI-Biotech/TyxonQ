from __future__ import annotations

from typing import Any, Dict
from fastapi import FastAPI, Request
import json
from fastapi.responses import Response

from tyxonq.applications.chem.classical_chem_cloud.server import cpu_chem
from tyxonq.applications.chem.classical_chem_cloud.server  import gpu_chem


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
async def classical_compute(request: Request):
    # 直接接收 JSON 为 dict，放宽对字段类型的限制
    payload: Dict[str, Any] = await request.json()
    result = _route_backend(payload)
    # 直接用 json.dumps 生成字符串响应，避免 jsonable_encoder 的严格检查
    body = json.dumps(result)
    return Response(content=body, media_type="application/json")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8009)

    #pip install fastapi uvicorn pydantic
    #pip install pyscf
    #pip install gpu4pyscf-cuda12x 
    #pip install cutensor-cu12
