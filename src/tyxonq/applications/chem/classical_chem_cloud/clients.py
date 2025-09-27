"""Cloud classical computation client (unified for CPU/GPU/auto).

We do NOT offload VQE optimization; only heavy classical PySCF kernels and
pure-classical methods are sent to a remote service via HTTP.
"""

from __future__ import annotations

from typing import Dict, Any, Optional
import numpy as np
from openfermion import QubitOperator
import base64
import os
import tempfile
import json
from urllib import request as _urlreq
from urllib.error import URLError, HTTPError

from .config import CloudClassicalConfig


class TyxonQClassicalClient:
    """Unified classical cloud client.

    - device in {"cpu", "gpu", "auto"} is passed through to server payload
    - verbose=True returns additional outputs and artifacts
    """

    def __init__(self, device: str = "auto", config: CloudClassicalConfig | None = None):
        self.provider = "tyxonq"
        self.device = str(device or "auto").lower()
        self.config = config or CloudClassicalConfig()
        self.provider_config = self.config.get_provider_config(self.provider, self.device)

    # ---- HTTP helpers ----
    def _post(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        endpoint = str(self.provider_config.get("endpoint", "")).rstrip("/") + "/classical/compute"
        data = json.dumps(payload).encode("utf-8")
        req = _urlreq.Request(endpoint, data=data, headers={"Content-Type": "application/json"}, method="POST")
        timeout = int(self.provider_config.get("default_config", {}).get("timeout", 7200))
        try:
            with _urlreq.urlopen(req, timeout=timeout) as resp:
                raw = resp.read().decode("utf-8")
                return json.loads(raw)
        except (HTTPError, URLError) as e:
            raise RuntimeError(f"Classical server request failed: {e}")


    def submit_classical_calculation(self, task_spec: Dict[str, Any]) -> Dict[str, Any]:
        """Submit classical calculation to cloud (mocked locally).

        Supported methods: "hf_integrals", "fci", "ccsd", "dft", "mp2", "casscf".
        Accepts keys:
          - molecule_data, basis, active_space, method_options, classical_device, verbose
        """
        verbose = bool(task_spec.get("verbose", False))
        payload = dict(task_spec)
        payload["classical_device"] = self.device
        payload["verbose"] = verbose
        return self._post(payload)

    # retained for potential future serialization needs
    def _serialize_hamiltonian(self, h_qubit_op: QubitOperator) -> Dict[str, Any]:
        terms = {}
        for term, coeff in h_qubit_op.terms.items():
            key = str(term) if term else ""
            terms[key] = complex(coeff)
        return {"terms": terms}


__all__ = [
    "TyxonQClassicalClient",
]