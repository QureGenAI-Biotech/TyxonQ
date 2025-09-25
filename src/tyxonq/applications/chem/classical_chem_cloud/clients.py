"""Cloud classical computation clients for different providers and devices."""

from __future__ import annotations

from typing import Dict, Any
import numpy as np
from openfermion import QubitOperator

from .core import CloudClassicalConfig


class CloudClassicalClient:
    """Abstract base class for cloud classical computation clients."""
    
    def __init__(self, provider: str, device: str, config: CloudClassicalConfig = None):
        self.provider = provider
        self.device = device
        self.config = config or CloudClassicalConfig()
        self.provider_config = self.config.get_provider_config(provider, device)
    
    def submit_energy_calculation(self, task_spec: Dict[str, Any]) -> Dict[str, Any]:
        """Submit UCC energy calculation task."""
        raise NotImplementedError
    
    def submit_energy_grad_calculation(self, task_spec: Dict[str, Any]) -> Dict[str, Any]:
        """Submit UCC energy and gradient calculation task."""
        raise NotImplementedError
    
    def submit_classical_calculation(self, task_spec: Dict[str, Any]) -> Dict[str, Any]:
        """Submit pure classical calculation task (FCI, CCSD, DFT, etc.)."""
        raise NotImplementedError

    # ---- shared helpers ----
    def _hf_integrals_from_molecule(self, task_spec: Dict[str, Any]) -> Dict[str, Any]:
        """Compute HF and return MO-basis integrals via PySCF (mock cloud).

        Input task_spec expects keys:
          - molecule_data: {atom, basis, charge, spin}
          - active_space: Optional[Tuple[int, int]]
          - aslst: Optional[List[int]]
        """
        from pyscf import gto, scf  # type: ignore
        from tyxonq.applications.chem.chem_libs.hamiltonians_chem_library.hamiltonian_builders import (
            get_integral_from_hf as _get_integral_from_hf,
        )

        mdat = dict(task_spec.get("molecule_data", {}))
        atom = mdat.get("atom")
        basis = mdat.get("basis", "sto-3g")
        charge = int(mdat.get("charge", 0))
        spin = int(mdat.get("spin", 0))

        m = gto.Mole()
        m.atom = atom
        m.basis = basis
        m.charge = charge
        m.spin = spin
        m.build()

        hf = scf.RHF(m)
        hf.chkfile = None
        hf.verbose = 0
        hf.kernel()

        active_space = task_spec.get("active_space")
        aslst = task_spec.get("aslst")
        int1e, int2e, e_core = _get_integral_from_hf(hf, active_space=active_space, aslst=aslst)

        return {
            "int1e": np.asarray(int1e, dtype=float).tolist(),
            "int2e": np.asarray(int2e, dtype=float).tolist(),
            "e_core": float(e_core),
            "e_hf": float(getattr(hf, "e_tot", 0.0)),
            "mo_coeff": np.asarray(getattr(hf, "mo_coeff", None), dtype=float).tolist() if getattr(hf, "mo_coeff", None) is not None else None,
            "nelectron": int(getattr(m, "nelectron", 0)),
            "nao": int(getattr(m, "nao", 0)),
            "spin": int(getattr(m, "spin", 0)),
            "basis": str(getattr(m, "basis", "")),
        }


class TyxonQClassicalGPUClient(CloudClassicalClient):
    """Client for TyxonQ classical GPU computation (formerly ByteQC)."""
    
    def __init__(self, config: CloudClassicalConfig = None):
        super().__init__("tyxonq", "gpu", config)
    
    def submit_energy_calculation(self, task_spec: Dict[str, Any]) -> Dict[str, Any]:
        """Submit energy calculation to TyxonQ classical GPU cluster."""
        # Transform task to TyxonQ classical GPU format
        gpu_task = {
            "computation_type": "quantum_chemistry_ucc",
            "method": "ucc_energy",
            "parameters": {
                "circuit_parameters": task_spec["params"],
                "hamiltonian_data": self._serialize_hamiltonian(task_spec["h_qubit_op"]),
                "system_spec": {
                    "n_qubits": task_spec["n_qubits"],
                    "n_electrons": task_spec["n_elec_s"],
                    "excitation_operators": task_spec["ex_ops"],
                    "parameter_mapping": task_spec["param_ids"],
                    "fermion_mode": task_spec["mode"]
                }
            },
            "gpu_config": {
                **self.provider_config["default_config"]
            }
        }
        
        # Submit to TyxonQ classical GPU service
        result = self._submit_to_tyxonq_gpu(gpu_task)
        return {"energy": result["computed_energy"]}
    
    def submit_energy_grad_calculation(self, task_spec: Dict[str, Any]) -> Dict[str, Any]:
        """Submit energy and gradient calculation to TyxonQ classical GPU cluster."""
        gpu_task = {
            "computation_type": "quantum_chemistry_ucc",
            "method": "ucc_energy_gradient", 
            "parameters": {
                "circuit_parameters": task_spec["params"],
                "hamiltonian_data": self._serialize_hamiltonian(task_spec["h_qubit_op"]),
                "system_spec": {
                    "n_qubits": task_spec["n_qubits"],
                    "n_electrons": task_spec["n_elec_s"],
                    "excitation_operators": task_spec["ex_ops"],
                    "parameter_mapping": task_spec["param_ids"],
                    "fermion_mode": task_spec["mode"]
                },
                "gradient_method": "parameter_shift"
            },
            "gpu_config": {
                **self.provider_config["default_config"]
            }
        }
        
        result = self._submit_to_tyxonq_gpu(gpu_task)
        return {
            "energy": result["computed_energy"],
            "gradient": result["computed_gradient"]
        }
    
    def submit_classical_calculation(self, task_spec: Dict[str, Any]) -> Dict[str, Any]:
        """Submit pure classical calculation to TyxonQ classical GPU cluster."""
        method = task_spec.get("method", "fci")
        if method == "hf_integrals":
            return self._hf_integrals_from_molecule(task_spec)
        
        gpu_task = {
            "computation_type": "pure_classical_chemistry",
            "method": method,  # "fci", "ccsd", "dft", etc.
            "parameters": {
                "molecule_data": task_spec.get("molecule_data"),
                "basis_set": task_spec.get("basis", "sto-3g"),
                "active_space": task_spec.get("active_space"),
                "method_options": task_spec.get("method_options", {})
            },
            "gpu_config": {
                **self.provider_config["default_config"]
            }
        }
        
        result = self._submit_to_tyxonq_gpu(gpu_task)
        return {
            "energy": result.get("computed_energy"),
            "additional_properties": result.get("properties", {})
        }
    


    
    def _serialize_hamiltonian(self, h_qubit_op: QubitOperator) -> Dict[str, Any]:
        """Serialize QubitOperator for cloud transmission."""
        terms = {}
        for term, coeff in h_qubit_op.terms.items():
            key = str(term) if term else ""
            terms[key] = complex(coeff)
        return {"terms": terms}
    
    def _submit_to_tyxonq_gpu(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Submit task to TyxonQ classical GPU service (mock implementation)."""
        # Mock response for demonstration
        n_params = len(task["parameters"].get("circuit_parameters", []))
        return {
            "computed_energy": -1.0 + 0.1 * np.random.random(),
            "computed_gradient": (0.1 * np.random.random(n_params)).tolist()
        }


class TyxonQClassicalCPUClient(CloudClassicalClient):
    """Client for TyxonQ classical CPU computation (PySCF-based)."""
    
    def __init__(self, config: CloudClassicalConfig = None):
        super().__init__("tyxonq", "cpu", config)
    
    def submit_energy_calculation(self, task_spec: Dict[str, Any]) -> Dict[str, Any]:
        """Submit energy calculation to TyxonQ classical CPU cluster."""
        cpu_task = {
            "computation_type": "quantum_chemistry_ucc",
            "method": "ucc_statevector_cpu",
            "parameters": {
                "circuit_parameters": task_spec["params"],
                "hamiltonian_data": self._serialize_hamiltonian(task_spec["h_qubit_op"]),
                "system_spec": {
                    "n_qubits": task_spec["n_qubits"],
                    "n_electrons": task_spec["n_elec_s"],
                    "excitation_operators": task_spec["ex_ops"],
                    "parameter_mapping": task_spec["param_ids"],
                    "fermion_mode": task_spec["mode"]
                }
            },
            "cpu_config": {
                **self.provider_config["default_config"]
            }
        }
        
        result = self._submit_to_tyxonq_cpu(cpu_task)
        return {"energy": result["computed_energy"]}
    
    def submit_energy_grad_calculation(self, task_spec: Dict[str, Any]) -> Dict[str, Any]:
        """Submit energy and gradient calculation to TyxonQ classical CPU cluster."""
        cpu_task = {
            "computation_type": "quantum_chemistry_ucc",
            "method": "ucc_energy_gradient_cpu",
            "parameters": {
                "circuit_parameters": task_spec["params"],
                "hamiltonian_data": self._serialize_hamiltonian(task_spec["h_qubit_op"]),
                "system_spec": {
                    "n_qubits": task_spec["n_qubits"],
                    "n_electrons": task_spec["n_elec_s"],
                    "excitation_operators": task_spec["ex_ops"],
                    "parameter_mapping": task_spec["param_ids"],
                    "fermion_mode": task_spec["mode"]
                },
                "gradient_method": "finite_difference"
            },
            "cpu_config": {
                **self.provider_config["default_config"]
            }
        }
        
        result = self._submit_to_tyxonq_cpu(cpu_task)
        return {
            "energy": result["computed_energy"],
            "gradient": result["computed_gradient"]
        }
    
    def submit_classical_calculation(self, task_spec: Dict[str, Any]) -> Dict[str, Any]:
        """Submit pure classical calculation to TyxonQ classical CPU cluster."""
        method = task_spec.get("method", "fci")
        if method == "hf_integrals":
            return self._hf_integrals_from_molecule(task_spec)
        
        cpu_task = {
            "computation_type": "pure_classical_chemistry",
            "method": method,
            "parameters": {
                "molecule_data": task_spec.get("molecule_data"),
                "basis_set": task_spec.get("basis", "sto-3g"),
                "active_space": task_spec.get("active_space"),
                "method_options": task_spec.get("method_options", {})
            },
            "cpu_config": {
                "num_threads": 64,  # High parallelization for CPU
                **self.provider_config["default_config"]
            }
        }
        
        result = self._submit_to_tyxonq_cpu(cpu_task)
        return {
            "energy": result.get("computed_energy"),
            "additional_properties": result.get("properties", {})
        }
    
    def _serialize_hamiltonian(self, h_qubit_op: QubitOperator) -> Dict[str, Any]:
        """Serialize QubitOperator for cloud transmission."""
        terms = {}
        for term, coeff in h_qubit_op.terms.items():
            key = str(term) if term else ""
            terms[key] = complex(coeff)
        return {"terms": terms}
    
    def _submit_to_tyxonq_cpu(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Submit task to TyxonQ classical CPU service (mock implementation)."""
        # Mock response for demonstration
        n_params = len(task["parameters"].get("circuit_parameters", []))
        return {
            "computed_energy": -0.5 + 0.05 * np.random.random(),
            "computed_gradient": (0.05 * np.random.random(n_params)).tolist()
        }


__all__ = [
    "CloudClassicalClient",
    "TyxonQClassicalGPUClient", 
    "TyxonQClassicalCPUClient"
]