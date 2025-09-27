"""Cloud-accelerated pure classical quantum chemistry methods.

This module provides cloud-accelerated versions of traditional quantum chemistry
methods like FCI, CCSD, DFT, etc., maintaining PySCF-compatible interfaces while
leveraging TyxonQ cloud infrastructure for massive speedups.
"""

from __future__ import annotations

from typing import Dict, Any, Optional, Union
import numpy as np
from pyscf.gto import Mole

from .config import CloudClassicalConfig, create_classical_client


class CloudClassicalMethodsWrapper:
    """Wrapper for cloud-accelerated classical quantum chemistry methods."""
    
    def __init__(
        self,
        molecule: Mole,
        classical_provider: str = "tyxonq",
        classical_device: str = "auto"
    ):
        self.molecule = molecule
        self.classical_provider = classical_provider
        self.classical_device = classical_device
        
        # Always initialize cloud client (this wrapper is for cloud execution)
        config = CloudClassicalConfig()
        device = classical_device if classical_device != "auto" else self._auto_select_device()
        provider = classical_provider if classical_provider == "tyxonq" else "tyxonq"
        self.cloud_client = create_classical_client(provider, device, config)
    
    def _auto_select_device(self) -> str:
        """Auto-select device based on molecule size and complexity."""
        n_atoms = self.molecule.natm
        n_electrons = self.molecule.nelectron
        
        # Simple heuristic for device selection
        if n_atoms > 10 or n_electrons > 20:
            return "gpu"  # Large systems benefit from GPU
        else:
            return "cpu"  # Small systems can use CPU efficiently
    
    def _prepare_molecule_data(self) -> Dict[str, Any]:
        """Prepare molecule data for cloud transmission."""
        return {
            "atom": self.molecule.atom,
            "basis": str(self.molecule.basis),
            "charge": self.molecule.charge,
            "spin": self.molecule.spin,
            "natm": self.molecule.natm,
            "nelectron": self.molecule.nelectron,
            "nao": self.molecule.nao
        }
    
    def fci(self, verbose: bool = False, **kwargs) -> float:
        """Cloud-accelerated Full Configuration Interaction (always server-side)."""
        task_spec = {
            "method": "fci",
            "molecule_data": self._prepare_molecule_data(),
            "method_options": kwargs,
            "n_atoms": self.molecule.natm,
            "classical_device": self.classical_device,
            "verbose": bool(verbose),
        }
        
        result = self.cloud_client.submit_classical_calculation(task_spec)
        return result["energy"]
    
    def ccsd(self, verbose: bool = False, **kwargs) -> float:
        """Cloud-accelerated Coupled Cluster Singles Doubles (always server-side)."""
        task_spec = {
            "method": "ccsd",
            "molecule_data": self._prepare_molecule_data(),
            "method_options": kwargs,
            "n_atoms": self.molecule.natm,
            "classical_device": self.classical_device,
            "verbose": bool(verbose),
        }
        
        result = self.cloud_client.submit_classical_calculation(task_spec)
        return result["energy"]
    
    def ccsd_t(self, verbose: bool = False, **kwargs) -> float:
        """Cloud-accelerated CCSD(T) (mapped to CCSD with triples hint)."""
        task_spec = {
            "method": "ccsd(t)",
            "molecule_data": self._prepare_molecule_data(),
            "method_options": {"triples": True, **kwargs},
            "n_atoms": self.molecule.natm,
            "classical_device": self.classical_device,
            "verbose": bool(verbose),
        }
        
        result = self.cloud_client.submit_classical_calculation(task_spec)
        return result["energy"]
    
    def dft(self, functional: str = "b3lyp", verbose: bool = False, **kwargs) -> float:
        """Cloud-accelerated Density Functional Theory (always server-side)."""
        task_spec = {
            "method": "dft",
            "molecule_data": self._prepare_molecule_data(),
            "method_options": {"functional": functional, **kwargs},
            "n_atoms": self.molecule.natm,
            "classical_device": self.classical_device,
            "verbose": bool(verbose),
        }
        
        result = self.cloud_client.submit_classical_calculation(task_spec)
        return result["energy"]
    
    def mp2(self, verbose: bool = False, **kwargs) -> float:
        """Cloud-accelerated Møller-Plesset perturbation theory (server-side)."""
        task_spec = {
            "method": "mp2",
            "molecule_data": self._prepare_molecule_data(),
            "method_options": kwargs,
            "n_atoms": self.molecule.natm,
            "classical_device": self.classical_device,
            "verbose": bool(verbose),
        }
        
        result = self.cloud_client.submit_classical_calculation(task_spec)
        return result["energy"]
    
    def casscf(self, ncas: int, nelecas: int, verbose: bool = False, **kwargs) -> float:
        """Cloud-accelerated Complete Active Space SCF (always server-side)."""
        task_spec = {
            "method": "casscf",
            "molecule_data": self._prepare_molecule_data(),
            "method_options": {"ncas": ncas, "nelecas": nelecas, **kwargs},
            "n_atoms": self.molecule.natm,
            "active_space": (nelecas, ncas),
            "classical_device": self.classical_device,
            "verbose": bool(verbose),
        }
        
        result = self.cloud_client.submit_classical_calculation(task_spec)
        return result["energy"]


def cloud_classical_methods(
    molecule: Mole,
    classical_provider: str = "tyxonq",
    classical_device: str = "auto"
) -> CloudClassicalMethodsWrapper:
    """Factory function for cloud-accelerated classical methods.
    
    Args:
        molecule: PySCF molecule object
        classical_provider: Provider for classical computation ("local", "tyxonq")
        classical_device: Device type ("auto", "gpu", "cpu")
    
    Returns:
        CloudClassicalMethodsWrapper with cloud-accelerated methods
    
    Example:
        >>> mol = gto.Mole()
        >>> mol.atom = 'H 0 0 0; H 0 0 0.74'
        >>> mol.basis = 'cc-pvdz'
        >>> mol.build()
        >>> 
        >>> # Cloud-accelerated methods
        >>> cloud_methods = cloud_classical_methods(mol, 
        ...     classical_provider="tyxonq", classical_device="gpu")
        >>> 
        >>> # Massive speedup for large systems
        >>> energy_fci = cloud_methods.fci()
        >>> energy_ccsd = cloud_methods.ccsd()
        >>> energy_dft = cloud_methods.dft(functional="b3lyp")
    """
    return CloudClassicalMethodsWrapper(molecule, classical_provider, classical_device)


__all__ = ["CloudClassicalMethodsWrapper", "cloud_classical_methods"]