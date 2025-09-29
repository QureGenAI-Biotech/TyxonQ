


from typing import Tuple, List, Union

import numpy as np
from pyscf.gto.mole import Mole
from pyscf.scf import RHF as _RHF
from openfermion.transforms import jordan_wigner

from tyxonq.core.ir.circuit import Circuit
from tyxonq.applications.chem.chem_libs.circuit_chem_library.ansatz_puccd import generate_puccd_ex_ops
from tyxonq.libs.hamiltonian_encoding.pauli_io import reverse_qop_idx, rdm_mo2ao
from tyxonq.applications.chem.chem_libs.hamiltonians_chem_library.hamiltonian_builders import (
    get_integral_from_hf,
    get_hop_from_integral,
)
from .ucc import UCC
from tyxonq.libs.circuits_library.qubit_state_preparation import get_circuit_givens_swap
from pyscf import fci as _fci


class PUCCD(UCC):
    """
    Run paired UCC calculation.
    The interfaces are similar to :class:`UCCSD <tencirchem.UCCSD>`.


    # todo: more documentation here and make the references right.
    # separate docstring examples for a variety of methods, such as energy()
    # also need to add a few comment on make_rdm1/2
    # https://arxiv.org/pdf/2002.00035.pdf
    # https://arxiv.org/pdf/1503.04878.pdf
    Paired UCC (pUCCD) aligned to new UCC base (device by default)."""

    def __init__(
        self,
        mol: Union[Mole, _RHF],
        init_method: str = "mp2",
        *,
        active_space: Tuple[int, int] | None = None,
        active_orbital_indices: List[int] | None = None,
        mo_coeff: np.ndarray | None = None,
        runtime: str | None = None,
        numeric_engine: str | None = None,
        run_hf: bool = True,
        run_mp2: bool = False,
        run_ccsd: bool = False,
        run_fci: bool = False,
    ) -> None:
        # RHF setup (robustly detect SCF object by attributes)
        if hasattr(mol, "kernel") and hasattr(mol, "mo_coeff"):
            hf = mol  # already an SCF object
        else:
            hf = _RHF(mol)
        if mo_coeff is not None:
            hf.mo_coeff = np.asarray(mo_coeff)
        hf.chkfile = None
        hf.verbose = 0
        if run_hf:
            hf.kernel()
        # integrals and core energy
        int1e, int2e, e_core = get_integral_from_hf(hf, active_space=active_space, active_orbital_indices=active_orbital_indices)
        if active_space is None:
            n_elec = int(getattr(hf.mol, "nelectron"))
            n_cas = int(getattr(hf.mol, "nao"))
        else:
            n_elec, n_cas = int(active_space[0]), int(active_space[1])
        no = n_elec // 2
        nv = n_cas - no
        # qubit Hamiltonian
        fop = get_hop_from_integral(int1e, int2e)
        n_qubits = 2 * n_cas
        hq = reverse_qop_idx(jordan_wigner(fop), n_qubits)
        na = no
        nb = n_elec - na
        # pUCCD excitations and init guess (pair excitations only)
        ex_ops, param_ids, init_guess = generate_puccd_ex_ops(no, nv, None)
        # Ensure contiguous param_ids for numeric engines
        unique_ids = np.unique(param_ids)
        id_map = {int(old): idx for idx, old in enumerate(unique_ids)}
        param_ids = [id_map[int(i)] for i in param_ids]
        init_guess = list(np.asarray(init_guess)[unique_ids])
        # map to base UCC
        super().__init__(
            n_qubits=n_qubits,
            n_elec_s=(na, nb),
            h_qubit_op=hq,
            runtime=str(runtime or ("numeric" if numeric_engine is not None else "device")),
            mode="fermion",
            ex_ops=ex_ops,
            param_ids=param_ids,
            init_state=None,
            decompose_multicontrol=False,
            trotter=False,
        )
        self.e_core = float(e_core)
        self.init_guess = np.asarray(init_guess, dtype=np.float64) if len(init_guess) > 0 else np.zeros(0, dtype=np.float64)
        self.numeric_engine = numeric_engine
        self._int1e = np.asarray(int1e)
        self._int2e = np.asarray(int2e)
        self.n_elec = int(n_elec)
        self.hf = hf

    # Convenience from_integral constructor
    @classmethod
    def from_integral(
        cls,
        int1e: np.ndarray,
        int2e: np.ndarray,
        n_elec: Union[int, Tuple[int, int]],
        *,
        runtime: str = "device",
        numeric_engine: str | None = None,
    ) -> "PUCCD":
        if isinstance(n_elec, int):
            assert n_elec % 2 == 0
            n_elec_s = (n_elec // 2, n_elec // 2)
        else:
            n_elec_s = (int(n_elec[0]), int(n_elec[1]))
        n_cas = int(len(int1e))
        no = int(n_elec_s[0])
        nv = n_cas - no
        ex_ops, param_ids, init_guess = generate_puccd_ex_ops(no, nv, None)
        unique_ids = np.unique(param_ids)
        id_map = {int(old): idx for idx, old in enumerate(unique_ids)}
        param_ids = [id_map[int(i)] for i in param_ids]
        init_guess = list(np.asarray(init_guess)[unique_ids])
        from openfermion.transforms import jordan_wigner as _jw
        fop = get_hop_from_integral(int1e, int2e)
        n_qubits = 2 * n_cas
        hq = reverse_qop_idx(_jw(fop), n_qubits)
        inst = cls.__new__(cls)
        UCC.__init__(
            inst,
            n_qubits=n_qubits,
            n_elec_s=(int(n_elec_s[0]), int(n_elec_s[1])),
            h_qubit_op=hq,
            runtime=("numeric" if numeric_engine is not None else runtime),
            mode="fermion",
            ex_ops=ex_ops,
            param_ids=param_ids,
            init_state=None,
            decompose_multicontrol=False,
            trotter=False,
        )
        inst.e_core = 0.0
        inst.init_guess = np.asarray(init_guess, dtype=np.float64) if len(init_guess) > 0 else np.zeros(0, dtype=np.float64)
        inst.numeric_engine = numeric_engine
        inst._int1e = np.asarray(int1e)
        inst._int2e = np.asarray(int2e)
        inst.n_elec = int(sum(n_elec_s))
        return inst

    # Legacy helpers not needed; excitations built in constructors
    def get_ex1_ops(self, t1: np.ndarray = None):
        raise NotImplementedError

    def get_ex2_ops(self, t2: np.ndarray = None):
        raise NotImplementedError

    def get_ex_ops(self, t1: np.ndarray = None, t2: np.ndarray = None) -> Tuple[List[Tuple], List[int], List[float]]:
        no, nv = self.no, self.nv
        if t2 is None:
            t2 = np.zeros((no, no, nv, nv))
        else:
            if t2.shape == (2 * no, 2 * no, 2 * nv, 2 * nv):
                t2 = t2[0::2, 0::2, 0::2, 0::2]
            else:
                assert t2.shape == (no, no, nv, nv)
        return generate_puccd_ex_ops(no, nv, t2)

    # ---- RDM in MO basis (spin-traced) specialized for pUCCD ----
    def make_rdm1(self, params=None, *, basis: str = "MO") -> np.ndarray:
        # Build CI vector under current ansatz (numeric statevector path)
        civ = np.asarray(self.civector(params), dtype=np.float64)
        n_orb = int(self.n_qubits // 2)
        rdm1_cas = _fci.direct_spin1.make_rdm1(civ, n_orb, self.n_elec_s)
        rdm1_mo = np.asarray(rdm1_cas, dtype=np.float64)
        if str(basis).upper() == "AO":
            return rdm_mo2ao(rdm1_mo, self.hf.mo_coeff)
        return rdm1_mo

    def make_rdm2(self, params=None, *, basis: str = "MO") -> np.ndarray:
        civ = np.asarray(self.civector(params), dtype=np.float64)
        n_orb = int(self.n_qubits // 2)
        rdm2_cas = _fci.direct_spin1.make_rdm12(civ, n_orb, self.n_elec_s)[1]
        rdm2_mo = np.asarray(rdm2_cas, dtype=np.float64)
        if str(basis).upper() == "AO":
            return rdm_mo2ao(rdm2_mo, self.hf.mo_coeff)
        return rdm2_mo

    def get_circuit(self, params=None, trotter=False, givens_swap=False) -> Circuit:
        """
        Get the circuit as TyxonQ ``Circuit`` object.

        Parameters
        ----------
        params: Tensor, optional
            The circuit parameters. Defaults to None, which uses the optimized parameter.
            If :func:`kernel` is not called before, the initial guess is used.
        trotter: bool, optional
            Whether Trotterize the UCC factor into Pauli strings.
            Defaults to False.
        givens_swap: bool, optional
            Whether return the circuit with Givens-Swap gates.

        Returns
        -------
        circuit: :class:`tc.Circuit`
            The quantum circuit.
        """
        if not givens_swap:
            return super().get_circuit(params, trotter=trotter)
        params = self._check_params_argument(params, strict=False)
        # givens-swap preparation (legacy helper)
        return get_circuit_givens_swap(params, self.n_qubits, self.n_elec, self.init_state)

    @property
    def e_puccd(self):
        """
        Returns pUCCD energy
        """
        return self.energy()
