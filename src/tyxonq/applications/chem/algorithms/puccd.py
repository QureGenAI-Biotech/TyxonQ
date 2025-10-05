


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
from pyscf.cc.addons import spatial2spin
from tyxonq.applications.chem.chem_libs.quantum_chem_library.ci_state_mapping import get_ci_strings
from pyscf.fci import cistring


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
        runtime: str = 'device',
        numeric_engine: str | None = None,
        run_fci: bool = False,
        classical_provider: str = 'local',
        classical_device: str = 'auto',
        # Optional PySCF-style direct molecule construction
        atom: object | None = None,
        basis: str = "sto-3g",
        unit: str = "Angstrom",
        charge: int = 0,
        spin: int = 0,
        **kwargs
    ) -> None:

        super().__init__(
            mol=mol,
            init_method=init_method,
            active_space=active_space,
            active_orbital_indices=active_orbital_indices,
            mo_coeff=mo_coeff,
            mode="hcb",
            runtime=runtime,
            numeric_engine=numeric_engine,
            run_fci=run_fci,
            atom=atom,
            basis=basis,
            unit=unit,
            charge=charge,
            spin=spin,
            classical_provider=classical_provider,
            classical_device=classical_device,
            **kwargs
        )
        self.ex_ops, self.param_ids, self.init_guess = self.get_ex_ops(self.t1, self.t2)

    # Legacy helpers not needed; excitations built in constructors
    def get_ex1_ops(self, t1: np.ndarray = None):
        raise NotImplementedError

    def get_ex2_ops(self, t2: np.ndarray = None):
        raise NotImplementedError

    def get_ex_ops(self, t1: np.ndarray = None, t2: np.ndarray = None) -> Tuple[List[Tuple], List[int], List[float]]:
        no, nv = self.no, self.nv
        if t2 is None:
            t2 = np.zeros((no, no, nv, nv))

        t2 = spatial2spin(t2)
        return generate_puccd_ex_ops(no, nv, t2)

    # ---- RDM in MO basis (spin-traced) specialized for pUCCD ----
    def make_rdm1(self, statevector=None, basis: str = "AO",**kwargs) -> np.ndarray:
    #     # Build CI vector under current ansatz (numeric statevector path)

        civector = self._statevector_to_civector(statevector)
        ci_strings = get_ci_strings(self.n_qubits, self.n_elec_s, self.mode)

        n_active = self.n_qubits
        rdm1_cas = np.zeros([n_active] * 2)
        for i in range(n_active):
            bitmask = 1 << i
            arraymask = (ci_strings & bitmask) == bitmask
            value = float(civector @ (arraymask * civector))
            rdm1_cas[i, i] = 2 * value
        rdm1 = self.embed_rdm_cas(rdm1_cas)
        if basis == "MO":
            return rdm1
        else:
            return rdm_mo2ao(rdm1, self.hf.mo_coeff)

    def make_rdm2(self,  statevector=None, basis: str = "AO",**kwargs) -> np.ndarray:
        civector = self._statevector_to_civector(statevector)
        ci_strings = get_ci_strings(self.n_qubits, self.n_elec_s, self.mode)

        n_active = self.n_qubits
        rdm2_cas = np.zeros([n_active] * 4)
        for p in range(n_active):
            for q in range(p + 1):
                maskq = 1 << q
                maskp = 1 << p
                maskpq = maskp + maskq
                arraymask = (ci_strings & maskq) == maskq
                if p == q:
                    value = float(civector @ (arraymask * civector))
                else:
                    arraymask &= ((~ci_strings) & maskp) == maskp
                    excitation = ci_strings ^ maskpq
                    addr = cistring.strs2addr(n_active, self.n_elec // 2, excitation)
                    value = float(civector @ (arraymask * civector[addr]))

                rdm2_cas[p, q, p, q] = rdm2_cas[q, p, q, p] = value
                if p == q:
                    continue
                arraymask = (ci_strings & maskpq) == maskpq
                value = float(civector @ (arraymask * civector))

                rdm2_cas[p, p, q, q] = rdm2_cas[q, q, p, p] = 2 * value
                rdm2_cas[p, q, q, p] = rdm2_cas[q, p, p, q] = -value
        rdm2_cas *= 2
        rdm2 = self.embed_rdm_cas(rdm2_cas)
        # no need to transpose
        if basis == "MO":
            return rdm2
        else:
            return rdm_mo2ao(rdm2, self.hf.mo_coeff)



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
