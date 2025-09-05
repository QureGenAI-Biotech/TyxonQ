


from typing import Tuple, List, Union

import numpy as np
from pyscf.gto.mole import Mole
from pyscf.scf import RHF
from pyscf.cc.addons import spatial2spin
from pyscf.fci import cistring
from tyxonq import Circuit

from tyxonq.applications.chem.chem_libs.circuit_chem_library.ansatz_puccd import generate_puccd_ex_ops
from tyxonq.libs.hamiltonian_encoding.pauli_io import rdm_mo2ao
from .ucc import UCC
from tyxonq.applications.chem.chem_libs.quantum_chem_library.ci_state_mapping import get_ci_strings
from tyxonq.libs.circuits_library.qubit_state_preparation import get_circuit_givens_swap


class PUCCD(UCC):
    """
    Run paired UCC calculation.
    The interfaces are similar to :class:`UCCSD <tencirchem.UCCSD>`.
    """

    # todo: more documentation here and make the references right.
    # separate docstring examples for a variety of methods, such as energy()
    # also need to add a few comment on make_rdm1/2
    # https://arxiv.org/pdf/2002.00035.pdf
    # https://arxiv.org/pdf/1503.04878.pdf

    def __init__(
        self,
        mol: Union[Mole, RHF],
        init_method: str = "mp2",
        active_space: Tuple[int, int] = None,
        aslst: List[int] = None,
        mo_coeff: np.ndarray = None,
        engine: str = None,
        run_hf: bool = True,
        run_mp2: bool = True,
        run_ccsd: bool = True,
        run_fci: bool = True,
    ):
        r"""
        Initialize the class with molecular input.

        Parameters
        ----------
        mol: Mole or RHF
            The molecule as PySCF ``Mole`` object or the PySCF ``RHF`` object
        init_method: str, optional
            How to determine the initial amplitude guess. Accepts ``"mp2"`` (default), ``"ccsd"``,``"fe"``
            and ``"zeros"``.
        active_space: Tuple[int, int], optional
            Active space approximation. The first integer is the number of electrons and the second integer is
            the number or spatial-orbitals. Defaults to None.
        aslst: List[int], optional
            Pick orbitals for the active space. Defaults to None which means the orbitals are sorted by energy.
            The orbital index is 0-based.

            .. note::
                See `PySCF document <https://pyscf.org/user/mcscf.html#picking-an-active-space>`_
                for choosing the active space orbitals. Here orbital index is 0-based, whereas in PySCF by default it
                is 1-based.

        mo_coeff: np.ndarray, optional
            Molecule coefficients. If provided then RHF is skipped.
            Can be used in combination with the ``init_state`` attribute.
            Defaults to None which means RHF orbitals are used.
        engine: str, optional
            The engine to run the calculation. See :ref:`advanced:Engines` for details.
        run_hf: bool, optional
            Whether run HF for molecule orbitals. Defaults to ``True``.
        run_mp2: bool, optional
            Whether run MP2 for initial guess and energy reference. Defaults to ``True``.
        run_ccsd: bool, optional
            Whether run CCSD for initial guess and energy reference. Defaults to ``True``.
        run_fci: bool, optional
            Whether run FCI  for energy reference. Defaults to ``True``.

        See Also
        --------
        tencirchem.UCCSD
        tencirchem.KUPCCGSD
        tencirchem.UCC
        """
        super().__init__(
            mol,
            init_method,
            active_space,
            aslst,
            mo_coeff,
            mode="hcb",
            engine=engine,
            run_hf=run_hf,
            run_mp2=run_mp2,
            run_ccsd=run_ccsd,
            run_fci=run_fci,
        )
        self.ex_ops, self.param_ids, self.init_guess = self.get_ex_ops(self.t1, self.t2)

    def get_ex1_ops(self, t1: np.ndarray = None):
        """Not applicable. Use :func:`get_ex_ops`."""
        raise NotImplementedError

    def get_ex2_ops(self, t2: np.ndarray = None):
        """Not applicable. Use :func:`get_ex_ops`."""
        raise NotImplementedError

    def get_ex_ops(self, t1: np.ndarray = None, t2: np.ndarray = None) -> Tuple[List[Tuple], List[int], List[float]]:
        """
        Get paired excitation operators for pUCCD ansatz.

        Parameters
        ----------
        t1: np.ndarray, optional
            Not used. Kept for consistency with the parent method.
        t2: np.ndarray, optional
            Not used. Kept for consistency with the parent method.

        Returns
        -------
        ex_op: List[Tuple]
            The excitation operators. Each operator is represented by a tuple of ints.
        param_ids: List[int]
            The mapping from excitations to parameters.
        init_guess: List[float]
            The initial guess for the parameters.

        Examples
        --------
        >>> from tencirchem import PUCCD
        >>> from tencirchem.molecule import h2
        >>> puccd = PUCCD(h2)
        >>> ex_op, param_ids, init_guess = puccd.get_ex_ops()
        >>> ex_op
        [(1, 0)]
        >>> param_ids
        [0]
        >>> init_guess  # doctest:+ELLIPSIS
        [...]
        """
        no, nv = self.no, self.nv
        if t2 is None:
            t2 = np.zeros((no, no, nv, nv))
        else:
            if t2.shape == (2 * no, 2 * no, 2 * nv, 2 * nv):
                # compress to spatial block for init guess
                t2 = t2[0::2, 0::2, 0::2, 0::2]
            else:
                assert t2.shape == (no, no, nv, nv)
        return generate_puccd_ex_ops(no, nv, t2)

    def make_rdm1(self, statevector=None, basis="AO"):
        __doc__ = super().make_rdm1.__doc__

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

    def make_rdm2(self, statevector=None, basis="AO"):
        __doc__ = super().make_rdm2.__doc__

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
        else:
            params = self._check_params_argument(params, strict=False)
            return get_circuit_givens_swap(params, self.n_qubits, self.n_elec, self.init_state)

    @property
    def e_puccd(self):
        """
        Returns pUCCD energy
        """
        return self.energy()
