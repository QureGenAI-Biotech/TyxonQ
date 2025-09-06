


from typing import Tuple, List, Union

import numpy as np
from pyscf.gto.mole import Mole
from pyscf.scf import RHF
from pyscf.scf import ROHF

import warnings as _warnings
from .ucc import UCC
from openfermion.transforms import jordan_wigner
from tyxonq.libs.hamiltonian_encoding.pauli_io import reverse_qop_idx
from tyxonq.applications.chem.chem_libs.hamiltonians_chem_library.hamiltonian_builders import (
    get_integral_from_hf,
    get_hop_from_integral,
)
from tyxonq.applications.chem.chem_libs.circuit_chem_library.ansatz_uccsd import (
    generate_uccsd_ex1_ops,
    generate_uccsd_ex2_ops,
)
from ..constants import DISCARD_EPS


class UCCSD(UCC):
    """
    Run UCCSD calculation. For a comprehensive tutorial see :doc:`/tutorial_jupyter/ucc_functions`.

    Examples
    --------
    >>> import numpy as np
    >>> from tencirchem import UCCSD
    >>> from tencirchem.molecule import h2
    >>> uccsd = UCCSD(h2)
    >>> e_ucc = uccsd.kernel()
    >>> np.testing.assert_allclose(e_ucc, uccsd.e_fci, atol=1e-10)
    >>> e_hf = uccsd.energy(np.zeros(uccsd.n_params))
    >>> np.testing.assert_allclose(e_hf, uccsd.e_hf, atol=1e-10)
    """

    def __init__(
        self,
        mol: Union[Mole, RHF],
        init_method: str = "mp2",
        active_space: Tuple[int, int] = None,
        aslst: List[int] = None,
        mo_coeff: np.ndarray = None,
        pick_ex2: bool = True,
        epsilon: float = DISCARD_EPS,
        sort_ex2: bool = True,
        mode: str = "fermion",
        runtime: str = None,
        numeric_engine: str | None = None,
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
        pick_ex2: bool, optional
            Whether screen out two body excitations based on the inital guess amplitude.
            Defaults to True, which means excitations with amplitude less than ``epsilon`` (see below) are discarded.
            The argument will be set to ``False`` if initial guesses are set to zero.
        mode: str, optional
            How to deal with particle symmetry. Possible values are ``"fermion"``, ``"qubit"``.
            Default to ``"fermion"``.
        epsilon: float, optional
            The threshold to discard two body excitations. Defaults to 1e-12.
        sort_ex2: bool, optional
            Whether sort two-body excitations in the ansatz based on the initial guess amplitude.
            Large excitations come first. Defaults to True.
            Note this could lead to different ansatz for the same molecule at different geometry.
            The argument will be set to ``False`` if initial guesses are set to zero.
        runtime: str, optional
            The runtime to run the calculation (e.g., 'device').
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
        tencirchem.KUPCCGSD
        tencirchem.PUCCD
        tencirchem.UCC
        """
        # --- RHF setup ---
        # Avoid fragile isinstance on PySCF factories; detect by attributes
        if hasattr(mol, "mol") and hasattr(mol, "kernel"):
            hf = mol
        else:
            hf = RHF(mol)
        if mo_coeff is not None:
            hf.mo_coeff = np.asarray(mo_coeff)
        hf.chkfile = None
        hf.verbose = 0
        if run_hf:
            hf.kernel()
        self.hf = hf

        # --- Integrals and core energy ---
        int1e, int2e, e_core = get_integral_from_hf(hf, active_space=active_space, aslst=aslst)
        # Active space electron/orbital counts
        if active_space is None:
            n_elec = int(getattr(hf.mol, "nelectron"))
            n_cas = int(getattr(hf.mol, "nao"))
        else:
            n_elec, n_cas = int(active_space[0]), int(active_space[1])
        self.active_space = (n_elec, n_cas)
        self.inactive_occ = 0
        self.inactive_vir = 0
        self.no = n_elec // 2
        self.nv = n_cas - self.no

        # --- Reference energies ---
        try:
            self.e_hf = float(getattr(hf, "e_tot", 0.0))
        except Exception:
            self.e_hf = float("nan")
        try:
            if run_fci:
                from pyscf import fci as _fci  # type: ignore

                self.e_fci = float(_fci.FCI(hf).kernel()[0])
            else:
                self.e_fci = float("nan")
        except Exception:
            self.e_fci = float("nan")

        # --- Initial amplitudes t1/t2 according to init_method ---
        t1 = np.zeros((self.no, self.nv))
        t2 = np.zeros((self.no, self.no, self.nv, self.nv))
        method = (init_method or "mp2").lower()
        mp2_amp = None
        if method in ("mp2", "ccsd", "fe") and (run_mp2 or method == "mp2"):
            try:
                from pyscf.mp import MP2  # type: ignore

                _mp = MP2(hf)
                _mp.kernel()
                mp2_full = np.asarray(getattr(_mp, "t2", None))
                if mp2_full is not None and mp2_full.ndim >= 4:
                    mp2_amp = np.abs(mp2_full[: self.no, : self.no, : self.nv, : self.nv])
            except Exception:
                mp2_amp = None
        if method in ("ccsd", "fe") and run_ccsd:
            try:
                from pyscf.cc import ccsd as _cc  # type: ignore

                cc = _cc.CCSD(hf)
                cc.kernel()
                cc_t1 = np.asarray(getattr(cc, "t1", None))
                if cc_t1 is not None and cc_t1.shape[0] >= self.no and cc_t1.shape[1] >= self.nv:
                    t1 = np.asarray(cc_t1[: self.no, : self.nv], dtype=float)
                cc_t2 = np.asarray(getattr(cc, "t2", None))
                if cc_t2 is not None and cc_t2.ndim >= 4:
                    t2 = np.abs(cc_t2[: self.no, : self.no, : self.nv, : self.nv])
                elif mp2_amp is not None:
                    t2 = np.asarray(mp2_amp, dtype=float)
            except Exception:
                if mp2_amp is not None:
                    t2 = np.asarray(mp2_amp, dtype=float)
        elif method == "mp2" and mp2_amp is not None:
            t2 = np.asarray(mp2_amp, dtype=float)
        # zeros: keep t1/t2 as zeros

        # --- Ex-ops & init guesses ---
        self.t2_discard_eps = epsilon
        if method == "zeros":
            self.pick_ex2 = self.sort_ex2 = False
        else:
            self.pick_ex2 = bool(pick_ex2)
            self.sort_ex2 = bool(sort_ex2)
        ex1_ops, ex1_param_ids, ex1_init_guess = generate_uccsd_ex1_ops(self.no, self.nv, t1, mode=mode)
        ex2_ops, ex2_param_ids, ex2_init_guess = generate_uccsd_ex2_ops(self.no, self.nv, t2, mode=mode)
        ex2_ops, ex2_param_ids, ex2_init_guess = self.pick_and_sort(ex2_ops, ex2_param_ids, ex2_init_guess, self.pick_ex2, self.sort_ex2)
        ex_ops = ex1_ops + ex2_ops
        param_ids = ex1_param_ids + [i + max(ex1_param_ids) + 1 for i in ex2_param_ids]
        init_guess = ex1_init_guess + ex2_init_guess

        # --- Map to QubitOperator ---
        n_qubits = 2 * n_cas
        fop = get_hop_from_integral(int1e, int2e)
        hq = reverse_qop_idx(jordan_wigner(fop), n_qubits)
        na = self.no
        nb = n_elec - na

        # --- Initialize internal UCC (new signature) ---
        # If numeric_engine is specified and runtime not provided, default to numeric path
        _runtime = str(runtime or ("numeric" if numeric_engine is not None else "device"))

        super().__init__(
            n_qubits=n_qubits,
            n_elec_s=(na, nb),
            h_qubit_op=hq,
            runtime=_runtime,
            mode=str(mode),
            ex_ops=ex_ops,
            param_ids=param_ids,
            init_state=None,
            decompose_multicontrol=False,
            trotter=False,
        )
        self.e_core = float(e_core)
        # adopt generated init guesses
        self.init_guess = np.asarray(init_guess, dtype=np.float64) if len(init_guess) > 0 else np.zeros(0, dtype=np.float64)
        # remember preferred numeric engine if provided
        self.numeric_engine = numeric_engine
        # Back-compat attributes used by tests
        self.n_elec = int(n_elec)
        self.civector_size = int(self.n_qubits if hasattr(self, 'n_qubits') else (2 * n_cas))

    def get_ex_ops(self, t1: np.ndarray = None, t2: np.ndarray = None) -> Tuple[List[Tuple], List[int], List[float]]:
        """
        Get one-body and two-body excitation operators for UCCSD ansatz.
        Pick and sort two-body operators if ``self.pick_ex2`` and ``self.sort_ex2`` are set to ``True``.

        Parameters
        ----------
        t1: np.ndarray, optional
            Initial one-body amplitudes based on e.g. CCSD
        t2: np.ndarray, optional
            Initial two-body amplitudes based on e.g. MP2

        Returns
        -------
        ex_op: List[Tuple]
            The excitation operators. Each operator is represented by a tuple of ints.
        param_ids: List[int]
            The mapping from excitations to parameters.
        init_guess: List[float]
            The initial guess for the parameters.

        See Also
        --------
        get_ex1_ops: Get one-body excitation operators.
        get_ex2_ops: Get two-body excitation operators.

        Examples
        --------
        >>> from tencirchem import UCCSD
        >>> from tencirchem.molecule import h2
        >>> uccsd = UCCSD(h2)
        >>> ex_op, param_ids, init_guess = uccsd.get_ex_ops()
        >>> ex_op
        [(3, 2), (1, 0), (1, 3, 2, 0)]
        >>> param_ids
        [0, 0, 1]
        >>> init_guess  # doctest:+ELLIPSIS
        [0.0, ...]
        """
        # Delegate ex-op generation to libs to keep one source of truth
        ex1_ops, ex1_param_ids, ex1_init_guess = generate_uccsd_ex1_ops(self.no, self.nv, t1, mode=self.mode)
        ex2_ops, ex2_param_ids, ex2_init_guess = generate_uccsd_ex2_ops(self.no, self.nv, t2, mode=self.mode)

        # screen out symmetrically not allowed excitation
        ex2_ops, ex2_param_ids, ex2_init_guess = self.pick_and_sort(
            ex2_ops, ex2_param_ids, ex2_init_guess, self.pick_ex2, self.sort_ex2
        )

        ex_op = ex1_ops + ex2_ops
        param_ids = ex1_param_ids + [i + max(ex1_param_ids) + 1 for i in ex2_param_ids]
        init_guess = ex1_init_guess + ex2_init_guess
        return ex_op, param_ids, init_guess

    def pick_and_sort(self, ex_ops, param_ids, init_guess, do_pick=True, do_sort=True):
        # sort operators according to amplitude
        if do_sort:
            sorted_ex_ops = sorted(zip(ex_ops, param_ids), key=lambda x: -np.abs(init_guess[x[1]]))
        else:
            sorted_ex_ops = list(zip(ex_ops, param_ids))
        ret_ex_ops = []
        ret_param_ids = []
        for ex_op, param_id in sorted_ex_ops:
            # discard operators with tiny amplitude.
            # The default eps is so small that the screened out excitations are probably not allowed
            if do_pick and np.abs(init_guess[param_id]) < self.t2_discard_eps:
                continue
            ret_ex_ops.append(ex_op)
            ret_param_ids.append(param_id)
        assert len(ret_ex_ops) != 0
        unique_ids = np.unique(ret_param_ids)
        ret_init_guess = np.array(init_guess)[unique_ids]
        id_mapping = {old: new for new, old in enumerate(unique_ids)}
        ret_param_ids = [id_mapping[i] for i in ret_param_ids]
        return ret_ex_ops, ret_param_ids, list(ret_init_guess)

    @property
    def e_uccsd(self) -> float:
        """
        Returns UCCSD energy
        """
        return self.energy()


class ROUCCSD(UCC):
    def __init__(
        self,
        mol: Union[Mole, ROHF],
        active_space: Tuple[int, int] = None,
        aslst: List[int] = None,
        mo_coeff: np.ndarray = None,
        engine: str = "civector",
        run_hf: bool = True,
        # for API consistency with UCC
        run_mp2: bool = False,
        run_ccsd: bool = False,
        run_fci: bool = True,
    ):
        init_method: str = "zeros"
        # ROHF does not support mp2 and ccsd
        run_mp2: bool = False
        run_ccsd: bool = False

        super().__init__(
            mol,
            init_method,
            active_space,
            aslst,
            mo_coeff,
            engine=engine,
            run_hf=run_hf,
            run_mp2=run_mp2,
            run_ccsd=run_ccsd,
            run_fci=run_fci,
        )
        no = int(np.sum(self.hf.mo_occ == 2)) - self.inactive_occ
        ns = int(np.sum(self.hf.mo_occ == 1))
        nv = int(np.sum(self.hf.mo_occ == 0)) - self.inactive_vir
        assert no + ns + nv == self.active_space[1]
        # assuming single electrons in alpha
        noa = no + ns
        nva = nv
        nob = no
        nvb = ns + nv

        def alpha_o(_i):
            return self.active_space[1] + _i

        def alpha_v(_i):
            return self.active_space[1] + noa + _i

        def beta_o(_i):
            return _i

        def beta_v(_i):
            return nob + _i

        # single excitations
        self.ex_ops = []
        for i in range(noa):
            for a in range(nva):
                # alpha to alpha
                ex_op_a = (alpha_v(a), alpha_o(i))
                self.ex_ops.append(ex_op_a)
        for i in range(nob):
            for a in range(nvb):
                # beta to beta
                ex_op_b = (beta_v(a), beta_o(i))
                self.ex_ops.append(ex_op_b)

        # double excitations
        # 2 alphas
        for i in range(noa):
            for j in range(i):
                for a in range(nva):
                    for b in range(a):
                        ex_op_aa = (alpha_v(b), alpha_v(a), alpha_o(i), alpha_o(j))
                        self.ex_ops.append(ex_op_aa)
        # 2 betas
        for i in range(nob):
            for j in range(i):
                for a in range(nvb):
                    for b in range(a):
                        ex_op_bb = (beta_v(b), beta_v(a), beta_o(i), beta_o(j))
                        self.ex_ops.append(ex_op_bb)

        # 1 alpha + 1 beta
        for i in range(noa):
            for j in range(nob):
                for a in range(nva):
                    for b in range(nvb):
                        ex_op_ab = (beta_v(b), alpha_v(a), alpha_o(i), beta_o(j))
                        self.ex_ops.append(ex_op_ab)

        self.param_ids = list(range(len(self.ex_ops)))
        self.init_guess = np.zeros_like(self.param_ids)
