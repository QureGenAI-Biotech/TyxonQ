


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
from tyxonq.applications.chem.classical_chem_cloud.config import create_classical_client, CloudClassicalConfig
from ..constants import DISCARD_EPS


class UCCSD(UCC):
    """
    Run UCCSD calculation. For a comprehensive tutorial see :doc:`/tutorial_jupyter/ucc_functions`.

    Examples
    --------
    >>> import numpy as np
    >>> from tyxonq.chem import UCCSD
    >>> from tyxonq.chem.molecule import h2
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
        active_orbital_indices: List[int] = None,
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
        *,
        classical_provider: str = "local",
        classical_device: str = "auto",
        # Optional PySCF-style direct molecule construction
        atom: object | None = None,
        basis: str = "sto-3g",
        unit: str = "Angstrom",
        charge: int = 0,
        spin: int = 0,
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
        active_orbital_indices: List[int], optional
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
        tyxonq.chem.KUPCCGSD
        tyxonq.chem.PUCCD
        tyxonq.chem.UCC
        """
        # --- Optional: build Mole from PySCF-style inputs if `atom` provided ---
        if atom is not None:
            try:
                from pyscf import gto as _gto  # type: ignore
                mm = _gto.Mole()
                mm.atom = atom  # accepts string or list spec per PySCF
                mm.unit = str(unit)
                mm.basis = str(basis)
                mm.charge = int(charge)
                mm.spin = int(spin)
                mm.build()
                mol = mm  # override input
            except Exception:
                pass

        # --- RHF setup (with optional cloud offload for HF/integrals) ---
        # Normalize PySCF molecule object
        _py_mol = getattr(mol, "mol", mol)
        # Create a local HF container regardless, to keep compatibility with tests accessing self.hf
        hf = RHF(_py_mol)
        if mo_coeff is not None:
            hf.mo_coeff = np.asarray(mo_coeff)
        hf.chkfile = None
        hf.verbose = 0

        if str(classical_provider) != "local":
            # Offload HF convergence and MO-basis integrals to cloud
            client = create_classical_client(str(classical_provider), str(classical_device), CloudClassicalConfig())
            mdat = {
                "atom": getattr(_py_mol, "atom", None),
                "basis": getattr(_py_mol, "basis", "sto-3g"),
                "charge": int(getattr(_py_mol, "charge", 0)),
                "spin": int(getattr(_py_mol, "spin", 0)),
            }
            task = {
                "method": "hf_integrals",
                "molecule_data": mdat,
                "active_space": active_space,
                "active_orbital_indices": active_orbital_indices,
            }
            _res = client.submit_classical_calculation(task)
            # Load results
            int1e = np.asarray(_res["int1e"])  # type: ignore[index]
            int2e = np.asarray(_res["int2e"])  # type: ignore[index]
            e_core = float(_res["e_core"])  # type: ignore[index]
            try:
                hf.mo_coeff = np.asarray(_res.get("mo_coeff")) if _res.get("mo_coeff") is not None else getattr(hf, "mo_coeff", None)  # type: ignore[attr-defined]
            except Exception:
                pass
            try:
                hf.e_tot = float(_res.get("e_hf", 0.0))  # type: ignore[attr-defined]
            except Exception:
                pass
            # In cloud mode, skip local HF run
            self.hf = hf
        else:
            # Local HF
            if run_hf:
                hf.kernel()
            self.hf = hf
            # MO-basis integrals from local HF
            int1e, int2e, e_core = get_integral_from_hf(hf, active_space=active_space, active_orbital_indices=active_orbital_indices)
        # Active space electron/orbital counts
        if active_space is None:
            n_elec = int(getattr(hf.mol, "nelectron"))
            n_cas = int(getattr(hf.mol, "nao"))
        else:
            n_elec, n_cas = int(active_space[0]), int(active_space[1])
        self.active_space = (n_elec, n_cas)
        self.active_orbital_indices = active_orbital_indices
        self.inactive_occ = 0
        self.inactive_vir = 0
        self.no = n_elec // 2
        self.nv = n_cas - self.no

        # --- Reference energies (HF/FCI) ---
        # e_hf
        try:
            self.e_hf = float(getattr(hf, "e_tot", 0.0))
        except Exception:
            self.e_hf = float("nan")
        # e_fci (optionally cloud)
        if run_fci:
            if str(classical_provider) != "local":
                try:
                    client = create_classical_client(str(classical_provider), str(classical_device), CloudClassicalConfig())
                    mdat = {
                        "atom": getattr(_py_mol, "atom", None),
                        "basis": getattr(_py_mol, "basis", "sto-3g"),
                        "charge": int(getattr(_py_mol, "charge", 0)),
                        "spin": int(getattr(_py_mol, "spin", 0)),
                    }
                    res_fci = client.submit_classical_calculation({
                        "method": "fci",
                        "molecule_data": mdat,
                        "active_space": active_space,
                    })
                    self.e_fci = float(res_fci.get("energy", float("nan")))
                except Exception:
                    self.e_fci = float("nan")
            else:
                try:
                    from pyscf import fci as _fci  # type: ignore
                    self.e_fci = float(_fci.FCI(hf).kernel()[0])
                except Exception:
                    self.e_fci = float("nan")
        else:
            self.e_fci = float("nan")

        # --- Initial amplitudes t1/t2 according to init_method ---
        t1 = np.zeros((self.no, self.nv))
        t2 = np.zeros((self.no, self.no, self.nv, self.nv))
        method = (init_method or "mp2").lower()
        # In cloud HF mode, we avoid local MP2/CCSD amplitude calculations and fall back to zeros
        if str(classical_provider) != "local":
            run_mp2 = False
            run_ccsd = False
            if method in ("mp2", "ccsd"):
                method = "zeros"
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
        # For active_orbital_indices active-space selections, avoid discarding any excitations to match CASCI reference
        if active_orbital_indices is not None:
            self.pick_ex2 = False
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
        # Store integrals for later runtime construction
        self._int1e = np.asarray(int1e)
        self._int2e = np.asarray(int2e)
        # Back-compat attributes used by tests
        self.n_elec = int(n_elec)
        self.civector_size = int(self.n_qubits if hasattr(self, 'n_qubits') else (2 * n_cas))

        # Tighten optimization when using active-space (including active_orbital_indices) to match CASCI reference
        if active_space is not None:
            # More iterations and stricter tolerances to drive deltaE < 1e-5
            self.scipy_minimize_options = {"ftol": 1e-12, "gtol": 1e-8, "maxiter": 600}

    # ---- Compatibility: expose FermionOperator Hamiltonian (electronic part, without e_core) ----
    @property
    def h_fermion_op(self):
        """Return FermionOperator for total Hamiltonian (electronic + e_core).

        Older tests expect using this with a mapping (e.g., parity) to produce
        a qubit Hamiltonian including the constant energy shift.
        """
        from openfermion import FermionOperator as _FOP  # lazy import
        hop = get_hop_from_integral(self._int1e, self._int2e)
        try:
            core = float(getattr(self, "e_core", 0.0))
        except Exception:
            core = 0.0
        if abs(core) > 0:
            hop += _FOP((), core)
        return hop

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
        >>> from tyxonq.chem import UCCSD
        >>> from tyxonq.chem.molecule import h2
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

    # ---- Convenience builders from integrals ----
    @classmethod
    def from_integral(
        cls,
        int1e: np.ndarray,
        int2e: np.ndarray,
        n_elec: Union[int, Tuple[int, int]],
        e_core: float | None = None,
        ovlp: np.ndarray | None = None,
        *,
        mode: str = "fermion",
        runtime: str = "device",
        pick_ex2: bool = False,
        sort_ex2: bool = False,
        epsilon: float = DISCARD_EPS,
        numeric_engine: str | None = None,
    ) -> "UCCSD":
        # Derive CAS sizes
        n_cas = int(len(int1e))
        if isinstance(n_elec, int):
            assert n_elec % 2 == 0
            n_elec_s = (n_elec // 2, n_elec // 2)
        else:
            n_elec_s = (int(n_elec[0]), int(n_elec[1]))
        na, nb = int(n_elec_s[0]), int(n_elec_s[1])
        no = na
        nv = n_cas - no

        # Build qubit Hamiltonian
        from openfermion.transforms import jordan_wigner as _jw
        from tyxonq.libs.hamiltonian_encoding.pauli_io import reverse_qop_idx as _rev
        from tyxonq.applications.chem.chem_libs.hamiltonians_chem_library.hamiltonian_builders import get_hop_from_integral as _hop

        fop = _hop(int1e, int2e)
        n_qubits = 2 * n_cas
        hq = _rev(_jw(fop), n_qubits)

        # Create bare instance bypassing __init__ and initialize UCC base
        inst = cls.__new__(cls)
        UCC.__init__(
            inst,
            n_qubits=n_qubits,
            n_elec_s=(na, nb),
            h_qubit_op=hq,
            runtime=("numeric" if numeric_engine is not None else runtime),
            mode=mode,
            ex_ops=None,
            param_ids=None,
            init_state=None,
            decompose_multicontrol=False,
            trotter=False,
        )
        # Record energies and preferences
        inst.e_core = float(e_core) if e_core is not None else 0.0
        inst.numeric_engine = numeric_engine
        # cache QubitOperator for runtime shots==0 fast path
        inst._qop_cached = hq
        inst._int1e = np.asarray(int1e)
        inst._int2e = np.asarray(int2e)
        inst.n_elec = int(na + nb)
        inst.civector_size = int(inst.n_qubits)

        # Generate UCCSD excitations and init guess (T1/T2 zeros if no CC info)
        inst.t2_discard_eps = epsilon
        inst.pick_ex2 = bool(pick_ex2)
        inst.sort_ex2 = bool(sort_ex2)
        t1 = np.zeros((no, nv))
        t2 = np.zeros((no, no, nv, nv))
        ex1_ops, ex1_param_ids, ex1_init_guess = generate_uccsd_ex1_ops(no, nv, t1, mode=mode)
        ex2_ops, ex2_param_ids, ex2_init_guess = generate_uccsd_ex2_ops(no, nv, t2, mode=mode)
        ex2_ops, ex2_param_ids, ex2_init_guess = inst.pick_and_sort(ex2_ops, ex2_param_ids, ex2_init_guess, inst.pick_ex2, inst.sort_ex2)
        ex_ops = ex1_ops + ex2_ops
        param_ids = ex1_param_ids + [i + max(ex1_param_ids) + 1 for i in ex2_param_ids]
        init_guess = ex1_init_guess + ex2_init_guess
        # Normalize param ids to contiguous range and align init guess
        unique_ids = np.unique(param_ids)
        id_map = {int(old): idx for idx, old in enumerate(unique_ids)}
        param_ids = [id_map[int(i)] for i in param_ids]
        init_vec = np.array(init_guess)[unique_ids]
        inst.ex_ops = ex_ops
        inst.param_ids = param_ids
        inst.init_guess = np.asarray(init_vec, dtype=np.float64)
        # Reference FCI energy for assertions
        try:
            from pyscf.fci import direct_spin1 as _fci_ds1  # type: ignore
            inst.e_fci = float(_fci_ds1.FCI().kernel(int1e, int2e, n_cas, (na, nb))[0] + (float(e_core) if e_core is not None else 0.0))
        except Exception:
            inst.e_fci = float('nan')
        return inst

    # Use base class numeric path; runtime construction now injects CI Hamiltonian centrally


class ROUCCSD(UCC):
    def __init__(
        self,
        mol: Union[Mole, ROHF],
        active_space: Tuple[int, int] = None,
        active_orbital_indices: List[int] = None,
        mo_coeff: np.ndarray = None,
        numeric_engine: str | None = None,
        run_hf: bool = True,
        # for API consistency with UCC
        run_mp2: bool = False,
        run_ccsd: bool = False,
        run_fci: bool = True,
        *,
        classical_provider: str = "local",
        classical_device: str = "auto",
    ):
        # --- ROHF setup (open-shell) ---
        _py_mol = getattr(mol, "mol", mol)
        hf = ROHF(_py_mol)
        if mo_coeff is not None:
            hf.mo_coeff = np.asarray(mo_coeff)
        hf.chkfile = None
        hf.verbose = 0

        if str(classical_provider) != "local":
            client = create_classical_client(str(classical_provider), str(classical_device), CloudClassicalConfig())
            mdat = {
                "atom": getattr(_py_mol, "atom", None),
                "basis": getattr(_py_mol, "basis", "sto-3g"),
                "charge": int(getattr(_py_mol, "charge", 0)),
                "spin": int(getattr(_py_mol, "spin", 0)),
            }
            _res = client.submit_classical_calculation({
                "method": "hf_integrals",
                "molecule_data": mdat,
                "active_space": active_space,
                "active_orbital_indices": active_orbital_indices,
            })
            int1e = np.asarray(_res["int1e"])  # type: ignore[index]
            int2e = np.asarray(_res["int2e"])  # type: ignore[index]
            e_core = float(_res["e_core"])  # type: ignore[index]
            try:
                hf.mo_coeff = np.asarray(_res.get("mo_coeff")) if _res.get("mo_coeff") is not None else getattr(hf, "mo_coeff", None)  # type: ignore[attr-defined]
            except Exception:
                pass
            try:
                hf.e_tot = float(_res.get("e_hf", 0.0))  # type: ignore[attr-defined]
            except Exception:
                pass
        else:
            if run_hf:
                hf.kernel()

        # --- Integrals and core energy ---
        if str(classical_provider) == "local":
            int1e, int2e, e_core = get_integral_from_hf(hf, active_space=active_space, active_orbital_indices=active_orbital_indices)

        # Active space: electrons and spatial orbitals
        if active_space is None:
            n_elec = int(getattr(hf.mol, "nelectron"))
            n_cas = int(getattr(hf.mol, "nao"))
        else:
            n_elec, n_cas = int(active_space[0]), int(active_space[1])
        self.active_space = (n_elec, n_cas)

        # Derive CAS occupations from (n_elec, spin) to avoid dependence on MO ordering
        # spin = N_alpha - N_beta in PySCF Mole
        spin = int(getattr(getattr(hf, "mol", None), "spin", 0))
        n_alpha = (int(n_elec) + spin) // 2
        n_beta = (int(n_elec) - spin) // 2
        # Doubly occupied spatial orbitals in CAS equals n_beta; singly occupied equals spin
        no = int(n_beta)
        ns = int(spin)
        nv = int(n_cas) - (no + ns)
        if nv < 0:
            # Fallback: clamp to zero and adjust no to keep counts valid in small CAS
            nv = 0
            no = max(0, int(n_cas) - ns)
        assert no >= 0 and ns >= 0 and nv >= 0 and (no + ns + nv) == int(n_cas)

        # alpha/beta occupied and virtual counts in CAS
        noa = no + ns  # alpha occupied count (doubles + singles)
        nob = no       # beta occupied count (doubles only)
        nva = nv       # alpha virtual count
        nvb = ns + nv  # beta virtual count (beta has fewer occupied)

        # --- Reference energies (optional FCI) ---
        if run_fci:
            if str(classical_provider) != "local":
                try:
                    client = create_classical_client(str(classical_provider), str(classical_device), CloudClassicalConfig())
                    mdat = {
                        "atom": getattr(_py_mol, "atom", None),
                        "basis": getattr(_py_mol, "basis", "sto-3g"),
                        "charge": int(getattr(_py_mol, "charge", 0)),
                        "spin": int(getattr(_py_mol, "spin", 0)),
                    }
                    res_fci = client.submit_classical_calculation({
                        "method": "fci",
                        "molecule_data": mdat,
                        "active_space": active_space,
                    })
                    self.e_fci = float(res_fci.get("energy", float("nan")))
                except Exception:
                    self.e_fci = float("nan")
            else:
                try:
                    from pyscf.fci import direct_spin1 as _fci_ds1  # type: ignore
                    # CAS FCI on (int1e, int2e) with (na, nb) in CAS, then add core energy
                    self.e_fci = float(_fci_ds1.FCI().kernel(int1e, int2e, n_cas, (noa, nob))[0] + e_core)
                except Exception:
                    self.e_fci = float("nan")
        else:
            self.e_fci = float("nan")

        # --- Ex-ops (open-shell mapping) ---
        def alpha_o(_i):
            return self.active_space[1] + _i

        def alpha_v(_i):
            return self.active_space[1] + noa + _i

        def beta_o(_i):
            return _i

        def beta_v(_i):
            return nob + _i

        ex_ops: list[tuple] = []
        # single excitations
        for i in range(noa):
            for a in range(nva):
                ex_ops.append((alpha_v(a), alpha_o(i)))  # alpha→alpha
        for i in range(nob):
            for a in range(nvb):
                ex_ops.append((beta_v(a), beta_o(i)))    # beta→beta

        # double excitations
        # 2 alphas
        for i in range(noa):
            for j in range(i):
                for a in range(nva):
                    for b in range(a):
                        ex_ops.append((alpha_v(b), alpha_v(a), alpha_o(i), alpha_o(j)))
        # 2 betas
        for i in range(nob):
            for j in range(i):
                for a in range(nvb):
                    for b in range(a):
                        ex_ops.append((beta_v(b), beta_v(a), beta_o(i), beta_o(j)))
        # 1 alpha + 1 beta
        for i in range(noa):
            for j in range(nob):
                for a in range(nva):
                    for b in range(nvb):
                        ex_ops.append((beta_v(b), alpha_v(a), alpha_o(i), beta_o(j)))

        param_ids = list(range(len(ex_ops)))
        init_guess = np.zeros_like(param_ids)

        # --- Build qubit Hamiltonian ---
        n_qubits = 2 * n_cas
        fop = get_hop_from_integral(int1e, int2e)
        hq = reverse_qop_idx(jordan_wigner(fop), n_qubits)

        # --- Initialize base UCC with open-shell (na, nb) ---
        super().__init__(
            n_qubits=n_qubits,
            n_elec_s=(noa, nob),
            h_qubit_op=hq,
            runtime=("numeric" if numeric_engine is not None else "device"),
            mode="fermion",
            ex_ops=ex_ops,
            param_ids=param_ids,
            init_state=None,
            decompose_multicontrol=False,
            trotter=False,
        )

        # record energies and preferences
        self.e_core = float(e_core)
        self.init_guess = np.asarray(init_guess, dtype=np.float64)
        self.numeric_engine = numeric_engine
        self._int1e = np.asarray(int1e)
        self._int2e = np.asarray(int2e)
        # For CI engines, provide ci_hamiltonian via algorithms.UCC.energy/energy_and_grad
        self.n_elec = int(n_elec)

    # ---- Convenience: build ROUCCSD directly from integrals (open-shell CAS) ----
    @classmethod
    def from_integral(
        cls,
        int1e: np.ndarray,
        int2e: np.ndarray,
        n_elec: Union[int, Tuple[int, int]],
        e_core: float | None = None,
        *,
        mode: str = "fermion",
        runtime: str = "device",
        numeric_engine: str | None = None,
    ) -> "ROUCCSD":
        import numpy as _np
        from openfermion.transforms import jordan_wigner as _jw
        from tyxonq.libs.hamiltonian_encoding.pauli_io import reverse_qop_idx as _rev
        from tyxonq.applications.chem.chem_libs.hamiltonians_chem_library.hamiltonian_builders import get_hop_from_integral as _hop

        n_cas = int(len(int1e))
        if isinstance(n_elec, int):
            # Derive (na, nb) for open-shell if odd, closed-shell if even
            if n_elec % 2 == 0:
                n_elec_s = (n_elec // 2, n_elec // 2)
            else:
                n_elec_s = ((n_elec + 1) // 2, (n_elec - 1) // 2)
        else:
            n_elec_s = (int(n_elec[0]), int(n_elec[1]))

        na, nb = int(n_elec_s[0]), int(n_elec_s[1])

        # Occupation counts in CAS
        # Doubly occupied spatial orbitals equals min(na, nb); singles equals |na-nb|
        spin = abs(na - nb)
        no = min(na, nb)
        ns = spin
        nv = n_cas - (no + ns)
        if nv < 0:
            nv = 0
            no = max(0, n_cas - ns)
        noa = no + ns
        nob = no
        nva = nv
        nvb = ns + nv

        # Build open-shell excitation list
        ex_ops: list[tuple] = []
        # single excitations
        def alpha_o(_i: int) -> int: return n_cas + _i
        def alpha_v(_i: int) -> int: return n_cas + noa + _i
        def beta_o(_i: int) -> int: return _i
        def beta_v(_i: int) -> int: return nob + _i

        for i in range(noa):
            for a in range(nva):
                ex_ops.append((alpha_v(a), alpha_o(i)))
        for i in range(nob):
            for a in range(nvb):
                ex_ops.append((beta_v(a), beta_o(i)))
        # double excitations
        for i in range(noa):
            for j in range(i):
                for a in range(nva):
                    for b in range(a):
                        ex_ops.append((alpha_v(b), alpha_v(a), alpha_o(i), alpha_o(j)))
        for i in range(nob):
            for j in range(i):
                for a in range(nvb):
                    for b in range(a):
                        ex_ops.append((beta_v(b), beta_v(a), beta_o(i), beta_o(j)))
        for i in range(noa):
            for j in range(nob):
                for a in range(nva):
                    for b in range(nvb):
                        ex_ops.append((beta_v(b), alpha_v(a), alpha_o(i), beta_o(j)))

        param_ids = list(range(len(ex_ops)))
        init_guess = _np.zeros_like(param_ids)

        # Hamiltonian (qubit)
        n_qubits = 2 * n_cas
        fop = _hop(int1e, int2e)
        hq = _rev(_jw(fop), n_qubits)

        # Initialize base UCC fields on a new instance of cls
        inst = cls.__new__(cls)
        UCC.__init__(
            inst,
            n_qubits=n_qubits,
            n_elec_s=(na, nb),
            h_qubit_op=hq,
            runtime=("device" if numeric_engine is not None else runtime),
            mode=str(mode),
            ex_ops=ex_ops,
            param_ids=param_ids,
            init_state=None,
            decompose_multicontrol=False,
            trotter=False,
        )
        inst.e_core = float(e_core) if e_core is not None else 0.0
        inst.init_guess = _np.asarray(init_guess, dtype=_np.float64)
        inst.numeric_engine = numeric_engine
        inst._int1e = _np.asarray(int1e)
        inst._int2e = _np.asarray(int2e)
        inst.n_elec = int(na + nb)
        return inst
