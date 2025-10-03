from time import time
import logging
from typing import Tuple, List, Union

import numpy as np
from pyscf.gto.mole import Mole
from pyscf.scf import RHF

from .ucc import UCC as _UCCBase
from openfermion.transforms import jordan_wigner
from tyxonq.libs.hamiltonian_encoding.pauli_io import reverse_qop_idx
from tyxonq.applications.chem.chem_libs.hamiltonians_chem_library.hamiltonian_builders import (
    get_hop_from_integral,
)
from tyxonq.applications.chem.chem_libs.circuit_chem_library.ansatz_kupccgsd import (
    generate_kupccgsd_ex1_ops,
    generate_kupccgsd_ex2_ops,
    generate_kupccgsd_ex_ops,
)

from pyscf.fci import direct_spin1

logger = logging.getLogger(__name__)


class KUPCCGSD(_UCCBase):
    """
    Run :math:`k`-UpCCGSD calculation.
    The interfaces are similar to :class:`UCCSD <tencirchem.UCCSD>`.
    """

    def __init__(
        self,
        mol: Union[Mole, RHF],
        active_space: Tuple[int, int] = None,
        active_orbital_indices: List[int] = None,
        mo_coeff: np.ndarray = None,
        k: int = 3,
        n_tries: int = 1,
        runtime: str = 'device',
        run_fci: bool = False,
        numeric_engine: str | None = None,
        classical_provider: str = 'local',
        classical_device: str = 'auto',
        # Optional PySCF-style direct molecule construction
        atom: object | None = None,
        basis: str = "sto-3g",
        unit: str = "Angstrom",
        charge: int = 0,
        spin: int = 0,
        **kwargs
    ):
        r"""
        Initialize the class with molecular input.

        Parameters
        ----------
        mol: Mole or RHF
            The molecule as PySCF ``Mole`` object or the PySCF ``RHF`` object
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
        k: int, optional
            The number of layers in the ansatz. Defaults to 3
        n_tries: int, optional
            The number of different initial points used for VQE calculation.
            For large circuits usually a lot of runs are required for good accuracy.
            Defaults to 1.
        runtime: str, optional
            The runtime to run the calculation (e.g., 'device').
        run_hf: bool, optional
            Whether run HF for molecule orbitals. Defaults to ``True``.
        run_mp2: bool, optional
            Whether run MP2 for energy reference. Defaults to ``True``.
        run_ccsd: bool, optional
            Whether run CCSD for energy reference. Defaults to ``True``.
        run_fci: bool, optional
            Whether run FCI for energy reference. Defaults to ``True``.

        See Also
        --------
        tyxonq.UCCSD
        tyxonq.PUCCD
        tyxonq.UCC
        """



        super().__init__(
            mol=mol,
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
                # the number of layers
        self.k = k
        # the number of different initialization
        self.n_tries = n_tries
        self.ex_ops, self.param_ids, self.init_guess = self.get_ex_ops(self.t1, self.t2)
        self.init_guess_list = [self.init_guess]
        for _ in range(self.n_tries - 1):
            self.init_guess_list.append(np.random.rand(self.n_params) - 0.5)
        self.e_tries_list = []
        self.opt_res_list = []
        self.staging_time = self.opt_time = None

    def kernel(self, **opts):
        _, stating_time = self.get_opt_function(with_time=True)

        time1 = time()
        for i in range(self.n_tries):
            logger.info(f"k-UpCCGSD try {i}")
            if self.n_tries == 1:
                if not np.allclose(self.init_guess, self.init_guess_list[0]):
                    logger.info("Inconsistent `self.init_guess` and `self.init_guess_list`.  Use `self.init_guess`.")
            else:
                self.init_guess = self.init_guess_list[i]
            # Forward runtime options; rely on UCCSD.kernel implementation
            e_try = super().kernel(**opts)
            # Prefer optimizer result stored by base class; fallback to simple namespace
            r = self.opt_res
            self.opt_res_list.append(r)
            logger.info(f"k-UpCCGSD try {i} energy {float(r.fun)}")
        self.opt_res_list.sort(key=lambda x: x.fun)
        self.e_tries_list = [float(res.fun) for res in self.opt_res_list]
        time2 = time()

        self.staging_time = stating_time
        self.opt_time = time2 - time1
        self.opt_res = self.opt_res_list[0]
        self.opt_res.e_tries = self.e_tries_list

        if not self.opt_res.success:
            logger.warning("Optimization failed. See `.opt_res` for details.")

        self.init_guess = self.opt_res.init_guess
        return float(self.opt_res.fun)

    def get_ex_ops(self, t1: np.ndarray = None, t2: np.ndarray = None) -> Tuple[List[Tuple], List[int], np.ndarray]:
        """
        Get one-body and two-body excitation operators for :math:`k`-UpCCGSD ansatz.
        The excitations are generalized and two-body excitations are restricted to paired ones.
        Initial guesses are generated randomly.

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
        init_guess: np.ndarray
            The initial guess for the parameters.

        See Also
        --------
        get_ex1_ops: Get generalized one-body excitation operators.
        get_ex2_ops: Get generalized paired two-body excitation operators.

        Examples
        --------
        >>> from tyxonq.chem import KUPCCGSD
        >>> from tyxonq.chem.molecule import h2
        >>> kupccgsd = KUPCCGSD(h2)
        >>> ex_op, param_ids, init_guess = kupccgsd.get_ex_ops()
        >>> ex_op
        [(1, 3, 2, 0), (3, 2), (1, 0), (1, 3, 2, 0), (3, 2), (1, 0), (1, 3, 2, 0), (3, 2), (1, 0)]
        >>> param_ids
        [0, 1, 1, 2, 3, 3, 4, 5, 5]
        >>> init_guess  # doctest:+ELLIPSIS
        array([...])
        """
        ex_op, param_ids, init_guess = generate_kupccgsd_ex_ops(self.no, self.nv, self.k)
        return ex_op, param_ids, init_guess

    def get_ex1_ops(self, t1: np.ndarray = None) -> Tuple[List[Tuple], List[int], np.ndarray]:
        """
        Get generalized one-body excitation operators.

        Parameters
        ----------
        t1: np.ndarray, optional
            Not used. Kept for consistency with the parent method.

        Returns
        -------
        ex_op: List[Tuple]
            The excitation operators. Each operator is represented by a tuple of ints.
        param_ids: List[int]
            The mapping from excitations to parameters.
        init_guess: np.ndarray
            The initial guess for the parameters.

        See Also
        --------
        get_ex2_ops: Get generalized paired two-body excitation operators.
        get_ex_ops: Get one-body and two-body excitation operators for :math:`k`-UpCCGSD ansatz.
        """
        assert t1 is None
        return generate_kupccgsd_ex1_ops(self.no, self.nv)

    def get_ex2_ops(self, t2: np.ndarray = None) -> Tuple[List[Tuple], List[int], np.ndarray]:
        """
        Get generalized paired two-body excitation operators.

        Parameters
        ----------
        t2: np.ndarray, optional
            Not used. Kept for consistency with the parent method.

        Returns
        -------
        ex_op: List[Tuple]
            The excitation operators. Each operator is represented by a tuple of ints.
        param_ids: List[int]
            The mapping from excitations to parameters.
        init_guess: np.ndarray
            The initial guess for the parameters.

        See Also
        --------
        get_ex1_ops: Get one-body excitation operators.
        get_ex_ops: Get one-body and two-body excitation operators for :math:`k`-UpCCGSD ansatz.
        """

        assert t2 is None
        return generate_kupccgsd_ex2_ops(self.no, self.nv)

    @property
    def e_kupccgsd(self):
        """
        Returns :math:`k`-UpCCGSD energy
        """
        return self.energy()