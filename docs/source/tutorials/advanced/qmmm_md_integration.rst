.. _qmmm_md_integration:

=========================================
QM/MM and Molecular Dynamics Integration
=========================================

Since v1.2.0, TyxonQ plugs into the molecular-dynamics (MD) ecosystem as a
**quantum-chemistry force provider**. The QM region is solved by TyxonQ's
correlated electronic solvers (UCCSD/HEA/SQD), while all nuclear gradients are
**reused from PySCF's analytic gradient machinery** — no gradient code is
hand-written.

.. note::
   All runnable versions of the snippets below live in ``examples/qmmm/``
   (tutorials E1–E9). Each example states its dependencies, expected runtime,
   and graceful exit behavior when a dependency is missing.

Overview
========

The single facade is :code:`qc_scanner` — a PySCF-gradient-scanner-compatible
callable returning :code:`(energy, gradient)` in atomic units (Hartree /
Hartree·Bohr⁻¹):

.. code-block:: python

   from tyxonq.applications.chem.interfaces import qc_scanner

   scan = qc_scanner("O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587",
                     basis="sto-3g", active_space=(4, 4), method="uccsd")
   e, de = scan(coords_bohr)      # rebuilds geometry per call, keeps recipe

Electrostatic embedding adds the MM environment to the QM Hamiltonian, with
per-step updates and MM back-reaction forces:

.. code-block:: python

   scan = qc_scanner(spec, basis="sto-3g", active_space=(4, 4), method="uccsd",
                     mm_charges=(mm_coords_bohr, mm_charges_e))  # cluster embedding
   e, de = scan(coords_bohr)
   de_mm = scan.mm_gradient()             # dE/dR_mm (force = -de_mm)
   scan.set_mm_charges(new_mm_coords)     # per-step environment update (MD loop)

Periodic systems use ``pyscf.qmmm.pbc`` Ewald embedding instead — pass
``mm_lattice`` (3×3 diagonal) plus ``rcut_ewald``/``rcut_hcore``; the scanner
enforces upstream's geometric validity guards at construction and on every
update.

Choosing the QM backend
=======================

``method=`` selects the electronic solver: ``"uccsd"`` (default for accuracy),
``"rouccsd"`` (open shell), ``"hea"`` (hardware-efficient ansatz), ``"sqd"``
(frozen-subspace selected CI). Measured on the water-dimer embedding benchmark
(CAS(4,4)/STO-3G), HEA sits ~7.5e-3 Ha above UCCSD — the expected
expressivity gap of an RY-only ansatz — while the full embedding/gradient
chain works identically.

HEA additionally forwards **execution options to the device layer**, so the
same QM/MM workflow runs on simulators or **real quantum hardware** without
code changes:

.. code-block:: python

   scan = qc_scanner(spec, basis="sto-3g", active_space=(4, 4),
                     method="hea",
                     solver_kwargs={"runtime": "device",
                                    "provider": "tyxonq",   # or qcos/quafu/simulator
                                    "device": "homebrew_s2",
                                    "shots": 2048})
   # real hardware: set the token first
   import tyxonq as tq
   tq.set_token("YOUR_API_KEY", provider="tyxonq")

Engine adapters
===============

Four thin adapters expose the same scanner to different MD topologies:

* **ASE** — ``TyxonQCalculator``: region partitioning via ``qm_indices``;
  full-system forces = QM gradient ⊕ MM back-reaction. Drives ASE optimizers
  and MD, and serves as an i-PI force field.
* **i-PI** — ``TyxonQDriver``: thin shell over ``ipi.pes._ase``; enables
  three-process runs: LAMMPS (``fix ipi``) + i-PI server + TyxonQ driver.
* **OpenMM** — ``create_tyxonq_system`` / mixed systems, and
  ``create_qmmm_ee_system`` for native electrostatic embedding (double-counting
  surgery: QM charges zeroed, intra-QM bonded terms removed, exceptions cleared).
* **MDI** — ``TyxonQMdiEngine``: ``@DEFAULT`` command table with the
  ``>NLATTICE/>CLATTICE/>LATTICE`` embedding channel; atomic units end to end.

Minimal ASE example
-------------------

.. code-block:: python

   from ase.build import molecule
   from tyxonq.applications.chem.interfaces import TyxonQCalculator

   atoms = molecule("H2O")
   atoms.calc = TyxonQCalculator(
       active_space=(4, 4), method="uccsd",
       qm_indices=[0, 1, 2],      # QM region (here: the whole molecule)
   )
   print(atoms.get_potential_energy())   # eV, ASE convention
   print(atoms.get_forces())             # eV/Angstrom

Installation
============

.. code-block:: bash

   pip install "tyxonq[md]"        # ase + openmm + openmmml
   pip install -U ipi              # optional: i-PI driver (package 'i-PI' is a stub!)
   pip install "pymdi>=1.4"        # optional: MDI line (imports as 'mdi')

Note the package-name traps: PyPI name ``openmmml`` (not ``openmm-ml``),
``ipi`` (not ``i-PI``), ``pymdi`` (imports as ``mdi``).

Known approximations
====================

MM back-reaction forces reuse upstream analytic expressions that lack
post-HF orbital-response terms (~4.3e-5 Ha/Bohr baseline bias vs finite
difference; smooth in geometry). Thermostatted MD is fully supported;
**strict NVE energy-conservation diagnostics are not valid** on embedded
paths. See ``examples/qmmm/md_lammps_qmmm_pbc/VALIDATION.md`` for the full
validation record.

Tutorial map (examples/qmmm/)
=============================

=====  =====================================  =================================================
ID     File                                   Content
=====  =====================================  =================================================
E1/E2  ``md_qc_scanner_basics.py`` …        scanner basics + vibrational analysis (UCCSD/HEA)
E3     ``md_ase_optimize_and_md.py``        ASE optimization and MD with ``TyxonQCalculator``
E4     ``md_ipi_driver_server.py``          i-PI driver (classical MD + PIMD variants)
E5     ``md_openmm_pure_qm.py``             pure-QM MD inside OpenMM
E6     ``md_openmm_qmmm_solvated.py``       OpenMM QM/MM with region partitioning
E6b    ``md_openmm_qmmm_electrostatic.py``  OpenMM native electrostatic embedding (no MDI)
E7     ``md_lammps_fix_ipi/``               LAMMPS + i-PI + TyxonQ three-process pipeline
E8-A   ``md_lammps_qmmm_embedded/``         production cluster QM/MM (three processes)
E8-B   ``md_lammps_qmmm_pbc/``              periodic Ewald QM/MM + validation archive
E9     ``md_mdi_qmmm_embedded.py``          MDI dedicated line (Python driver, two processes)
=====  =====================================  =================================================

See Also
========

* ``MD_INTEGRATION_PLAN.md`` in the repository root — full design and acceptance record
* ``tests_applications_chem/`` — use cases 1–12 (39 tests, all green with ``method="uccsd"``)
* :doc:`../../quantum_chemistry/index` - Quantum chemistry documentation
