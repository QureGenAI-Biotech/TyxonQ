======================
Libraries & Components
======================

TyxonQ provides a rich ecosystem of reusable libraries and components that accelerate quantum algorithm development. These libraries range from low-level quantum kernels to high-level circuit templates, enabling both researchers and practitioners to build sophisticated quantum applications efficiently.

.. contents:: Table of Contents
   :depth: 2
   :local:

Overview
========

The TyxonQ library ecosystem is organized into four main categories:

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: 🎨 Circuit Templates
      :link: circuits_library/index
      :link-type: doc

      Pre-built parameterized circuits for VQE, QAOA, UCC, and other quantum algorithms.

   .. grid-item-card:: ⚛️ Quantum Kernels
      :link: quantum_library/index
      :link-type: doc

      Low-level quantum computational kernels for statevector, density matrix, and MPS simulations.

   .. grid-item-card:: 🧬 Hamiltonian Encoding
      :link: hamiltonian_encoding/index
      :link-type: doc

      Tools for fermion-to-qubit mappings, Pauli term grouping, and efficient Hamiltonian representation.

   .. grid-item-card:: 📈 Optimizers
      :link: optimizer/index
      :link-type: doc

      Advanced optimization algorithms including SOAP and interoperability with SciPy.

Architectural Philosophy
========================

The library design follows these principles:

**Modularity**
   Each library is self-contained and can be used independently

**Composability**
   Libraries work together seamlessly for complex workflows

**Performance**
   Optimized implementations with GPU support where applicable

**Flexibility**
   Extensible APIs allowing custom components

**Integration**
   Deep integration with TyxonQ's compiler and device abstraction

.. toctree::
   :maxdepth: 2

   circuits_library/index
   quantum_library/index
   hamiltonian_encoding/index
   optimizer/index
