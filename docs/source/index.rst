==========================================
TyxonQ: Quantum Computing Framework
==========================================

**TyxonQ** is a powerful, flexible quantum computing framework designed for researchers and developers working on quantum algorithms, quantum chemistry applications, and AI-driven drug discovery (AIDD). Built with performance and extensibility in mind, TyxonQ provides a complete ecosystem for quantum circuit design, compilation, simulation, and execution on real quantum hardware.

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: 🚀 Getting Started
      :link: getting_started/index
      :link-type: doc

      New to TyxonQ? Start here with installation guides, quickstart tutorials, and basic concepts.

   .. grid-item-card:: 📚 User Guide
      :link: user_guide/index
      :link-type: doc

      Comprehensive guides covering core concepts, compiler pipeline, devices, and advanced features.

   .. grid-item-card:: 🧬 Quantum Chemistry
      :link: quantum_chemistry/index
      :link-type: doc

      Specialized documentation for quantum chemistry applications, including VQE, UCCSD, and molecular simulations.

   .. grid-item-card:: 🎓 Tutorials
      :link: tutorials/index
      :link-type: doc

      Step-by-step tutorials from beginner to advanced topics with executable examples.

   .. grid-item-card:: 📖 API Reference
      :link: api/index
      :link-type: doc

      Complete API documentation with detailed class and function references.

   .. grid-item-card:: ☁️ Cloud Services
      :link: cloud_services/index
      :link-type: doc

      Access quantum hardware and cloud computing resources through TyxonQ's cloud API.

Key Features
============

🎯 **Comprehensive Framework**
   Full-stack quantum computing solution from circuit design to hardware execution

⚡ **High Performance**
   Optimized compilation pipeline with support for GPU acceleration and distributed computing

🔧 **Flexible Backend System**
   Multiple numerical backends (NumPy, PyTorch, CuPyNumeric) for different use cases

🧪 **Quantum Chemistry Focus**
   Specialized tools for molecular simulations, VQE, UCCSD, and drug discovery applications

🌐 **Cloud Integration**
   Seamless access to quantum hardware including the Homebrew_S2 quantum processor

🎨 **Extensible Architecture**
   Plugin system and clean abstractions for custom devices, backends, and algorithms

Quick Example
=============

Here's a simple example of creating and running a quantum circuit:

.. code-block:: python

   import tyxonq as tq

   # Create a 2-qubit circuit
   circuit = tq.Circuit(2)
   circuit.h(0)
   circuit.cnot(0, 1)
   circuit.measure_all()

   # Compile and execute
   compiled = circuit.compile()
   result = compiled.device('statevector').run(shots=1000)
   
   print(result.counts)
   # Output: {'00': 500, '11': 500} (approximately)

For more examples, see the :doc:`getting_started/quickstart` guide.

Documentation Structure
=======================

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   getting_started/index
   getting_started/installation
   getting_started/quickstart
   getting_started/first_circuit
   getting_started/first_chemistry
   getting_started/basic_concepts

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user_guide/index
   user_guide/core/index
   user_guide/compiler/index
   user_guide/devices/index
   user_guide/numerics/index
   user_guide/postprocessing/index
   user_guide/advanced/index

.. toctree::
   :maxdepth: 2
   :caption: Quantum Chemistry

   quantum_chemistry/index
   quantum_chemistry/fundamentals/index
   quantum_chemistry/algorithms/index
   quantum_chemistry/molecule/index
   quantum_chemistry/runtimes/index
   quantum_chemistry/aidd/index

.. toctree::
   :maxdepth: 2
   :caption: Libraries & Components

   libraries/index
   libraries/circuits_library/index
   libraries/quantum_library/index
   libraries/hamiltonian_encoding/index
   libraries/optimizer/index

.. toctree::
   :maxdepth: 2
   :caption: Cloud Services

   cloud_services/index
   cloud_services/getting_started
   cloud_services/device_management
   cloud_services/task_submission
   cloud_services/api_reference
   cloud_services/hardware_access

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   tutorials/index
   tutorials/beginner/index
   tutorials/intermediate/index
   tutorials/advanced/index

.. toctree::
   :maxdepth: 2
   :caption: Examples

   examples/index
   examples/basic_examples
   examples/chemistry_examples
   examples/optimization_examples
   examples/cloud_examples
   examples/advanced_examples

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/index
   api/core/index
   api/compiler/index
   api/devices/index
   api/numerics/index
   api/postprocessing/index
   api/applications/index
   api/libs/index
   api/cloud/index
   api/utils/index

.. toctree::
   :maxdepth: 2
   :caption: Developer Guide

   developer_guide/index
   developer_guide/contributing
   developer_guide/architecture_overview
   developer_guide/extending_tyxonq
   developer_guide/custom_devices
   developer_guide/custom_backends
   developer_guide/plugin_system
   developer_guide/testing_guidelines

.. toctree::
   :maxdepth: 2
   :caption: Technical References

   technical_references/index
   technical_references/whitepaper
   technical_references/architecture_design
   technical_references/performance_optimization
   technical_references/comparison_with_other_frameworks
   technical_references/research_papers

.. toctree::
   :maxdepth: 1
   :caption: Additional Resources

   faq
   glossary
   changelog
   bibliography

Community & Support
===================

- **GitHub Repository**: `QureGenAI-Biotech/TyxonQ <https://github.com/QureGenAI-Biotech/TyxonQ>`_
- **Issue Tracker**: Report bugs and request features on `GitHub Issues <https://github.com/QureGenAI-Biotech/TyxonQ/issues>`_
- **PyPI Package**: `tyxonq <https://pypi.org/project/tyxonq>`_
- **Contributing**: See our :doc:`developer_guide/contributing` guide

License
=======

TyxonQ is released under the Apache License 2.0. See the `LICENSE <https://github.com/QureGenAI-Biotech/TyxonQ/blob/main/LICENSE>`_ file for details.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
