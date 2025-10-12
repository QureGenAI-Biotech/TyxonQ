==========================================
TyxonQ: Quantum Computing Framework
==========================================

**TyxonQ** is a powerful, flexible quantum computing framework designed for researchers and developers working on quantum algorithms, quantum chemistry applications, and AI-driven drug discovery (AIDD). Built with performance and extensibility in mind, TyxonQ provides a complete ecosystem for quantum circuit design, compilation, simulation, and execution on real quantum hardware.

.. grid:: 3
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

   .. grid-item-card:: 📚 Libraries
      :link: libraries/index
      :link-type: doc

      Reusable components including circuit templates, quantum kernels, and optimization tools.

   .. grid-item-card:: 💡 Examples
      :link: examples/index
      :link-type: doc

      Practical examples demonstrating quantum algorithms, chemistry calculations, and real-world applications.

   .. grid-item-card:: 🔧 Developer Guide
      :link: developer_guide/index
      :link-type: doc

      Contributing guidelines, architecture overview, and extending TyxonQ with custom components.

   .. grid-item-card:: 📄 Technical References
      :link: technical_references/index
      :link-type: doc

      Whitepapers, architecture design, performance optimization, and research publications.

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

Here are simple examples showing both quantum circuit construction and quantum chemistry applications:

**Basic Quantum Circuit**

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

**Quantum Chemistry Calculation**

.. code-block:: python

   from tyxonq.applications.chem import UCCSD, HEA
   from tyxonq.applications.chem.molecule import h2

   # UCCSD calculation for H2 molecule
   uccsd = UCCSD(h2, run_fci=True)
   uccsd_energy = uccsd.kernel(runtime="numeric")
   
   # Hardware Efficient Ansatz (HEA) calculation
   hea = HEA.ry(uccsd.int1e, uccsd.int2e, uccsd.n_elec, uccsd.e_core, 
                layers=3, runtime="device")
   hea_energy = hea.kernel(shots=0, provider="simulator", device="statevector")
   
   print(f"HF energy:    {h2.hf_energy:.6f} Hartree")
   print(f"UCCSD energy: {uccsd_energy:.6f} Hartree")
   print(f"HEA energy:   {hea_energy:.6f} Hartree")
   # Expected output shows correlation energy capture

For more examples, see the :doc:`getting_started/quickstart` guide.

Main Navigation
===============

.. toctree::
   :maxdepth: 1
   :caption: Core Documentation

   getting_started/index
   user_guide/index
   quantum_chemistry/index
   tutorials/index

.. toctree::
   :maxdepth: 1
   :caption: Libraries & Tools

   libraries/index
   cloud_services/index
   examples/index
   api/index

.. toctree::
   :maxdepth: 1
   :caption: Development & References

   developer_guide/index
   technical_references/index

.. toctree::
   :maxdepth: 1
   :hidden:

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
