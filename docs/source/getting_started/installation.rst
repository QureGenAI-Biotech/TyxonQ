============
Installation
============

This guide covers installing TyxonQ on different platforms.

Quick Install
=============

The simplest way to install TyxonQ is via pip:

.. code-block:: bash

   pip install tyxonq

This will install TyxonQ with all required dependencies.

System Requirements
===================

**Minimum Requirements**

- Python 3.10, 3.11, or 3.12
- 4GB RAM
- 2GB disk space

**Recommended Requirements**

- Python 3.11 or higher
- 8GB RAM (16GB for large-scale simulations)
- 10GB disk space

Platform-Specific Instructions
===============================

Linux
-----

For Ubuntu/Debian:

.. code-block:: bash

   # Install system dependencies
   sudo apt-get update
   sudo apt-get install python3-dev python3-pip

   # Install TyxonQ
   pip install tyxonq

macOS
-----

Using Homebrew:

.. code-block:: bash

   # Install Python (if not already installed)
   brew install python@3.11

   # Install TyxonQ
   pip install tyxonq

Windows
-------

.. code-block:: powershell

   # Ensure Python is installed (download from python.org)
   
   # Install TyxonQ
   pip install tyxonq

Development Installation
=========================

To contribute to TyxonQ or use the latest development version:

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/QureGenAI-Biotech/TyxonQ.git
   cd TyxonQ

   # Install in editable mode with development dependencies
   pip install -e ".[dev]"

   # Run tests to verify installation
   pytest tests/

Verifying Installation
=======================

After installation, verify that TyxonQ is working correctly:

.. code-block:: python

   import tyxonq as tq
   
   # Check version
   print(tq.__version__)
   
   # Create a simple circuit to test
   circuit = tq.Circuit(2)
   circuit.h(0)
   circuit.cnot(0, 1)
   
   print("Installation successful!")
   print(f"Created circuit with {circuit.num_qubits} qubits")

Expected output:

.. code-block:: text

   0.9.9
   Installation successful!
   Created circuit with 2 qubits

Troubleshooting
===============

Import Errors
-------------

If you encounter import errors:

.. code-block:: bash

   # Reinstall with --force-reinstall
   pip install --force-reinstall tyxonq

Dependency Conflicts
--------------------

If you have dependency conflicts:

.. code-block:: bash

   # Create a fresh virtual environment
   python -m venv tyxonq_env
   source tyxonq_env/bin/activate  # On Windows: tyxonq_env\Scripts\activate
   
   # Install TyxonQ
   pip install tyxonq

Upgrading TyxonQ
================

To upgrade to the latest version:

.. code-block:: bash

   pip install --upgrade tyxonq

To upgrade to a specific version:

.. code-block:: bash

   pip install tyxonq==0.2.0

Uninstalling
============

To remove TyxonQ:

.. code-block:: bash

   pip uninstall tyxonq

Next Steps
==========

Now that you have TyxonQ installed, proceed to:

- :doc:`quickstart` - Get started in 5 minutes
- :doc:`first_circuit` - Create your first quantum circuit
- :doc:`basic_concepts` - Learn the fundamental concepts

For more help, see:

- :doc:`../faq` - Frequently asked questions
- `GitHub Issues <https://github.com/QureGenAI-Biotech/TyxonQ/issues>`_ - Report problems
