============
Installation
============

This guide covers installing TyxonQ on different platforms and configurations.

Quick Install
=============

The simplest way to install TyxonQ is via pip:

.. code-block:: bash

   pip install tyxonq

This will install TyxonQ with default dependencies using NumPy backend.

Installation Options
====================

TyxonQ supports several optional dependencies for enhanced functionality:

.. tab-set::

   .. tab-item:: Standard Installation
      
      Install TyxonQ with NumPy backend (CPU only):

      .. code-block:: bash

         pip install tyxonq

   .. tab-item:: PyTorch Support
      
      Install with PyTorch backend for automatic differentiation:

      .. code-block:: bash

         pip install tyxonq[torch]

   .. tab-item:: GPU Acceleration
      
      Install with CuPy for GPU-accelerated simulations:

      .. code-block:: bash

         pip install tyxonq[gpu]

   .. tab-item:: Full Installation
      
      Install all optional dependencies:

      .. code-block:: bash

         pip install tyxonq[all]

System Requirements
===================

**Minimum Requirements**

- Python 3.9 or higher
- 4GB RAM
- 2GB disk space

**Recommended Requirements**

- Python 3.10 or higher
- 8GB RAM (16GB for large-scale simulations)
- NVIDIA GPU with CUDA support (for GPU acceleration)
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

For GPU support on Linux:

.. code-block:: bash

   # Install CUDA toolkit (example for CUDA 11.8)
   # Follow NVIDIA's installation guide for your specific version
   
   # Install TyxonQ with GPU support
   pip install tyxonq[gpu]

macOS
-----

Using Homebrew:

.. code-block:: bash

   # Install Python (if not already installed)
   brew install python@3.11

   # Install TyxonQ
   pip install tyxonq

.. note::
   GPU acceleration via CUDA is not available on macOS. However, you can use the PyTorch backend with MPS (Metal Performance Shaders) on Apple Silicon:

   .. code-block:: bash

      pip install tyxonq[torch]

Windows
-------

.. code-block:: powershell

   # Ensure Python is installed (download from python.org)
   
   # Install TyxonQ
   pip install tyxonq

For GPU support on Windows:

.. code-block:: powershell

   # Install CUDA toolkit from NVIDIA
   # Then install TyxonQ with GPU support
   pip install tyxonq[gpu]

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

Docker Installation
===================

TyxonQ provides Docker images for reproducible environments:

.. code-block:: bash

   # Pull the latest image
   docker pull quregenai/tyxonq:latest

   # Run TyxonQ in a container
   docker run -it quregenai/tyxonq:latest python

   # With GPU support
   docker run --gpus all -it quregenai/tyxonq:gpu python

Conda Installation
==================

TyxonQ can be installed via Conda:

.. code-block:: bash

   # Create a new environment
   conda create -n tyxonq python=3.10
   conda activate tyxonq

   # Install TyxonQ
   conda install -c conda-forge tyxonq

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
   
   # Run on simulator
   result = circuit.compile().device('statevector').run(shots=100)
   print("Installation successful!")
   print(result.counts)

Expected output:

.. code-block:: text

   0.1.0
   Installation successful!
   {'00': 50, '11': 50}

Troubleshooting
===============

Import Errors
-------------

If you encounter import errors:

.. code-block:: bash

   # Reinstall with --force-reinstall
   pip install --force-reinstall tyxonq

GPU Issues
----------

If GPU acceleration is not working:

.. code-block:: python

   import tyxonq as tq
   
   # Check GPU availability
   print(tq.numerics.gpu_available())
   
   # Check CUDA version
   import cupy
   print(cupy.cuda.runtime.runtimeGetVersion())

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
