===========
Devices API
===========

Complete API reference for TyxonQ's device management and execution system.

.. contents:: Contents
   :depth: 3
   :local:

Overview
========

The Devices API provides interfaces for quantum hardware and simulators:

💻 **Local Simulators**
   Statevector, matrix product state, and density matrix simulators

☁️ **Cloud Devices**
   Access to TyxonQ cloud platform and quantum hardware

🔧 **Device Configuration**
   Provider, device, and execution settings

Device Execution
================

Basic Usage
-----------

.. code-block:: python

   from tyxonq import Circuit
   
   circuit = Circuit(2).h(0).cx(0, 1)
   
   # Run on local statevector simulator
   result = circuit.device(
       provider="local",
       device="statevector",
       shots=1024
   ).run()
   
   print(result)  # {'00': ~512, '11': ~512}

Method Chaining
---------------

.. code-block:: python

   # Chain device configuration with compilation
   result = (
       Circuit(3)
       .h(0)
       .cx(0, 1)
       .cx(1, 2)
       .compile(optimization_level=2)
       .device(provider="local", device="statevector", shots=0)
       .run()
   )

Local Simulators
================

Statevector Simulator
---------------------

**Best for**: Small to medium circuits (up to ~25 qubits)

.. code-block:: python

   # Exact simulation (no shot noise)
   circuit.device(
       provider="local",
       device="statevector",
       shots=0  # Exact expectation values
   ).run()
   
   # Shot-based sampling
   circuit.device(
       provider="local",
       device="statevector",
       shots=1024
   ).run()

**Characteristics**:

- Memory: O(2^n) for n qubits
- Speed: Very fast for ≤20 qubits
- Noise: Optional depolarizing noise
- Supports: All gates, measurements

Matrix Product State
--------------------

**Best for**: Larger circuits with low entanglement

.. code-block:: python

   circuit.device(
       provider="local",
       device="mps",  # or "matrix_product_state"
       shots=1024
   ).run()

**Characteristics**:

- Memory: Depends on entanglement
- Speed: Good for low-entanglement circuits
- Best for: 1D quantum systems, shallow circuits

Density Matrix Simulator
------------------------

**Best for**: Open quantum systems, mixed states

.. code-block:: python

   circuit.device(
       provider="local",
       device="density_matrix",
       shots=1024
   ).run()

**Characteristics**:

- Memory: O(4^n) for n qubits
- Supports: Noise models, mixed states
- Best for: Small systems with decoherence

Cloud Execution
===============

TyxonQ Cloud Platform
---------------------

.. code-block:: python

   import tyxonq as tq
   
   # Set API credentials
   tq.set_token(
       token="your-api-key",
       provider="tyxonq",
       device="homebrew_s2"
   )
   
   # Run on cloud
   result = (
       Circuit(2)
       .h(0)
       .cx(0, 1)
       .device(provider="tyxonq", device="homebrew_s2", shots=1024)
       .run()
   )

**Supported Devices**:

- ``homebrew_s2``: TyxonQ superconducting processor
- More devices coming soon

Device Configuration
====================

Global Defaults
---------------

Set global device defaults:

.. code-block:: python

   import tyxonq as tq
   
   # Set global defaults
   tq.device(
       provider="local",
       device="statevector",
       shots=2048
   )
   
   # All subsequent runs use these defaults
   result1 = Circuit(2).h(0).cx(0, 1).run()
   result2 = Circuit(3).h(0).h(1).cx(0, 2).run()
   
   # Override for specific run
   result3 = Circuit(2).h(0).device(shots=4096).run()

Retrieve Defaults
-----------------

.. code-block:: python

   # Get current defaults
   defaults = tq.get_device_defaults()
   print(defaults)
   # {'provider': 'local', 'device': 'statevector', 'shots': 2048}

Shot Configuration
==================

Exact Simulation
----------------

Use ``shots=0`` for exact expectation values:

.. code-block:: python

   # No sampling noise
   circuit.device(shots=0).run()

**Use when**:
- Algorithm development
- Debugging
- Maximum accuracy needed

Sampling-Based
--------------

Use ``shots > 0`` for realistic quantum behavior:

.. code-block:: python

   # Different shot counts
   circuit.device(shots=512).run()   # Quick, noisy
   circuit.device(shots=2048).run()  # Balanced (default)
   circuit.device(shots=8192).run()  # High precision

**Statistical Error**:

.. math::

   \sigma \propto \frac{1}{\sqrt{\text{shots}}}

Noise Modeling
==============

Depolarizing Noise
------------------

.. code-block:: python

   from tyxonq.devices import enable_noise
   
   # Enable depolarizing noise
   enable_noise(
       enabled=True,
       config={
           "type": "depolarizing",
           "p": 0.01  # Error probability
       }
   )
   
   # Run with noise
   result = circuit.device(
       provider="local",
       device="statevector"
   ).run()

Custom Noise Models
-------------------

.. code-block:: python

   # Configure custom noise
   enable_noise(
       enabled=True,
       config={
           "type": "custom",
           "single_qubit_error": 0.001,
           "two_qubit_error": 0.01,
           "readout_error": 0.02
       }
   )

Disable Noise
-------------

.. code-block:: python

   from tyxonq.devices import enable_noise
   
   # Disable noise
   enable_noise(enabled=False)

Device Information
==================

List Available Devices
----------------------

.. code-block:: python

   from tyxonq.devices import list_devices
   
   # List local simulators
   local_devices = list_devices(provider="local")
   print(local_devices)
   # ['statevector', 'mps', 'matrix_product_state', 'density_matrix']
   
   # List cloud devices
   cloud_devices = list_devices(
       provider="tyxonq",
       token="your-api-key"
   )
   print(cloud_devices)

Device Properties
-----------------

.. code-block:: python

   from tyxonq.devices import get_device_properties
   
   # Get device specs
   props = get_device_properties(
       provider="tyxonq",
       device="homebrew_s2"
   )
   
   print(f"Qubits: {props['n_qubits']}")
   print(f"Topology: {props['coupling_map']}")
   print(f"Gate fidelities: {props['gate_fidelities']}")

Advanced Features
=================

Batch Execution
---------------

Submit multiple circuits:

.. code-block:: python

   circuits = [
       Circuit(2).h(0).cx(0, 1),
       Circuit(2).h(0).h(1),
       Circuit(2).cx(0, 1)
   ]
   
   # Submit batch
   from tyxonq.devices import run
   
   tasks = run(
       provider="local",
       device="statevector",
       circuit=circuits,
       shots=1024
   )
   
   # Get results
   for task in tasks:
       print(task.result())

Asynchronous Execution
----------------------

.. code-block:: python

   # Submit without waiting
   task = circuit.device(
       provider="tyxonq",
       device="homebrew_s2",
       shots=2048
   ).submit()
   
   # Do other work...
   
   # Retrieve result later
   result = task.result(wait=True, timeout=60)

Result Caching
--------------

.. code-block:: python

   # Enable result caching
   circuit.device(
       provider="local",
       device="statevector",
       cache_results=True
   ).run()
   
   # Subsequent identical runs use cache
   result = circuit.run()  # Fast, from cache

Common Patterns
===============

Development Workflow
--------------------

.. code-block:: python

   # 1. Develop with exact simulation
   circuit = Circuit(3).h(0).cx(0, 1).cx(1, 2)
   
   exact_result = circuit.device(
       provider="local",
       device="statevector",
       shots=0  # Exact
   ).run()
   
   # 2. Test with shots
   test_result = circuit.device(shots=1024).run()
   
   # 3. Run on cloud
   import tyxonq as tq
   tq.set_token(token="your-key")
   
   cloud_result = circuit.device(
       provider="tyxonq",
       device="homebrew_s2",
       shots=2048
   ).run()

Performance Optimization
------------------------

.. code-block:: python

   # Choose simulator based on circuit size
   n_qubits = circuit.num_qubits
   
   if n_qubits <= 20:
       # Use statevector for small circuits
       device = "statevector"
       shots = 0  # Exact
   elif n_qubits <= 30:
       # Use MPS for larger low-entanglement
       device = "mps"
       shots = 2048
   else:
       # Use cloud for very large
       device = "homebrew_s2"
       shots = 4096
   
   result = circuit.device(
       provider="local" if device != "homebrew_s2" else "tyxonq",
       device=device,
       shots=shots
   ).run()

Error Handling
--------------

.. code-block:: python

   try:
       result = circuit.device(
           provider="tyxonq",
           device="homebrew_s2",
           shots=2048
       ).run()
   except TimeoutError:
       print("Device queue timeout")
   except ConnectionError:
       print("Network error, falling back to local")
       result = circuit.device(
           provider="local",
           device="statevector"
       ).run()

See Also
========

- :doc:`/user_guide/devices/index` - Devices User Guide
- :doc:`/api/core/index` - Core API
- :doc:`/cloud_services/index` - Cloud Services
- :doc:`/examples/basic_examples` - Device Examples
