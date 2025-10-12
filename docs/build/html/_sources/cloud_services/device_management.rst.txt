Device Management
=================

=================
Device Management
=================

TyxonQ provides device management capabilities for discovering, configuring, and monitoring quantum hardware through the cloud API.

Device Discovery
================

Listing Available Devices
-------------------------

Discover all quantum devices accessible with your API key:

.. code-block:: python

   import tyxonq as tq
   from tyxonq.cloud import apis
   
   # Configure authentication
   apis.set_token(YOUR_API_KEY)
   apis.set_provider("tyxonq")
   
   # List all TyxonQ devices
   devices = apis.list_devices()
   print("Available devices:", devices)
   # Output: ['tyxonq::homebrew_s2', ...]

Device Information
------------------

Retrieve detailed device specifications from the cloud:

.. code-block:: python

   from tyxonq.devices.hardware.tyxonq.driver import list_properties
   
   # Get device details
   props = list_properties(device="homebrew_s2", token=YOUR_API_KEY)
   
   for device in props:
       print(f"Device ID: {device['id']}")
       print(f"Name: {device['name']}")
       print(f"Qubits: {device['qubits']}")
       print(f"Queue length: {device['queue']}")
       print(f"Status: {device['status']}")

**Response Example**:

.. code-block:: python

   [
       {
           "id": "homebrew_s2",
           "name": "Homebrew S2",
           "qubits": 13,
           "queue": "quregenai.lab",
           "status": "online"
       }
   ]

Device Selection
================

Choosing the Right Device
-------------------------

**Local Simulators** (for development and testing):

.. code-block:: python

   from tyxonq.runtime.simulator import run
   
   # Statevector: Fast, exact, for small circuits (< 20 qubits)
   result = run(c, shots=100, method="statevector")
   
   # MPS: Scalable, for low-entanglement circuits
   result = run(c, shots=100, method="mps")

**Real Hardware** (for production):

.. code-block:: python

   from tyxonq.cloud import apis
   
   # TyxonQ Homebrew_S2 processor (13 qubits)
   task = apis.submit_task(
       circuit=c,
       shots=100,
       device="homebrew_s2"
   )
   result = task.results()

Device Optimization
===================

Hardware Optimization Flags
---------------------------

TyxonQ Homebrew_S2 supports **device-specific optimization** through query parameters. The optimization flags use **additive combination** (not independent levels):

**Available Optimization Flags**:

- ``o=1``: **Qubit mapping** (``enable_qos_qubit_mapping``) - Map logical qubits to physical qubits
- ``o=2``: **Gate decomposition** (``enable_qos_gate_decomposition``) - Decompose gates to native gate set
- ``o=4``: **Initial mapping** (``enable_qos_initial_mapping``) - Optimize initial qubit layout
- ``o=7``: **All optimizations** (1+2+4) - Enable all optimization features

**Important**: Flags are **additive**, not independent:

- ``o=3`` = qubit mapping (1) + gate decomposition (2)
- ``o=5`` = qubit mapping (1) + initial mapping (4)
- ``o=6`` = gate decomposition (2) + initial mapping (4)
- ``o=7`` = all three (1+2+4)

Method 1: Using Device String
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from tyxonq.cloud import apis
   
   # Qubit mapping only
   task = apis.submit_task(
       circuit=c,
       shots=100,
       device="homebrew_s2?o=1"
   )
   
   # Qubit mapping + gate decomposition (1+2=3)
   task = apis.submit_task(
       circuit=c,
       shots=100,
       device="homebrew_s2?o=3"
   )
   
   # All optimizations (1+2+4=7)
   task = apis.submit_task(
       circuit=c,
       shots=100,
       device="homebrew_s2?o=7"
   )

Method 2: Using Keyword Arguments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from tyxonq.cloud import apis
   
   # Enable specific optimizations via kwargs
   task = apis.submit_task(
       circuit=c,
       shots=100,
       device="homebrew_s2",
       enable_qos_qubit_mapping=True,       # o=1
       enable_qos_gate_decomposition=True,  # o=2
       enable_qos_initial_mapping=True      # o=4
   )
   # Equivalent to device="homebrew_s2?o=7"
   
   # Partial optimization
   task = apis.submit_task(
       circuit=c,
       shots=100,
       device="homebrew_s2",
       enable_qos_qubit_mapping=True,       # o=1
       enable_qos_gate_decomposition=False  # o=0
   )
   # Equivalent to device="homebrew_s2?o=1"

Optimization Selection Guide
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**When to use each optimization**:

.. code-block:: python

   # Pulse-level control: Disable all optimizations
   task = apis.submit_task(
       circuit=pulse_circuit,
       shots=100,
       device="homebrew_s2",  # No o= parameter
       enable_qos_gate_decomposition=False,
       enable_qos_qubit_mapping=False
   )
   
   # Gate-level with manual mapping: Only gate decomposition
   task = apis.submit_task(
       circuit=c,
       shots=100,
       device="homebrew_s2?o=2"
   )
   
   # Standard gate-level: Qubit mapping + gate decomposition
   task = apis.submit_task(
       circuit=c,
       shots=100,
       device="homebrew_s2?o=3"  # 1+2
   )
   
   # Best performance: All optimizations
   task = apis.submit_task(
       circuit=c,
       shots=100,
       device="homebrew_s2?o=7"  # 1+2+4
   )

Device Monitoring
=================

Queue Status
------------

Monitor device queue before submission:

.. code-block:: python

   from tyxonq.devices.hardware.tyxonq.driver import list_properties
   import time
   
   def wait_for_low_queue(max_queue=5, check_interval=30, token=None):
       """Wait until device queue is below threshold."""
       while True:
           props = list_properties(device="homebrew_s2", token=token)
           queue_length = props[0].get("queue", 0)
           
           print(f"Current queue: {queue_length} jobs")
           
           if queue_length <= max_queue:
               break
           
           print(f"Waiting {check_interval}s for queue to clear...")
           time.sleep(check_interval)
   
   # Wait for low queue before submitting
   wait_for_low_queue(max_queue=3, token=YOUR_API_KEY)
   
   from tyxonq.cloud import apis
   task = apis.submit_task(circuit=c, shots=100, device="homebrew_s2")

Device Health
-------------

Check device status before submission:

.. code-block:: python

   from tyxonq.devices.hardware.tyxonq.driver import list_properties
   
   props = list_properties(device="homebrew_s2", token=YOUR_API_KEY)
   device_info = props[0]
   
   # Check online status
   status = device_info.get("status")
   if status != "online":
       print(f"Warning: Device is {status}")
   else:
       print("Device is online and ready")
   
   # Check queue
   queue = device_info.get("queue", "N/A")
   print(f"Queue: {queue}")
   
   # Check qubit count
   qubits = device_info.get("qubits", 0)
   print(f"Available qubits: {qubits}")

Best Practices
==============

1. Check Device Status Before Submission
-----------------------------------------

.. code-block:: python

   from tyxonq.devices.hardware.tyxonq.driver import list_properties
   from tyxonq.cloud import apis
   
   def safe_submit(circuit, shots=100, token=None):
       """Submit with device status check."""
       # Check device availability
       props = list_properties(device="homebrew_s2", token=token)
       
       if not props or props[0].get("status") != "online":
           print("Device offline, using simulator")
           from tyxonq.runtime.simulator import run
           return run(circuit, shots=shots)
       
       # Submit to hardware
       task = apis.submit_task(
           circuit=circuit,
           shots=shots,
           device="homebrew_s2"
       )
       return task.results()

2. Use Appropriate Optimization Levels
--------------------------------------

.. code-block:: python

   from tyxonq.cloud import apis
   
   # Pulse-level experiments: No optimization
   task = apis.submit_task(
       circuit=pulse_circuit,
       shots=100,
       device="homebrew_s2",
       enable_qos_gate_decomposition=False,
       enable_qos_qubit_mapping=False
   )
   
   # Gate-level standard: Moderate optimization (o=3)
   task = apis.submit_task(
       circuit=gate_circuit,
       shots=100,
       device="homebrew_s2?o=3"
   )
   
   # Complex circuits: Full optimization (o=7)
   task = apis.submit_task(
       circuit=complex_circuit,
       shots=100,
       device="homebrew_s2?o=7"
   )

3. Handle Device Maintenance
----------------------------

.. code-block:: python

   from tyxonq.devices.hardware.tyxonq.driver import list_properties
   import time
   
   def wait_for_device_online(timeout=600, token=None):
       """Wait for device to come online after maintenance."""
       start_time = time.time()
       
       while time.time() - start_time < timeout:
           props = list_properties(device="homebrew_s2", token=token)
           
           if props and props[0].get("status") == "online":
               return True
           
           print("Device under maintenance, waiting...")
           time.sleep(30)
       
       return False

Troubleshooting
===============

**Device Not Found**

.. code-block:: python

   # Error: "Device 'homebrew_s2' not in available devices"
   
   from tyxonq.cloud import apis
   
   # Solution: Check your access permissions
   devices = apis.list_devices()
   print("You have access to:", devices)
   
   # If device not listed, request access at https://www.tyxonq.com/

**Queue Too Long**

.. code-block:: python

   from tyxonq.devices.hardware.tyxonq.driver import list_properties
   
   # Strategy 1: Wait for queue to clear
   wait_for_low_queue(max_queue=5)
   
   # Strategy 2: Use simulator instead
   from tyxonq.runtime.simulator import run
   result = run(c, shots=100)
   
   # Strategy 3: Schedule during off-peak hours
   # Check queue patterns over time

**Optimization Failures**

.. code-block:: python

   from tyxonq.cloud import apis
   
   # If optimization fails, try lower level
   try:
       task = apis.submit_task(
           circuit=c,
           shots=100,
           device="homebrew_s2?o=7"
       )
   except Exception:
       # Fallback to basic optimization
       task = apis.submit_task(
           circuit=c,
           shots=100,
           device="homebrew_s2?o=1"
       )

Next Steps
==========

- 📖 Read :doc:`task_submission` for detailed task management
- 📖 Read :doc:`api_reference` for complete API documentation  
- 📖 Read :doc:`hardware_access` for **Homebrew_S2 specifications** and **pulse-level programming**
