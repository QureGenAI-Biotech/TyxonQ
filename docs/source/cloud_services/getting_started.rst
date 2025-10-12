Getting Started
===============

===============
Getting Started
===============

TyxonQ provides seamless access to real quantum hardware through cloud services, enabling you to run quantum algorithms on actual quantum processors with **gate-level** and **pulse-level** control.

Overview
========

**TyxonQ Quantum Cloud Services** powered by **QureGenAI** give you access to:

- **Homebrew_S2** quantum processor - 13-qubit superconducting quantum computer
- **Gate-level execution** - Standard gate model quantum computing
- **Pulse-level control** - Direct microwave pulse programming via TQASM 0.2
- **Real-time task submission** - Submit and monitor quantum jobs via REST API
- **Unified programming model** - Same code runs on simulators and hardware
- **Hardware optimization** - Qubit mapping, gate decomposition, initial mapping

Obtaining API Access
====================

1. Register Account
-------------------

Visit the **TyxonQ Quantum Cloud Portal**:

🌐 https://www.tyxonq.com/

- Create an account with email verification
- Complete the registration process
- Access your user dashboard

2. Request API Key
------------------

From your dashboard:

1. Navigate to **API Keys** section
2. Click **Generate New Key**
3. Copy and save your API key (format: "username;priority")
4. **Important**: Keep your API key confidential - treat it like a password

3. Request Hardware Access
--------------------------

**Homebrew_S2** quantum processor access:

- Submit hardware access request through portal at https://www.tyxonq.com/
- Provide research/project description
- Await approval notification (typically 1-3 business days)
- Once approved, you can submit jobs to ``homebrew_s2`` device
- **Pulse-level access**: Requires additional approval for TQASM 0.2 programming

API Configuration
=================

Setting Your Token
------------------

**Method 1: In Code (Recommended for Testing)**

.. code-block:: python

   import tyxonq as tq
   from tyxonq.cloud import apis
   import getpass
   
   # Prompt for API key (secure input)
   API_KEY = getpass.getpass("Input your TyxonQ API_KEY: ")
   apis.set_token(API_KEY)
   apis.set_provider("tyxonq")
   
   # Verify configuration
   print("Token configured successfully!")

**Method 2: Environment Variable (Recommended for Production)**

.. code-block:: bash

   # Unix/Linux/macOS
   export TYXONQ_API_KEY="username;priority"
   
   # Windows PowerShell
   $env:TYXONQ_API_KEY="username;priority"

Then in Python:

.. code-block:: python

   import tyxonq as tq
   from tyxonq.cloud import apis
   
   # Token loaded automatically from environment
   apis.set_provider("tyxonq")

Your First Quantum Job
======================

Gate-Level: Bell State on Real Hardware
----------------------------------------

.. code-block:: python

   import tyxonq as tq
   from tyxonq.cloud import apis
   import getpass
   
   # Configure API key
   API_KEY = getpass.getpass("Input your TyxonQ API_KEY: ")
   apis.set_token(API_KEY)
   apis.set_provider("tyxonq")
   
   # Create Bell state circuit
   c = tq.Circuit(2)
   c.h(0)
   c.cx(0, 1)
   c.measure_z(0)
   c.measure_z(1)
   
   # Submit to real quantum hardware
   task = apis.submit_task(
       circuit=c,
       shots=100,
       device="homebrew_s2",
       enable_qos_qubit_mapping=True,
       enable_qos_gate_decomposition=True
   )
   
   # Wait and get results
   result = task.results()
   counts = result["result"]
   print("Measurement counts:", counts)
   # Expected: {"00": ~50, "11": ~50} (entangled state with hardware noise)

**Output Example**:

.. code-block:: text

   Measurement counts: {'00': 47, '11': 53}

Pulse-Level: Rabi Experiment on Real Hardware
----------------------------------------------

TyxonQ supports **pulse-level control** through TQASM 0.2 format:

.. code-block:: python

   import tyxonq as tq
   from tyxonq import Param, waveforms
   from tyxonq.cloud import apis
   import getpass
   
   # Configure API
   API_KEY = getpass.getpass("Input your TyxonQ API_KEY: ")
   apis.set_token(API_KEY)
   apis.set_provider("tyxonq")
   
   # Create pulse-level circuit
   qc = tq.Circuit(1)
   qc.use_pulse()  # Enable pulse mode
   
   # Define parametric calibration
   param = Param("q[0]")
   builder = qc.calibrate("rabi_test", [param])
   builder.new_frame("drive_frame", param)
   builder.play("drive_frame", waveforms.CosineDrag(50, 0.2, 0.0, 0.0))
   builder.build()
   
   # Add calibration call
   qc.add_calibration('rabi_test', ['q[0]'])
   qc.measure_z(0)
   
   # View generated TQASM 0.2 code
   print(qc.to_tqasm())
   # Output:
   # TQASM 0.2;
   # QREG q[1];
   # defcal rabi_test q[0] {
   #   frame drive_frame = newframe(q[0]);
   #   play(drive_frame, cosine_drag(50, 0.2, 0.0, 0.0));
   # }
   # rabi_test q[0];
   # MEASZ q[0];
   
   # Submit to hardware
   task = apis.submit_task(
       circuit=qc,
       shots=100,
       device="homebrew_s2",
       enable_qos_gate_decomposition=False,  # Keep pulse-level
       enable_qos_qubit_mapping=False
   )
   
   result = task.results()
   print("Rabi oscillation:", result["result"])

**Supported Waveforms**:

- ``CosineDrag(duration, amp, sigma, beta)`` - Cosine DRAG pulse
- ``Flattop(duration, amp, sigma)`` - Flat-top Gaussian
- ``Gaussian(duration, amp, sigma)`` - Gaussian envelope
- ``Sine(duration, amp, phase, freq, angle)`` - Sine wave
- ``Constant(duration, amp)`` - Constant amplitude
- ``GaussianSquare(duration, amp, sigma, width)`` - Gaussian square
- ``Cosine(duration, amp, freq, phase)`` - Cosine wave
- ``Drag(duration, amp, sigma, beta)`` - DRAG pulse

Comparing Simulator vs. Hardware
---------------------------------

.. code-block:: python

   import tyxonq as tq
   from tyxonq.cloud import apis
   
   # Same circuit
   c = tq.Circuit(2).h(0).cx(0, 1).measure_z(0).measure_z(1)
   
   # Run on local simulator (ideal)
   from tyxonq.runtime.simulator import run
   sim_result = run(c, shots=100)
   sim_counts = sim_result[0].get("result", {})
   
   # Run on real hardware (noisy)
   task = apis.submit_task(
       circuit=c,
       shots=100,
       device="homebrew_s2"
   )
   hw_counts = task.results()["result"]
   
   print("Simulator:", sim_counts)
   print("Hardware: ", hw_counts)
   # Simulator: Perfect {"00": 50, "11": 50}
   # Hardware:  Noisy {"00": 47, "01": 1, "10": 2, "11": 50}

Asynchronous Job Submission
---------------------------

For long-running jobs, use asynchronous mode:

.. code-block:: python

   import tyxonq as tq
   from tyxonq.cloud import apis
   import time
   
   c = tq.Circuit(4).h([0,1,2,3]).cx(0,1).cx(2,3).measure_all()
   
   # Submit without waiting
   task = apis.submit_task(
       circuit=c,
       shots=100,
       device="homebrew_s2"
   )
   
   print(f"Job submitted: {task.id}")
   print(f"Initial status: {task.status}")
   
   # Do other work...
   time.sleep(10)
   
   # Check if complete
   details = task.get_result(wait=False)
   print(f"Status: {details.get('result_meta', {}).get('raw', {}).get('task', {}).get('state')}")
   
   # Wait for completion with timeout
   final_result = task.get_result(wait=True, timeout=60)
   counts = final_result.get("result", {})
   print("Results:", counts)

Device Availability
===================

List Available Devices
----------------------

.. code-block:: python

   import tyxonq as tq
   from tyxonq.cloud import apis
   
   # Configure token first
   apis.set_token(YOUR_API_KEY)
   apis.set_provider("tyxonq")
   
   # List all available devices
   devices = apis.list_devices()
   print("Available devices:", devices)
   
   # Output: ['tyxonq::homebrew_s2', ...]

Check Device Properties
-----------------------

Retrieve detailed device specifications from the cloud:

.. code-block:: python

   from tyxonq.devices.hardware.tyxonq.driver import list_properties
   
   # Get all device properties
   props = list_properties(device="homebrew_s2", token=YOUR_API_KEY)
   
   for device in props:
       print(f"Device: {device['id']}")
       print(f"  Name: {device['name']}")
       print(f"  Qubits: {device['qubits']}")
       print(f"  Queue: {device['queue']}")
       print(f"  Status: {device['status']}")

Understanding Results
=====================

Result Format
-------------

Cloud jobs return results in the following format:

.. code-block:: python

   # From task.results()
   result = {
       "result": {
           "00": 47,
           "01": 1,
           "10": 2,
           "11": 50
       },
       "result_meta": {
           "shots": 100,
           "device": "tyxonq::homebrew_s2",
           "raw": {
               "task": {
                   "id": "<JOB_ID>",
                   "device": "homebrew_s2?o=3",
                   "qubits": 2,
                   "depth": 3,
                   "state": "completed",
                   "shots": 100,
                   "result": {"00": 47, "01": 1, "10": 2, "11": 50},
                   "ts": {
                       "pending": 1754275502265270,
                       "scheduled": 1754275502260031,
                       "completed": 1754275505649825
                   },
                   "runDur": 2532053,
                   "atChip": 1754275446369691,
                   "durChip": 120185
               }
           }
       }
   }

Processing Counts
-----------------

.. code-block:: python

   from tyxonq.postprocessing import metrics
   
   result = task.results()
   counts = result["result"]
   
   # Compute expectation values
   ez = metrics.expectation(counts, z=[0, 1])
   print(f"<Z0⊗Z1> = {ez}")
   
   # Compute fidelity to ideal state
   ideal_counts = {"00": 50, "11": 50}
   fidelity = metrics.fidelity(counts, ideal_counts)
   print(f"Fidelity = {fidelity}")

Best Practices
==============

1. Start with Simulators
------------------------

**Always test on simulators first**:

.. code-block:: python

   from tyxonq.runtime.simulator import run as sim_run
   from tyxonq.cloud import apis
   
   # Debug on local simulator
   c = create_complex_circuit()
   sim_result = sim_run(c, shots=100)
   print("Simulator result:", sim_result[0]["result"])
   
   # Once validated, run on hardware
   task = apis.submit_task(circuit=c, shots=100, device="homebrew_s2")
   hw_result = task.results()
   print("Hardware result:", hw_result["result"])

2. Optimize Shot Usage
----------------------

**Use appropriate shot counts**:

- **Quick tests**: 100-500 shots
- **Standard runs**: 1000-2000 shots  
- **High precision**: 4000-8000 shots
- **Research quality**: 10000+ shots

.. code-block:: python

   # Quick validation
   task = apis.submit_task(circuit=c, shots=100, device="homebrew_s2")
   
   # Production run
   task = apis.submit_task(circuit=c, shots=4000, device="homebrew_s2")

3. Handle Errors Gracefully
---------------------------

.. code-block:: python

   import tyxonq as tq
   from tyxonq.cloud import apis
   
   try:
       task = apis.submit_task(circuit=c, shots=100, device="homebrew_s2")
       result = task.get_result(wait=True, timeout=60)
       counts = result["result"]
   except ConnectionError:
       print("Network error - check connection")
   except TimeoutError:
       print("Job timeout - try reducing circuit depth")
   except RuntimeError as e:
       print(f"Execution error: {e}")
   except Exception as e:
       print(f"Unexpected error: {e}")

4. Monitor Queue Status
-----------------------

.. code-block:: python

   from tyxonq.devices.hardware.tyxonq.driver import list_properties
   
   # Check queue before submission
   props = list_properties(device="homebrew_s2", token=YOUR_API_KEY)
   queue_length = props[0].get("queue", 0)
   
   if queue_length > 10:
       print(f"Warning: {queue_length} jobs in queue")
   
   # Submit job
   task = apis.submit_task(circuit=c, shots=100, device="homebrew_s2")

5. Use Hardware Optimization
----------------------------

TyxonQ supports **device-specific optimization flags** via query parameters:

**Optimization Flags** (additive combination):

- ``o=1``: **Qubit mapping** - Map logical to physical qubits
- ``o=2``: **Gate decomposition** - Decompose to native gates  
- ``o=4``: **Initial mapping** - Optimize initial qubit layout
- ``o=7``: **All optimizations** (1+2+4) - Enable all features

.. code-block:: python

   # Basic qubit mapping only
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
   
   # Alternatively, use kwargs
   task = apis.submit_task(
       circuit=c,
       shots=100,
       device="homebrew_s2",
       enable_qos_qubit_mapping=True,       # o=1
       enable_qos_gate_decomposition=True,  # o=2
       enable_qos_initial_mapping=True      # o=4
   )

Troubleshooting
===============

Common Issues
-------------

**Authentication Failed**

.. code-block:: python

   # Error: "Invalid API key" or authentication error
   # Solution: Verify your token format (should be "username;priority")
   from tyxonq.cloud import apis
   apis.set_token("username;0")
   apis.set_provider("tyxonq")
   
   # Test connection
   devices = apis.list_devices()
   print(devices)  # Should list devices if token is valid

**Device Not Available**

.. code-block:: python

   # Error: Device 'homebrew_s2' not accessible
   # Solution: Check access permissions
   devices = apis.list_devices()
   
   if "tyxonq::homebrew_s2" not in devices:
       print("Hardware access not yet approved")
       print("Visit https://www.tyxonq.com/ to request access")

**Job Timeout**

.. code-block:: python

   # Increase timeout for complex circuits
   result = task.get_result(
       wait=True,
       timeout=300  # 5 minutes
   )

**Circuit Compilation Error**

.. code-block:: python

   # Error: Submit Task Failed
   # Solution: Check device properties and optimize circuit
   from tyxonq.devices.hardware.tyxonq.driver import list_properties
   
   props = list_properties(device="homebrew_s2", token=YOUR_API_KEY)
   print(f"Max qubits: {props[0]['qubits']}")
   print(f"Device status: {props[0]['status']}")
   
   # Reduce circuit complexity or wait for device to be online

Next Steps
==========

- 📖 Read :doc:`device_management` for advanced device configuration and optimization flags
- 📖 Read :doc:`task_submission` for detailed task management and monitoring
- 📖 Read :doc:`api_reference` for complete API documentation
- 📖 Read :doc:`hardware_access` for Homebrew_S2 specifications and **pulse-level programming**

**Ready for Pulse-Level Control?** 

Check out pulse programming examples:

.. code-block:: python

   from tyxonq import Circuit, Param, waveforms
   from tyxonq.cloud import apis
   
   # Rabi oscillation experiment
   qc = Circuit(1)
   qc.use_pulse()
   
   param = Param("q[0]")
   builder = qc.calibrate("rabi", [param])
   builder.new_frame("drive_frame", param)
   
   # Sweep pulse duration
   for duration in range(10, 100, 10):
       builder.play("drive_frame", waveforms.CosineDrag(duration, 0.2, 0.0, 0.0))
   
   builder.build()
   qc.add_calibration('rabi', ['q[0]'])
   qc.measure_z(0)
   
   # Submit to hardware
   task = apis.submit_task(
       circuit=qc,
       shots=100,
       device="homebrew_s2",
       enable_qos_gate_decomposition=False
   )
   
   print(f"TQASM 0.2 code:\n{qc.to_tqasm()}")
   print(f"Results: {task.results()}")
