Hardware Access
===============

===============
Hardware Access
===============

Complete guide to accessing **Homebrew_S2** quantum processor, including gate-level and **pulse-level programming**.

Homebrew_S2 Overview
====================

**Homebrew_S2** is TyxonQ's **13-qubit superconducting quantum processor**, designed and operated by QureGenAI for NISQ-era research and applications.

**Key Features**:

- **13 superconducting transmon qubits**
- **Two programming models**:
  
  - **Gate-level**: Standard quantum circuit model (OpenQASM 2.0)
  - **Pulse-level**: Direct microwave pulse control (TQASM 0.2)

- **Cloud API**: REST API with Bearer token authentication
- **Optimization flags**: Qubit mapping, gate decomposition, initial mapping

API Endpoint
------------

**Base URL**: ``https://api.tyxonq.com/qau-cloud/tyxonq/``

**API Version**: v1

**Full endpoint format**:

.. code-block:: text

   https://api.tyxonq.com/qau-cloud/tyxonq/api/v1/{command}

**Authentication**: Bearer token (format: ``username;priority``)

Device Properties
-----------------

Query real-time device specifications:

.. code-block:: python

   from tyxonq.devices.hardware.tyxonq.driver import list_properties
   
   props = list_properties(device="homebrew_s2", token=YOUR_API_KEY)
   
   for device in props:
       print(f"Device ID: {device['id']}")
       print(f"Name: {device['name']}")
       print(f"Qubits: {device['qubits']}")
       print(f"Queue: {device['queue']}")
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

Gate-Level Programming
======================

Standard Quantum Circuit Model
------------------------------

Gate-level programming uses the standard quantum circuit abstraction:

.. code-block:: python

   import tyxonq as tq
   from tyxonq.cloud import apis
   import getpass
   
   # Configure authentication
   API_KEY = getpass.getpass("Input your TyxonQ API_KEY: ")
   apis.set_token(API_KEY)
   apis.set_provider("tyxonq")
   
   # Create quantum circuit
   c = tq.Circuit(2)
   c.h(0)
   c.cx(0, 1)
   c.measure_z(0)
   c.measure_z(1)
   
   # Submit to Homebrew_S2 with optimization
   task = apis.submit_task(
       circuit=c,
       shots=100,
       device="homebrew_s2?o=7",  # Full optimization (1+2+4)
       enable_qos_qubit_mapping=True,
       enable_qos_gate_decomposition=True,
       enable_qos_initial_mapping=True
   )
   
   # Get results
   result = task.results()
   counts = result["result"]
   print("Measurement counts:", counts)

Hardware Optimization Flags
----------------------------

Homebrew_S2 supports **additive optimization flags**:

**Available Flags**:

- ``o=1``: **Qubit mapping** - Map logical to physical qubits
- ``o=2``: **Gate decomposition** - Decompose to native gates
- ``o=4``: **Initial mapping** - Optimize initial qubit layout
- ``o=7``: **All optimizations** (1+2+4)

**Examples**:

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

**Using kwargs** (alternative):

.. code-block:: python

   task = apis.submit_task(
       circuit=c,
       shots=100,
       device="homebrew_s2",
       enable_qos_qubit_mapping=True,       # o=1
       enable_qos_gate_decomposition=True,  # o=2
       enable_qos_initial_mapping=True      # o=4
   )
   # Equivalent to device="homebrew_s2?o=7"

OpenQASM 2.0 Submission
-----------------------

Submit pre-compiled OpenQASM code:

.. code-block:: python

   from tyxonq.cloud import apis
   
   qasm_code = """
   OPENQASM 2.0;
   include "qelib1.inc";
   qreg q[2];
   creg c[2];
   h q[0];
   cx q[0],q[1];
   measure q[0] -> c[0];
   measure q[1] -> c[1];
   """
   
   from tyxonq.devices.hardware.tyxonq.driver import submit_task
   
   tasks = submit_task(
       device="tyxonq::homebrew_s2?o=7",
       source=qasm_code,
       shots=100,
       lang="OPENQASM",
       token=YOUR_API_KEY
   )
   
   task = tasks[0]
   result = task.get_result(wait=True, timeout=60)
   print("Result:", result["result"])

Result Format
-------------

Gate-level jobs return measurement counts:

.. code-block:: python

   {
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
                   "device": "homebrew_s2?o=7",
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

Pulse-Level Programming
=======================

Overview
--------

**Pulse-level control** allows direct programming of microwave pulses sent to qubits, enabling:

- **Custom gate implementations**
- **Pulse shape optimization**
- **Quantum control experiments** (Rabi, Ramsey, DRAG)
- **Calibration and characterization**

TyxonQ implements pulse-level control through **TQASM 0.2** format, a domain-specific language for pulse programming.

TQASM 0.2 Format
----------------

**Key Concepts**:

- ``defcal``: Define calibration program (custom gate)
- ``frame``: Microwave control channel
- ``newframe``: Create frame on target qubit
- ``play``: Execute waveform on frame
- **Waveforms**: Parametric pulse shapes

**BNF Grammar**:

.. code-block:: bnf

   <pulse> ::= <defcal>
   
   <defcal> ::= "defcal" <id> <idlist> { <calgrammar> }
   
   <calgrammar> ::= <calstatement>
                  | <calgrammar> <calstatement>
   
   <calstatement> ::= <framedecl>
                   | <waveformplay>
   
   <framedecl> ::= "frame" <id> "=" "newframe" (<idlist>);
   
   <waveformplay> ::= "play" (<id>, <waveform>);
   
   <waveform> ::= <id> (<explist>)

**Example TQASM 0.2 Code**:

.. code-block:: text

   TQASM 0.2;
   QREG q[1];
   
   defcal rabi_test q[0] {
       frame drive_frame = newframe(q[0]);
       play(drive_frame, cosine_drag(50, 0.2, 0.0, 0.0));
   }
   
   rabi_test q[0];
   MEASZ q[0];

Supported Waveforms
-------------------

Homebrew_S2 supports the following parametric waveforms:

**1. CosineDrag**

.. code-block:: python

   waveforms.CosineDrag(duration, amp, sigma, beta)

- ``duration``: Pulse length (int, 0 < duration < 10000)
- ``amp``: Amplitude (real, |amp| ≤ 2)
- ``sigma``: Width parameter (real)
- ``beta``: DRAG coefficient (real)

**Mathematical Form**:

.. math::

   f(x) = A \cdot \left[\cos\left(\frac{\pi x}{T}\right) + \beta \frac{dC}{dx}\right]

where :math:`C(x) = \cos(\pi x / T)`, :math:`0 \le x < T`

**2. Gaussian**

.. code-block:: python

   waveforms.Gaussian(duration, amp, sigma)

- ``duration``: Pulse length
- ``amp``: Amplitude
- ``sigma``: Standard deviation

**Mathematical Form**:

.. math::

   f(x) = A \cdot \exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right), \quad 0 \le x < T

**3. Sine**

.. code-block:: python

   waveforms.Sine(duration, amp, phase, freq, angle)

- ``duration``: Pulse length
- ``amp``: Amplitude (|amp| ≤ 2)
- ``phase``: Phase in radians
- ``freq``: Frequency (reciprocal of sampling period)
- ``angle``: Complex phase factor in radians

**Mathematical Form**:

.. math::

   f(x) = A \sin(2\pi \cdot \text{freq} \cdot x + \text{phase}), \quad 0 \le x < T

   A = \text{amp} \cdot e^{i \cdot \text{angle}}

**4. Flattop**

.. code-block:: python

   waveforms.Flattop(duration, amp, sigma)

- Flat-top Gaussian envelope
- ``duration``: Total pulse length
- ``amp``: Amplitude
- ``sigma``: Rise/fall width

**5. Constant**

.. code-block:: python

   waveforms.Constant(duration, amp)

- Constant amplitude pulse
- ``duration``: Pulse length
- ``amp``: Amplitude

**6. GaussianSquare**

.. code-block:: python

   waveforms.GaussianSquare(duration, amp, sigma, width)

- Gaussian edges with flat middle
- ``duration``: Total duration
- ``amp``: Amplitude
- ``sigma``: Edge width
- ``width``: Flat region width

**7. Cosine**

.. code-block:: python

   waveforms.Cosine(duration, amp, freq, phase)

- Pure cosine wave
- ``duration``: Pulse length
- ``amp``: Amplitude
- ``freq``: Frequency
- ``phase``: Phase offset

**8. Drag**

.. code-block:: python

   waveforms.Drag(duration, amp, sigma, beta)

- DRAG pulse (derivative removal by adiabatic gate)
- ``duration``: Pulse length
- ``amp``: Amplitude
- ``sigma``: Width
- ``beta``: DRAG coefficient

DefcalBuilder Workflow
----------------------

TyxonQ provides ``DefcalBuilder`` for constructing pulse programs:

**Step 1**: Create Circuit in Pulse Mode

.. code-block:: python

   import tyxonq as tq
   from tyxonq import Param, waveforms
   
   qc = tq.Circuit(1)
   qc.use_pulse()  # Enable pulse mode

**Step 2**: Define Calibration

.. code-block:: python

   # Create parameter for target qubit
   param = Param("q[0]")
   
   # Start calibration builder
   builder = qc.calibrate("my_gate", [param])

**Step 3**: Add Frames and Waveforms

.. code-block:: python

   # Create microwave frame
   builder.new_frame("drive_frame", param)
   
   # Play waveform on frame
   builder.play("drive_frame", waveforms.CosineDrag(50, 0.2, 0.0, 0.0))

**Step 4**: Build and Apply

.. code-block:: python

   # Finalize calibration
   builder.build()
   
   # Add calibration call to circuit
   qc.add_calibration('my_gate', ['q[0]'])

**Step 5**: Generate TQASM

.. code-block:: python

   tqasm_code = qc.to_tqasm()
   print(tqasm_code)

Pulse Example: Rabi Oscillation
--------------------------------

Complete example of pulse-level Rabi experiment:

.. code-block:: python

   import tyxonq as tq
   from tyxonq import Param, waveforms
   from tyxonq.cloud import apis
   import getpass
   
   # Configure API
   API_KEY = getpass.getpass("Input your TyxonQ API_KEY: ")
   apis.set_token(API_KEY)
   apis.set_provider("tyxonq")
   
   # Create pulse circuit
   qc = tq.Circuit(1)
   qc.use_pulse()
   
   # Define parametric calibration
   param = Param("q[0]")
   builder = qc.calibrate("rabi_test", [param])
   builder.new_frame("drive_frame", param)
   builder.play("drive_frame", waveforms.CosineDrag(50, 0.2, 0.0, 0.0))
   builder.build()
   
   # Add calibration call
   qc.add_calibration('rabi_test', ['q[0]'])
   qc.measure_z(0)
   
   # View generated TQASM 0.2
   print("=" * 60)
   print("Generated TQASM 0.2 Code:")
   print("=" * 60)
   print(qc.to_tqasm())
   print("=" * 60)
   
   # Submit to hardware (NO optimization for pulse-level)
   task = apis.submit_task(
       circuit=qc,
       shots=100,
       device="homebrew_s2",
       enable_qos_gate_decomposition=False,  # Keep pulse-level
       enable_qos_qubit_mapping=False        # No mapping
   )
   
   # Get results
   result = task.results()
   print("Rabi oscillation result:", result["result"])

**Generated TQASM**:

.. code-block:: text

   TQASM 0.2;
   QREG q[1];
   
   defcal rabi_test q[0] {
       frame drive_frame = newframe(q[0]);
       play(drive_frame, cosine_drag(50, 0.2, 0.0, 0.0));
   }
   
   rabi_test q[0];
   MEASZ q[0];

Pulse Example: Parameter Scan
------------------------------

Scan pulse duration for Rabi oscillations:

.. code-block:: python

   import tyxonq as tq
   from tyxonq import Param, waveforms
   from tyxonq.cloud import apis
   import time
   
   def create_rabi_circuit(duration):
       """Create Rabi circuit with specific pulse duration."""
       qc = tq.Circuit(1)
       qc.use_pulse()
       
       param = Param("q[0]")
       builder = qc.calibrate("rabi", [param])
       builder.new_frame("drive_frame", param)
       builder.play("drive_frame", waveforms.CosineDrag(duration, 0.2, 0.0, 0.0))
       builder.build()
       
       qc.add_calibration('rabi', ['q[0]'])
       qc.measure_z(0)
       
       return qc
   
   # Sweep pulse duration
   results = []
   for duration in range(10, 100, 10):
       qc = create_rabi_circuit(duration)
       
       task = apis.submit_task(
           circuit=qc,
           shots=100,
           device="homebrew_s2",
           enable_qos_gate_decomposition=False,
           enable_qos_qubit_mapping=False
       )
       
       result = task.results()
       results.append((duration, result["result"]))
       
       print(f"Duration {duration}: {result['result']}")
       time.sleep(30)  # Wait between submissions
   
   # Analyze oscillation
   import matplotlib.pyplot as plt
   
   durations = [r[0] for r in results]
   excited_probs = [r[1].get("1", 0) / 100.0 for r in results]
   
   plt.plot(durations, excited_probs, 'o-')
   plt.xlabel("Pulse Duration")
   plt.ylabel("P(|1⟩)")
   plt.title("Rabi Oscillation")
   plt.show()

Advanced Pulse Features
-----------------------

**Multiple Frames**:

.. code-block:: python

   qc = tq.Circuit(2)
   qc.use_pulse()
   
   param0 = Param("q[0]")
   param1 = Param("q[1]")
   
   builder = qc.calibrate("two_qubit_gate", [param0, param1])
   builder.new_frame("drive_frame_0", param0)
   builder.new_frame("drive_frame_1", param1)
   builder.play("drive_frame_0", waveforms.Gaussian(100, 0.3, 10))
   builder.play("drive_frame_1", waveforms.Gaussian(100, 0.3, 10))
   builder.build()
   
   qc.add_calibration('two_qubit_gate', ['q[0]', 'q[1]'])

**Complex Waveform Sequences**:

.. code-block:: python

   builder = qc.calibrate("complex_gate", [param])
   builder.new_frame("drive_frame", param)
   
   # Sequence of pulses
   builder.play("drive_frame", waveforms.Gaussian(50, 0.1, 5))
   builder.play("drive_frame", waveforms.Constant(20, 0.2))
   builder.play("drive_frame", waveforms.Gaussian(50, 0.1, 5))
   
   builder.build()

Best Practices
==============

Gate-Level Best Practices
--------------------------

**1. Use Hardware Optimization**:

.. code-block:: python

   # Always use optimization for gate-level
   task = apis.submit_task(
       circuit=c,
       shots=100,
       device="homebrew_s2?o=7"  # Full optimization
   )

**2. Respect Qubit Connectivity**:

.. code-block:: python

   # Check coupling map before designing circuits
   from tyxonq.devices.hardware.tyxonq.driver import list_properties
   
   props = list_properties(device="homebrew_s2", token=YOUR_API_KEY)
   print(f"Available qubits: {props[0]['qubits']}")

**3. Validate Shot Count**:

.. code-block:: python

   # Use appropriate shots for precision
   # Quick test: 100-500
   # Standard: 1000-4000
   # High precision: 10000+
   
   task = apis.submit_task(circuit=c, shots=4000, device="homebrew_s2?o=7")

Pulse-Level Best Practices
---------------------------

**1. Disable Optimization**:

.. code-block:: python

   # CRITICAL: Disable gate decomposition for pulse-level
   task = apis.submit_task(
       circuit=pulse_circuit,
       shots=100,
       device="homebrew_s2",
       enable_qos_gate_decomposition=False,
       enable_qos_qubit_mapping=False
   )

**2. Validate Waveform Parameters**:

.. code-block:: python

   # Check parameter constraints:
   # - duration: 0 < duration < 10000
   # - amp: |amp| ≤ 2
   # - All parameters must be numeric
   
   waveform = waveforms.CosineDrag(
       duration=50,   # Valid
       amp=0.2,       # Valid (|0.2| ≤ 2)
       sigma=0.0,
       beta=0.0
   )

**3. Start with Simple Waveforms**:

.. code-block:: python

   # Begin with basic waveforms
   simple_pulse = waveforms.Constant(50, 0.1)
   
   # Progress to complex shapes
   drag_pulse = waveforms.Drag(100, 0.3, 10, 0.5)

**4. Use Appropriate Durations**:

.. code-block:: python

   # Typical durations:
   # - Single-qubit gates: 20-100 ns
   # - Two-qubit gates: 200-500 ns
   # - Calibration: 10-1000 ns
   
   rabi_waveform = waveforms.CosineDrag(50, 0.2, 0.0, 0.0)  # 50 ns

Troubleshooting
===============

Gate-Level Issues
-----------------

**Circuit Too Deep**:

.. code-block:: python

   # Error: Circuit depth exceeds limits
   # Solution: Use full optimization
   task = apis.submit_task(
       circuit=c,
       shots=100,
       device="homebrew_s2?o=7"  # Reduces depth
   )

**Gate Not Supported**:

.. code-block:: python

   # Error: Gate not in native set
   # Solution: Let optimization decompose automatically
   task = apis.submit_task(
       circuit=c,
       shots=100,
       device="homebrew_s2?o=2"  # Enables gate decomposition
   )

Pulse-Level Issues
------------------

**Waveform Parameter Error**:

.. code-block:: python

   # Error: Invalid parameter
   # Solution: Check constraints
   
   # BAD: duration out of range
   # waveform = waveforms.Gaussian(20000, 0.1, 5)  # duration > 10000
   
   # GOOD: valid duration
   waveform = waveforms.Gaussian(100, 0.1, 5)  # 0 < 100 < 10000

**TQASM Syntax Error**:

.. code-block:: python

   # Always validate generated TQASM
   tqasm_code = qc.to_tqasm()
   print(tqasm_code)  # Check for syntax
   
   # Common issues:
   # - Missing semicolons
   # - Incorrect parameter types
   # - Invalid frame names

**Optimization Conflict**:

.. code-block:: python

   # Error: Pulse code optimized away
   # Solution: DISABLE optimization
   task = apis.submit_task(
       circuit=pulse_circuit,
       shots=100,
       device="homebrew_s2",  # No ?o= parameter
       enable_qos_gate_decomposition=False,
       enable_qos_qubit_mapping=False
   )

Next Steps
==========

- 📖 Read :doc:`getting_started` for basic cloud access
- 📖 Read :doc:`device_management` for optimization flags reference
- 📖 Read :doc:`task_submission` for advanced job management
- 📖 Read :doc:`api_reference` for complete API documentation
- 📖 **Pulse documentation**: ``docs/pulse_support_en.md`` (detailed TQASM 0.2 specification)

**External Resources**:

- **TyxonQ Cloud Portal**: https://www.tyxonq.com/
- **API Documentation**: ``docs/tyxonq_cloud_api.md``
- **Pulse Specification**: ``docs/pulse_support_en.md``
