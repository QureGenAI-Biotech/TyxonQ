Task Submission
===============

===============
Task Submission
===============

TyxonQ provides flexible task submission APIs for running quantum circuits on cloud hardware through the TyxonQ Cloud API.

Task Submission Methods
=======================

Method 1: Cloud API (Recommended)
----------------------------------

The cloud API provides the most direct workflow:

.. code-block:: python

   import tyxonq as tq
   from tyxonq.cloud import apis
   import getpass
   
   # Configure API
   API_KEY = getpass.getpass("Input your TyxonQ API_KEY: ")
   apis.set_token(API_KEY)
   apis.set_provider("tyxonq")
   
   # Build circuit
   c = tq.Circuit(2)
   c.h(0)
   c.cx(0, 1)
   c.measure_all()
   
   # Submit task
   task = apis.submit_task(
       circuit=c,
       shots=100,
       device="homebrew_s2?o=7"
   )
   
   # Get results
   result = task.results()
   counts = result["result"]
   print("Counts:", counts)

**Advantages**:

- ✅ Direct cloud integration
- ✅ Real TyxonQTask objects
- ✅ Supports both gate-level and pulse-level
- ✅ Hardware optimization flags

Method 2: Low-Level Driver API
------------------------------

Use the hardware driver directly for advanced control:

.. code-block:: python

   from tyxonq.devices.hardware.tyxonq.driver import submit_task, get_task_details
   
   # Submit with OpenQASM source
   qasm_source = """
   OPENQASM 2.0;
   include "qelib1.inc";
   qreg q[2];
   creg c[2];
   h q[0];
   cx q[0],q[1];
   measure q[0] -> c[0];
   measure q[1] -> c[1];
   """
   
   tasks = submit_task(
       device="tyxonq::homebrew_s2?o=7",
       source=qasm_source,
       shots=100,
       lang="OPENQASM",
       token=YOUR_API_KEY
   )
   
   # Get results
   task = tasks[0]
   result = get_task_details(task, token=YOUR_API_KEY)
   counts = result["result"]
   print("Counts:", counts)

**Use Cases**:

- Submitting pre-compiled QASM from external tools
- Bypassing TyxonQ circuit compilation
- Direct API integration
- Batch submission with custom control

Method 3: Pulse-Level Submission
--------------------------------

Submit pulse-level circuits with TQASM 0.2:

.. code-block:: python

   import tyxonq as tq
   from tyxonq import Param, waveforms
   from tyxonq.cloud import apis
   
   # Create pulse circuit
   qc = tq.Circuit(1)
   qc.use_pulse()
   
   param = Param("q[0]")
   builder = qc.calibrate("rabi_test", [param])
   builder.new_frame("drive_frame", param)
   builder.play("drive_frame", waveforms.CosineDrag(50, 0.2, 0.0, 0.0))
   builder.build()
   
   qc.add_calibration('rabi_test', ['q[0]'])
   qc.measure_z(0)
   
   # Submit pulse circuit (NO optimization)
   task = apis.submit_task(
       circuit=qc,
       shots=100,
       device="homebrew_s2",
       enable_qos_gate_decomposition=False,  # Keep pulse-level
       enable_qos_qubit_mapping=False
   )
   
   result = task.results()
   print("Pulse result:", result["result"])

**Key Points for Pulse-Level**:

- ⚠️ **CRITICAL**: Disable all optimization flags
- Use `qc.use_pulse()` to enable pulse mode
- Generated TQASM 0.2 code is submitted directly

Task Parameters
===============

Required Parameters
-------------------

.. code-block:: python

   from tyxonq.cloud import apis
   
   task = apis.submit_task(
       circuit=c,               # Required: Circuit object
       shots=100,               # Required: Number of shots
       device="homebrew_s2"      # Required: Device name
   )

Optional Parameters
-------------------

.. code-block:: python

   task = apis.submit_task(
       circuit=c,
       shots=100,
       device="homebrew_s2",
       
       # Hardware optimization flags
       enable_qos_qubit_mapping=True,       # o=1
       enable_qos_gate_decomposition=True,  # o=2 
       enable_qos_initial_mapping=True      # o=4
   )
   
   # Alternative: Use device string
   task = apis.submit_task(
       circuit=c,
       shots=100,
       device="homebrew_s2?o=7"  # All optimizations (1+2+4)
   )

Batch Submission
----------------

Submit multiple circuits at once:

.. code-block:: python

   from tyxonq.cloud import apis
   
   # Create multiple circuits
   circuits = [
       tq.Circuit(2).h(0).cx(0,1).measure_all(),
       tq.Circuit(2).x(0).cx(0,1).measure_all(),
       tq.Circuit(2).y(0).cx(0,1).measure_all()
   ]
   
   # Submit all circuits
   tasks = []
   for i, circuit in enumerate(circuits):
       task = apis.submit_task(
           circuit=circuit,
           shots=100,
           device="homebrew_s2?o=7"
       )
       tasks.append(task)
       print(f"Submitted circuit {i}: task ID {task.id}")
   
   # Collect results
   results = []
   for i, task in enumerate(tasks):
       result = task.results()
       results.append(result)
       print(f"Circuit {i} result: {result['result']}")

Task Monitoring
===============

TyxonQTask Objects
------------------

The `apis.submit_task()` function returns `TyxonQTask` objects:

.. code-block:: python

   from tyxonq.cloud import apis
   
   task = apis.submit_task(circuit=c, shots=100, device="homebrew_s2")
   
   # Task properties
   print(f"Task ID: {task.id}")
   print(f"Device: {task.device}")
   print(f"Status: {task.status}")
   print(f"Async result: {task.async_result}")

Synchronous Execution
---------------------

Wait for task completion:

.. code-block:: python

   from tyxonq.cloud import apis
   
   # Submit and get results immediately
   task = apis.submit_task(circuit=c, shots=100, device="homebrew_s2")
   result = task.results()  # Blocks until complete
   counts = result["result"]
   print("Counts:", counts)

Asynchronous Execution
----------------------

Submit and check status later:

.. code-block:: python

   from tyxonq.cloud import apis
   import time
   
   # Submit without waiting
   task = apis.submit_task(circuit=c, shots=100, device="homebrew_s2")
   print(f"Task submitted: {task.id}")
   print(f"Initial status: {task.status}")
   
   # Do other work
   perform_other_computations()
   
   # Check status later (non-blocking)
   status_info = task.get_result(wait=False)
   task_state = status_info.get('result_meta', {}).get('raw', {}).get('task', {}).get('state', 'unknown')
   print(f"Current state: {task_state}")
   
   # Wait for completion when ready
   if task_state != "completed":
       final_result = task.get_result(wait=True, timeout=120)
       counts = final_result["result"]
       print("Final results:", counts)

Custom Polling Strategy
-----------------------

Implement custom polling logic:

.. code-block:: python

   from tyxonq.cloud import apis
   import time
   
   def poll_task_with_backoff(task, max_attempts=30, base_interval=2.0):
       """Poll task status with exponential backoff."""
       for attempt in range(max_attempts):
           details = task.get_result(wait=False)
           task_info = details.get('result_meta', {}).get('raw', {}).get('task', {})
           state = task_info.get('state', 'unknown')
           
           print(f"Attempt {attempt+1}: State = {state}")
           
           if state == "completed":
               return details
           elif state == "failed":
               error = task_info.get('error', 'Unknown error')
               raise RuntimeError(f"Task failed: {error}")
           
           # Exponential backoff
           wait_time = base_interval * (1.5 ** min(attempt, 5))
           time.sleep(wait_time)
       
       raise TimeoutError(f"Task {task.id} did not complete")
   
   # Use custom polling
   task = apis.submit_task(circuit=c, shots=100, device="homebrew_s2")
   result = poll_task_with_backoff(task)
   print("Final result:", result["result"])

Task Status
===========

Status Values
-------------

Tasks progress through the following states:

.. code-block:: python

   "submitted"   # Task accepted by API
   "processing"  # Processing in the cloud
   "running"     # Executing on hardware
   "completed"   # Successfully finished ("done", "success", "finished" also valid)
   "failed"      # Execution failed

Checking Status
---------------

.. code-block:: python

   from tyxonq.cloud import apis
   
   task = apis.submit_task(circuit=c, shots=100, device="homebrew_s2")
   
   # Get task status
   details = task.get_result(wait=False)
   task_info = details.get('result_meta', {}).get('raw', {}).get('task', {})
   state = task_info.get('state', 'unknown')
   
   if state in ["completed", "done", "success", "finished"]:
       counts = details["result"]
       print("Success:", counts)
   elif state == "failed":
       error = task_info.get("error", "Unknown error")
       print(f"Failed: {error}")
   elif state in ["submitted", "processing", "running"]:
       print(f"In progress: {state}")

Task Details
============

Retrieving Task Information
---------------------------

.. code-block:: python

   from tyxonq.cloud import apis
   
   task = apis.submit_task(circuit=c, shots=100, device="homebrew_s2")
   details = task.get_result(wait=False)
   
   # Task metadata
   print(f"Task ID: {task.id}")
   print(f"Device: {task.device}")
   print(f"Status: {task.status}")
   
   # Detailed information from API response
   raw_task = details.get('result_meta', {}).get('raw', {}).get('task', {})
   print(f"Queue: {raw_task.get('queue')}")
   print(f"Qubits: {raw_task.get('qubits')}")
   print(f"Depth: {raw_task.get('depth')}")
   print(f"Shots: {raw_task.get('shots')}")
   
   # Timing information
   timestamps = raw_task.get('ts', {})
   print(f"Pending: {timestamps.get('pending')}")
   print(f"Scheduled: {timestamps.get('scheduled')}")
   print(f"Completed: {timestamps.get('completed')}")

**Response Example**:

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
                   "queue": "quregenai.lab",
                   "device": "homebrew_s2?o=3",
                   "qubits": 2,
                   "depth": 3,
                   "state": "completed",
                   "shots": 100,
                   "result": {"00": 47, "01": 1, "10": 2, "11": 50},
                   "ts": {
                       "completed": 1754275505649825,
                       "pending": 1754275502265270,
                       "scheduled": 1754275502260031
                   },
                   "runDur": 2532053,
                   "atChip": 1754275446369691,
                   "durChip": 120185,
                   "task_type": "quantum_api"
               }
           }
       }
   }

Extracting Results
------------------

.. code-block:: python

   from tyxonq.cloud import apis
   
   task = apis.submit_task(circuit=c, shots=100, device="homebrew_s2")
   result = task.results()
   
   # Get measurement counts
   counts = result["result"]
   
   # Get execution metadata
   metadata = result.get("result_meta", {})
   raw_task = metadata.get("raw", {}).get("task", {})
   
   # Timing information
   run_duration = raw_task.get("runDur", 0)  # microseconds
   chip_time = raw_task.get("durChip", 0)    # microseconds
   
   print(f"Total runtime: {run_duration / 1000:.1f} ms")
   print(f"On-chip time: {chip_time / 1000:.1f} ms")
   print(f"Queue+overhead: {(run_duration - chip_time) / 1000:.1f} ms")

Task Management
===============

Error Handling
--------------

.. code-block:: python

   from tyxonq.cloud import apis
   import time
   
   def robust_submit(circuit, shots=100, max_retries=3):
       """Submit task with automatic retry on failure."""
       for attempt in range(max_retries):
           try:
               task = apis.submit_task(
                   circuit=circuit,
                   shots=shots,
                   device="homebrew_s2?o=7"
               )
               
               # Wait for completion with timeout
               result = task.get_result(wait=True, timeout=300)
               
               # Check if successful
               task_info = result.get('result_meta', {}).get('raw', {}).get('task', {})
               state = task_info.get('state', 'unknown')
               
               if state in ["completed", "done", "success"]:
                   return result
               else:
                   error = task_info.get('error', f'Task state: {state}')
                   print(f"Attempt {attempt+1} failed: {error}")
                   
           except Exception as e:
               print(f"Attempt {attempt+1} error: {e}")
           
           if attempt < max_retries - 1:
               print(f"Retrying... ({attempt+2}/{max_retries})")
               time.sleep(5)  # Wait before retry
       
       raise RuntimeError("All retry attempts failed")
   
   # Use with retry
   result = robust_submit(c, shots=100)
   print("Success:", result["result"])

Error Handling
==============

Common Errors
-------------

**Submission Errors**:

.. code-block:: python

   from tyxonq.cloud import apis
   
   try:
       task = apis.submit_task(
           circuit=c,
           shots=100,
           device="homebrew_s2?o=7"
       )
   except RuntimeError as e:
       if "Submit Task Failed" in str(e):
           print("Submission failed - check device status and circuit")
           # Check device properties
           from tyxonq.devices.hardware.tyxonq.driver import list_properties
           props = list_properties(device="homebrew_s2", token=YOUR_API_KEY)
           print(f"Device status: {props[0]['status']}")
       else:
           print(f"Other error: {e}")
   except ConnectionError as e:
       print(f"Network error: {e}")
   except Exception as e:
       print(f"Unexpected error: {e}")

**Execution Errors**:

.. code-block:: python

   task = apis.submit_task(circuit=c, shots=100, device="homebrew_s2")
   result = task.get_result(wait=True, timeout=120)
   
   task_info = result.get('result_meta', {}).get('raw', {}).get('task', {})
   state = task_info.get('state')
   
   if state == "failed":
       error = task_info.get("error", "Unknown error")
       print(f"Execution failed: {error}")
       
       # Common failure reasons:
       if "qubits" in error.lower():
           print("Solution: Reduce circuit size or use optimization")
       elif "depth" in error.lower():
           print("Solution: Use hardware optimization (o=7)")
       elif "timeout" in error.lower():
           print("Solution: Simplify circuit or increase timeout")

**Timeout Errors**:

.. code-block:: python

   from tyxonq.cloud import apis
   
   try:
       task = apis.submit_task(circuit=c, shots=100, device="homebrew_s2")
       result = task.get_result(wait=True, timeout=60)
   except TimeoutError:
       print("Task did not complete within 60 seconds")
       
       # Check current status
       details = task.get_result(wait=False)
       task_info = details.get('result_meta', {}).get('raw', {}).get('task', {})
       state = task_info.get('state', 'unknown')
       print(f"Current state: {state}")
       
       # Decide whether to continue waiting
       if state in ["running", "processing"]:
           print("Task still running, waiting longer...")
           result = task.get_result(wait=True, timeout=300)

Error Recovery
--------------

.. code-block:: python

   from tyxonq.cloud import apis
   from tyxonq.devices.hardware.tyxonq.driver import list_properties
   import time
   
   def robust_submit(circuit, shots=100):
       """Submit with comprehensive error handling."""
       try:
           # Check device availability first
           props = list_properties(device="homebrew_s2", token=YOUR_API_KEY)
           
           if not props or props[0].get("status") != "online":
               raise RuntimeError("Device offline, using simulator")
           
           # Submit to hardware
           task = apis.submit_task(
               circuit=circuit,
               shots=shots,
               device="homebrew_s2?o=7"
           )
           
           result = task.get_result(wait=True, timeout=300)
           return result, "hardware"
           
       except (ConnectionError, TimeoutError, RuntimeError) as e:
           print(f"Hardware error: {e}")
           print("Falling back to local simulator")
           
           # Fallback to local simulator
           from tyxonq.runtime.simulator import run
           result = run(circuit, shots=shots)
           
           # Format to match cloud API response
           formatted_result = {
               "result": result[0]["result"],
               "result_meta": {
                   "shots": shots,
                   "device": "simulator",
                   "raw": {"fallback": True}
               }
           }
           
           return formatted_result, "simulator"
   
   # Use with automatic fallback
   result, backend = robust_submit(c, shots=100)
   print(f"Executed on: {backend}")
   print(f"Result: {result['result']}") 

Best Practices
==============

1. Validate Before Submission
-----------------------------

.. code-block:: python

   from tyxonq.devices.hardware.tyxonq.driver import list_properties
   from tyxonq.cloud import apis
   
   def validate_circuit(circuit, token=None):
       """Validate circuit before submission."""
       props = list_properties(device="homebrew_s2", token=token)
       
       if not props:
           raise ValueError("Cannot retrieve device properties")
           
       device_info = props[0]
       
       # Check device status
       if device_info.get("status") != "online":
           raise ValueError(f"Device is {device_info.get('status')}")
       
       # Check qubit count
       max_qubits = device_info.get("qubits", 0)
       if circuit.num_qubits > max_qubits:
           raise ValueError(f"Circuit requires {circuit.num_qubits} qubits, device has {max_qubits}")
       
       return True
   
   # Validate before submission
   validate_circuit(c, token=YOUR_API_KEY)
   
   # Submit
   task = apis.submit_task(circuit=c, shots=100, device="homebrew_s2?o=7")

2. Use Appropriate Timeouts
---------------------------

.. code-block:: python

   from tyxonq.cloud import apis
   
   # Quick test: short timeout
   task = apis.submit_task(circuit=simple_circuit, shots=100, device="homebrew_s2")
   result = task.get_result(wait=True, timeout=30)
   
   # Complex circuit: longer timeout
   task = apis.submit_task(circuit=complex_circuit, shots=1000, device="homebrew_s2?o=7")
   result = task.get_result(wait=True, timeout=300)
   
   # Pulse experiment: very long timeout
   task = apis.submit_task(circuit=pulse_circuit, shots=100, device="homebrew_s2")
   result = task.get_result(wait=True, timeout=600)

3. Log Task IDs for Recovery
----------------------------

.. code-block:: python

   from tyxonq.cloud import apis
   import logging
   
   logging.basicConfig(level=logging.INFO)
   logger = logging.getLogger(__name__)
   
   task = apis.submit_task(circuit=c, shots=100, device="homebrew_s2")
   
   # Log task details
   logger.info(f"Submitted task: {task.id}")
   logger.info(f"Device: {task.device}")
   logger.info(f"Status: {task.status}")
   
   # Save to file for recovery
   with open("submitted_tasks.txt", "a") as f:
       f.write(f"{task.id}\t{task.device}\t{task.status}\n")

4. Monitor Resource Usage
-------------------------

.. code-block:: python

   from tyxonq.cloud import apis
   
   def track_usage(task):
       """Track quantum resource usage."""
       result = task.results()
       task_info = result.get('result_meta', {}).get('raw', {}).get('task', {})
       
       usage = {
           "shots_used": task_info.get("shots", 0),
           "qubits_used": task_info.get("qubits", 0),
           "circuit_depth": task_info.get("depth", 0),
           "runtime_ms": task_info.get("runDur", 0) / 1000,
           "chip_time_ms": task_info.get("durChip", 0) / 1000
       }
       
       # Log to usage tracking system
       print(f"Resource usage: {usage}")
       
       return usage
   
   # Track usage after task completion
   task = apis.submit_task(circuit=c, shots=100, device="homebrew_s2?o=7")
   result = task.results()
   usage = track_usage(task)

5. Optimize for Different Circuit Types
---------------------------------------

.. code-block:: python

   from tyxonq.cloud import apis
   
   # Gate-level circuit: Use full optimization
   gate_task = apis.submit_task(
       circuit=gate_circuit,
       shots=1000,
       device="homebrew_s2?o=7"  # All optimizations
   )
   
   # Pulse-level circuit: Disable optimization
   pulse_task = apis.submit_task(
       circuit=pulse_circuit,
       shots=100,
       device="homebrew_s2",  # No optimization flags
       enable_qos_gate_decomposition=False,
       enable_qos_qubit_mapping=False
   )
   
   # Small test circuit: Basic optimization only
   test_task = apis.submit_task(
       circuit=test_circuit,
       shots=100,
       device="homebrew_s2?o=1"  # Qubit mapping only
   )

Next Steps
==========

- 📖 Read :doc:`api_reference` for complete API documentation
- 📖 Read :doc:`device_management` for optimization flags and device configuration
- 📖 Read :doc:`hardware_access` for **Homebrew_S2 specifications** and **pulse-level programming**
- 📖 Read :doc:`getting_started` for quick start examples
