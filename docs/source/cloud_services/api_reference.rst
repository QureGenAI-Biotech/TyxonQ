API Reference
=============

=============
API Reference
=============

Complete reference for TyxonQ Cloud Services API based on the real implementation.

Cloud API Module
================

apis Module
-----------

The primary interface for cloud services.

set_token()
~~~~~~~~~~~

Configure API authentication token.

**Signature**:

.. code-block:: python

   def set_token(token: str) -> None

**Parameters**:

- ``token`` (str): Your API key (format: "username;priority")

**Example**:

.. code-block:: python

   from tyxonq.cloud import apis
   import getpass
   
   # Set token securely
   API_KEY = getpass.getpass("Input your TyxonQ API_KEY: ")
   apis.set_token(API_KEY)

set_provider()
~~~~~~~~~~~~~~

Set default provider for cloud operations.

**Signature**:

.. code-block:: python

   def set_provider(provider: str) -> None

**Parameters**:

- ``provider`` (str): Provider name ("tyxonq")

**Example**:

.. code-block:: python

   from tyxonq.cloud import apis
   
   apis.set_provider("tyxonq")

submit_task()
~~~~~~~~~~~~~

Submit quantum circuit for execution on cloud hardware.

**Signature**:

.. code-block:: python

   def submit_task(
       *,
       circuit: Circuit,
       shots: int,
       device: str,
       enable_qos_qubit_mapping: bool = None,
       enable_qos_gate_decomposition: bool = None,
       enable_qos_initial_mapping: bool = None
   ) -> TyxonQTask

**Parameters**:

- ``circuit`` (Circuit): TyxonQ Circuit object to execute
- ``shots`` (int): Number of measurement shots
- ``device`` (str): Device name with optional optimization flags (e.g., "homebrew_s2?o=7")
- ``enable_qos_qubit_mapping`` (bool, optional): Enable qubit mapping optimization (o=1)
- ``enable_qos_gate_decomposition`` (bool, optional): Enable gate decomposition (o=2)
- ``enable_qos_initial_mapping`` (bool, optional): Enable initial mapping optimization (o=4)

**Returns**: ``TyxonQTask`` object

**Example**:

.. code-block:: python

   from tyxonq.cloud import apis
   import tyxonq as tq
   
   # Create circuit
   c = tq.Circuit(2).h(0).cx(0,1).measure_all()
   
   # Submit with optimization flags via device string
   task = apis.submit_task(
       circuit=c,
       shots=100,
       device="homebrew_s2?o=7"  # All optimizations
   )
   
   # Alternative: using kwargs
   task = apis.submit_task(
       circuit=c,
       shots=100,
       device="homebrew_s2",
       enable_qos_qubit_mapping=True,
       enable_qos_gate_decomposition=True,
       enable_qos_initial_mapping=True
   )

list_devices()
~~~~~~~~~~~~~~

List available quantum devices.

**Signature**:

.. code-block:: python

   def list_devices(
       token: Optional[str] = None
   ) -> List[str]

**Parameters**:

- ``token`` (str, optional): Override authentication token

**Returns**: List of device names

**Example**:

.. code-block:: python

   from tyxonq.cloud import apis
   
   # List available devices
   devices = apis.list_devices()
   print("Available:", devices)
   # Output: ['tyxonq::homebrew_s2', ...]

Hardware Driver API
===================

tyxonq.devices.hardware.tyxonq.driver
--------------------------------------

Low-level driver for direct TyxonQ Cloud API access.

submit_task()
~~~~~~~~~~~~~

Submit task directly to TyxonQ Cloud API.

**Signature**:

.. code-block:: python

   def submit_task(
       device: str,
       token: Optional[str] = None,
       *,
       source: Optional[Union[str, Sequence[str]]] = None,
       shots: Union[int, Sequence[int]] = 1024,
       lang: str = "OPENQASM",
       **kws: Any,
   ) -> List[TyxonQTask]

**Parameters**:

- ``device`` (str): Device name with provider prefix (e.g., "tyxonq::homebrew_s2?o=7")
- ``token`` (str, optional): API authentication token
- ``source`` (str or List[str], optional): OpenQASM or TQASM source code
- ``shots`` (int or List[int]): Number of measurement shots (default 1024)
- ``lang`` (str): Source language ("OPENQASM" or "TQASM", default "OPENQASM")

**Returns**: List of ``TyxonQTask`` objects

**Example**:

.. code-block:: python

   from tyxonq.devices.hardware.tyxonq.driver import submit_task
   
   # Submit OpenQASM code
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
   
   task = tasks[0]
   print(f"Task ID: {task.id}")

get_task_details()
~~~~~~~~~~~~~~~~~~

Retrieve task status and results.

**Signature**:

.. code-block:: python

   def get_task_details(
       task: TyxonQTask, 
       token: Optional[str] = None
   ) -> Dict[str, Any]

**Parameters**:

- ``task`` (TyxonQTask): Task object from submit_task()
- ``token`` (str, optional): API authentication token

**Returns**: Task details dictionary

**Example**:

.. code-block:: python

   from tyxonq.devices.hardware.tyxonq.driver import get_task_details
   
   # Get task status and results
   details = get_task_details(task, token=YOUR_API_KEY)
   
   # Extract results
   counts = details["result"]
   metadata = details["result_meta"]
   raw_response = metadata["raw"]
   
   print("Measurement counts:", counts)
   print("Device:", metadata["device"])
   print("Shots:", metadata["shots"])

list_devices()
~~~~~~~~~~~~~~

List available devices from TyxonQ Cloud API.

**Signature**:

.. code-block:: python

   def list_devices(
       token: Optional[str] = None, 
       **kws: Any
   ) -> List[str]

**Parameters**:

- ``token`` (str, optional): API authentication token

**Returns**: List of device names with provider prefix

**Example**:

.. code-block:: python

   from tyxonq.devices.hardware.tyxonq.driver import list_devices
   
   devices = list_devices(token=YOUR_API_KEY)
   print("Available devices:", devices)
   # Output: ['tyxonq::homebrew_s2', ...]

list_properties()
~~~~~~~~~~~~~~~~~

Get detailed device properties from cloud.

**Signature**:

.. code-block:: python

   def list_properties(
       device: str, 
       token: Optional[str] = None
   ) -> Dict[str, Any]

**Parameters**:

- ``device`` (str): Device name (e.g., "homebrew_s2")
- ``token`` (str, optional): API authentication token

**Returns**: List of device property dictionaries

**Example**:

.. code-block:: python

   from tyxonq.devices.hardware.tyxonq.driver import list_properties
   
   props = list_properties(device="homebrew_s2", token=YOUR_API_KEY)
   
   for device in props:
       print(f"Device ID: {device['id']}")
       print(f"Name: {device['name']}")
       print(f"Qubits: {device['qubits']}")
       print(f"Queue: {device['queue']}")
       print(f"Status: {device['status']}")

TyxonQTask Class
================

The ``TyxonQTask`` class represents a submitted quantum job.

Class Definition
----------------

.. code-block:: python

   @dataclass
   class TyxonQTask:
       def __init__(self, id: str, device: str, status: str, task_info: None, async_result: bool):
           self.id = id
           self.device = device
           self._result = None
           self.task_info = None
           self.async_result = True
           self.status = status
           self.result_metadata = None

Attributes
----------

- ``id`` (str): Unique task identifier
- ``device`` (str): Target device name
- ``status`` (str): Current task status
- ``async_result`` (bool): Whether task is asynchronous
- ``task_info``: Additional task information
- ``result_metadata``: Cached result metadata

Methods
-------

get_result()
~~~~~~~~~~~~

Retrieve task results with optional waiting.

**Signature**:

.. code-block:: python

   def get_result(
       self, 
       token: Optional[str] = None, 
       *, 
       wait: bool = True, 
       poll_interval: float = 2.0, 
       timeout: float = 60.0
   ) -> Dict[str, Any]

**Parameters**:

- ``token`` (str, optional): Override authentication token
- ``wait`` (bool): Wait for completion (default True)
- ``poll_interval`` (float): Polling interval in seconds (default 2.0)
- ``timeout`` (float): Maximum wait time in seconds (default 60.0)

**Returns**: Task results dictionary

**Example**:

.. code-block:: python

   from tyxonq.cloud import apis
   
   # Submit task
   task = apis.submit_task(circuit=c, shots=100, device="homebrew_s2")
   
   # Non-blocking status check
   status = task.get_result(wait=False)
   print(f"Current status: {status}")
   
   # Blocking wait for completion
   result = task.get_result(wait=True, timeout=120)
   counts = result["result"]
   print("Final counts:", counts)

results()
~~~~~~~~~

Convenience method to get final results (blocks until complete).

**Example**:

.. code-block:: python

   from tyxonq.cloud import apis
   
   task = apis.submit_task(circuit=c, shots=100, device="homebrew_s2")
   result = task.results()  # Equivalent to task.get_result(wait=True)
   counts = result["result"]

Response Format
===============

Standard API Response
---------------------

All TyxonQ Cloud API responses follow this format:

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

Field Descriptions
------------------

**Top Level**:

- ``result``: Measurement counts dictionary
- ``result_meta``: Metadata about execution

**Result Metadata**:

- ``shots``: Number of shots executed
- ``device``: Device identifier
- ``raw``: Raw API response from cloud

**Task Information** (in ``raw.task``):

- ``id``: Unique job identifier
- ``queue``: Queue name (e.g., "quregenai.lab")
- ``device``: Device with optimization flags
- ``qubits``: Number of qubits used
- ``depth``: Circuit depth
- ``state``: Task state ("completed", "failed", etc.)
- ``shots``: Shot count
- ``result``: Measurement counts

**Timing Information** (in ``raw.task.ts``):

- ``pending``: Timestamp when submitted
- ``scheduled``: Timestamp when scheduled
- ``completed``: Timestamp when completed

**Performance Metrics** (in ``raw.task``):

- ``runDur``: Total runtime in microseconds
- ``atChip``: Timestamp when executed on chip
- ``durChip``: On-chip execution time in microseconds

Complete Example
================

Putting it all together:

.. code-block:: python

   import tyxonq as tq
   from tyxonq.cloud import apis
   from tyxonq import Param, waveforms
   import getpass
   
   # 1. Authentication
   API_KEY = getpass.getpass("API Key: ")
   apis.set_token(API_KEY)
   apis.set_provider("tyxonq")
   
   # 2. Device discovery
   devices = apis.list_devices()
   print("Available:", devices)
   
   # 3. Gate-level circuit
   gate_circuit = tq.Circuit(2)
   gate_circuit.h(0)
   gate_circuit.cx(0, 1)
   gate_circuit.measure_all()
   
   # Submit gate-level with optimization
   gate_task = apis.submit_task(
       circuit=gate_circuit,
       shots=100,
       device="homebrew_s2?o=7"  # All optimizations
   )
   
   gate_result = gate_task.results()
   print("Gate result:", gate_result["result"])
   
   # 4. Pulse-level circuit
   pulse_circuit = tq.Circuit(1)
   pulse_circuit.use_pulse()
   
   param = Param("q[0]")
   builder = pulse_circuit.calibrate("rabi_test", [param])
   builder.new_frame("drive_frame", param)
   builder.play("drive_frame", waveforms.CosineDrag(50, 0.2, 0.0, 0.0))
   builder.build()
   
   pulse_circuit.add_calibration('rabi_test', ['q[0]'])
   pulse_circuit.measure_z(0)
   
   # Submit pulse-level WITHOUT optimization
   pulse_task = apis.submit_task(
       circuit=pulse_circuit,
       shots=100,
       device="homebrew_s2",  # No optimization flags
       enable_qos_gate_decomposition=False,
       enable_qos_qubit_mapping=False
   )
   
   pulse_result = pulse_task.results()
   print("Pulse result:", pulse_result["result"])
   
   # 5. Monitor task timing
   def analyze_timing(result):
       raw_task = result["result_meta"]["raw"]["task"]
       run_dur = raw_task.get("runDur", 0)
       chip_dur = raw_task.get("durChip", 0)
       
       print(f"Total runtime: {run_dur / 1000:.1f} ms")
       print(f"On-chip time: {chip_dur / 1000:.1f} ms")
       print(f"Queue+overhead: {(run_dur - chip_dur) / 1000:.1f} ms")
   
   analyze_timing(gate_result)
   analyze_timing(pulse_result)

Next Steps
==========

- 📖 Read :doc:`getting_started` for quick start guide
- 📖 Read :doc:`device_management` for optimization flags and device configuration
- 📖 Read :doc:`task_submission` for advanced task management workflows
- 📖 Read :doc:`hardware_access` for **Homebrew_S2 specifications** and **pulse-level programming**
- 📖 **External docs**: ``docs/tyxonq_cloud_api.md`` and ``docs/pulse_support_en.md``
