DefcalLibrary: Hardware Calibration Management
==============================================

Overview
--------

The ``DefcalLibrary`` class provides a framework for managing hardware-specific quantum gate calibrations. It enables users to:

- Store optimal pulse parameters measured from hardware characterization
- Organize calibrations by gate and qubit
- Query calibrations during circuit compilation
- Persist calibrations to JSON for deployment across systems
- Enable hardware-aware pulse generation during compilation

This system bridges the gap between hardware characterization and quantum circuit execution, allowing optimal gate parameters to be automatically applied when compiling gate-level circuits to pulses.

Key Concepts
------------

**Calibration Data**
   A calibration represents the optimal pulse parameters for a specific quantum gate on a specific qubit (or qubit pair). It includes:

   - Gate name (e.g., "x", "h", "cx")
   - Target qubits (single qubit or multi-qubit)
   - Pulse waveform object (e.g., ``Drag``, ``Gaussian``)
   - Parameters (amplitude, duration, sigma, beta, etc.)
   - Metadata (description, hardware identifier, timestamp)

**Hardware Heterogeneity**
   Real quantum processors have qubit-to-qubit variations due to:

   - Fabrication tolerances
   - Frequency spreading
   - Coupling variations
   
   DefcalLibrary captures these variations with per-qubit calibrations.

**Compilation Priority**
   When converting gates to pulses, the compiler uses:

   1. **DefcalLibrary** (if available) - User-provided hardware calibrations
   2. **Circuit metadata** - Legacy calibration format
   3. **Default decomposition** - Generic physics-based pulses

API Reference
-------------

DefcalLibrary Class
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   class DefcalLibrary(hardware: str = "Homebrew_S2")

Initialize a calibration library for a specific hardware platform.

**Parameters:**

- ``hardware`` (str): Hardware identifier (default: "Homebrew_S2")

**Example:**

.. code-block:: python

   from tyxonq.compiler.pulse_compile_engine import DefcalLibrary
   
   lib = DefcalLibrary(hardware="Homebrew_S2")

add_calibration()
~~~~~~~~~~~~~~~~~

.. code-block:: python

   def add_calibration(
       gate: str,
       qubits: Union[int, Tuple[int, ...], List[int]],
       pulse: Any,
       params: Optional[Dict[str, Any]] = None,
       description: str = ""
   ) -> None

Add or update a gate calibration in the library.

**Parameters:**

- ``gate`` (str): Gate name (e.g., "x", "h", "cx")
- ``qubits`` (int, tuple, or list): Target qubit(s)
- ``pulse``: Pulse waveform object (e.g., ``Drag``, ``Gaussian``)
- ``params`` (dict, optional): Calibration parameters (duration, amplitude, etc.)
- ``description`` (str, optional): Human-readable description

**Example:**

.. code-block:: python

   from tyxonq import waveforms
   
   # Single-qubit calibration
   x_pulse = waveforms.Drag(amp=0.8, duration=40, sigma=10, beta=0.18)
   lib.add_calibration(
       gate="x",
       qubits=(0,),
       pulse=x_pulse,
       params={"duration": 40, "amp": 0.8},
       description="X gate on q0 from hardware characterization"
   )
   
   # Two-qubit calibration
   cx_pulse = waveforms.Drag(amp=0.35, duration=160, sigma=40, beta=0.1)
   lib.add_calibration(
       gate="cx",
       qubits=(0, 1),
       pulse=cx_pulse,
       params={"duration": 160, "amp": 0.35}
   )

get_calibration()
~~~~~~~~~~~~~~~~~

.. code-block:: python

   def get_calibration(
       gate: str,
       qubits: Union[int, Tuple[int, ...], List[int], None] = None
   ) -> Union[CalibrationData, List[CalibrationData], None]

Retrieve calibration(s) by gate and qubits.

**Parameters:**

- ``gate`` (str): Gate name
- ``qubits`` (int, tuple, list, or None): Target qubits
  
  - If specified: returns exact match or None
  - If None: returns list of all calibrations for the gate

**Returns:**

- Single ``CalibrationData`` if exact match
- List of ``CalibrationData`` if qubits is None
- None if no match found

**Example:**

.. code-block:: python

   # Get specific calibration
   calib = lib.get_calibration("x", (0,))
   if calib:
       print(f"X on q0: amplitude={calib.params['amp']}")
   
   # Get all X gate calibrations
   all_x = lib.get_calibration("x", None)
   print(f"Total X calibrations: {len(all_x)}")

list_calibrations()
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def list_calibrations(
       gate: Optional[str] = None,
       qubit: Optional[int] = None
   ) -> List[CalibrationData]

List calibrations with optional filtering.

**Parameters:**

- ``gate`` (str, optional): Filter by gate name
- ``qubit`` (int, optional): Filter by qubit index

**Returns:**

List of matching ``CalibrationData`` objects

**Example:**

.. code-block:: python

   # All X gates
   x_calibs = lib.list_calibrations(gate="x")
   
   # All gates on q0
   q0_calibs = lib.list_calibrations(qubit=0)
   
   # All calibrations
   all_calibs = lib.list_calibrations()

has_calibration()
~~~~~~~~~~~~~~~~~

.. code-block:: python

   def has_calibration(
       gate: str,
       qubits: Union[int, Tuple[int, ...], List[int]]
   ) -> bool

Check if a calibration exists.

**Parameters:**

- ``gate`` (str): Gate name
- ``qubits`` (int, tuple, or list): Target qubits

**Returns:**

True if calibration exists, False otherwise

**Example:**

.. code-block:: python

   if lib.has_calibration("x", (0,)):
       print("X gate on q0 is calibrated")

export_to_json()
~~~~~~~~~~~~~~~~

.. code-block:: python

   def export_to_json(filepath: Union[str, Path]) -> None

Export calibration library to JSON file.

**Parameters:**

- ``filepath`` (str or Path): Output JSON file path

**Example:**

.. code-block:: python

   lib.export_to_json("homebrew_s2_calibrations.json")

import_from_json()
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def import_from_json(
       filepath: Union[str, Path],
       pulse_factory: Optional[callable] = None,
       overwrite: bool = False
   ) -> None

Import calibrations from JSON file.

**Parameters:**

- ``filepath`` (str or Path): Input JSON file path
- ``pulse_factory`` (callable, optional): Function to reconstruct pulse objects
- ``overwrite`` (bool): If True, clear existing calibrations before import

**Example:**

.. code-block:: python

   lib = DefcalLibrary()
   lib.import_from_json("homebrew_s2_calibrations.json")

validate()
~~~~~~~~~~

.. code-block:: python

   def validate() -> bool

Validate all calibrations in the library.

**Checks:**

- All calibrations have non-None pulse objects
- Gate names are valid strings
- Qubit indices are non-negative integers
- Parameters are properly structured

**Returns:**

True if all calibrations are valid

**Example:**

.. code-block:: python

   if lib.validate():
       print("All calibrations are valid")

summary()
~~~~~~~~~

.. code-block:: python

   def summary() -> str

Return human-readable summary of library contents.

**Returns:**

Formatted string with calibration summary

**Example:**

.. code-block:: python

   print(lib.summary())
   # Output:
   # DefcalLibrary Summary
   # ==================================================
   # Hardware: Homebrew_S2
   # Total Calibrations: 7
   # Created: 2025-10-30T12:34:56.123456
   #
   # Calibrations by Gate:
   #   X:
   #     q[0]: Drag
   #     q[1]: Drag
   #   H:
   #     q[0]: Drag
   #   CX:
   #     q[0,1]: Drag

CalibrationData Class
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   @dataclass
   class CalibrationData:
       gate: str
       qubits: Tuple[int, ...]
       pulse: Any
       params: Dict[str, Any]
       timestamp: Optional[datetime]
       description: str
       hardware: str

Represents a single quantum gate calibration.

**Attributes:**

- ``gate``: Gate name (lowercase)
- ``qubits``: Target qubit indices
- ``pulse``: Pulse waveform object
- ``params``: Calibration parameters (duration, amplitude, etc.)
- ``timestamp``: When calibration was created
- ``description``: Human-readable description
- ``hardware``: Hardware identifier

Integration with Compilation
-----------------------------

Using DefcalLibrary with Gate-to-Pulse Compilation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``GateToPulsePass`` compiler automatically queries DefcalLibrary during compilation:

.. code-block:: python

   from tyxonq import Circuit, waveforms
   from tyxonq.compiler.pulse_compile_engine import DefcalLibrary
   from tyxonq.compiler.pulse_compile_engine.native.gate_to_pulse import GateToPulsePass
   
   # Step 1: Create and populate calibration library
   lib = DefcalLibrary(hardware="Homebrew_S2")
   
   x_pulse = waveforms.Drag(amp=0.8, duration=40, sigma=10, beta=0.18)
   lib.add_calibration("x", (0,), x_pulse, {"duration": 40, "amp": 0.8})
   
   # Step 2: Create gate-level circuit
   circuit = Circuit(3)
   circuit.h(0)
   circuit.x(0)
   circuit.cx(0, 1)
   circuit.measure_z(0)
   
   # Step 3: Compile with defcal-aware compiler
   compiler = GateToPulsePass(defcal_library=lib)
   
   device_params = {
       "qubit_freq": [5.0e9, 5.05e9, 4.95e9],
       "anharmonicity": [-330e6, -330e6, -330e6],
   }
   
   pulse_circuit = compiler.execute_plan(
       circuit,
       device_params=device_params,
       mode="pulse_only"
   )
   
   # Step 4: Execute with both modes
   # Mode A: Realistic sampling (shots > 0)
   result = pulse_circuit.device(provider="simulator").run(shots=1024)
   
   # Mode B: Ideal simulation (shots = 0)
   state = pulse_circuit.state(backend="numpy")

Compilation Priority
~~~~~~~~~~~~~~~~~~~~

The compiler uses the following priority when converting gates to pulses:

1. **DefcalLibrary** - User-provided calibrations (highest priority)
2. **Circuit metadata** - Legacy calibration format
3. **Default decomposition** - Physics-based gates (lowest priority)

This ensures that hardware-specific calibrations override default parameters.

Best Practices
--------------

**1. Hardware Characterization**

Perform characterization on real hardware once:

.. code-block:: python

   # One-time: Characterize hardware
   lib = DefcalLibrary(hardware="Homebrew_S2")
   
   # Measure optimal X gate on q0
   x_pulse_q0 = measure_x_gate(qubit=0)  # Custom measurement routine
   lib.add_calibration("x", (0,), x_pulse_q0)
   
   # Export for reuse
   lib.export_to_json("production_calibrations.json")

**2. Deployment**

Use saved calibrations across many programs:

.. code-block:: python

   # Many times: Use calibrations in production
   lib = DefcalLibrary()
   lib.import_from_json("production_calibrations.json")
   
   # Create and compile circuits using the calibrations
   circuit = Circuit(3)
   circuit.h(0)
   # ... build circuit ...
   
   compiler = GateToPulsePass(defcal_library=lib)
   pulse_circuit = compiler.execute_plan(circuit, device_params={...})
   pulse_circuit.device(provider="tyxonq").run(shots=1024)

**3. Validation**

Always validate calibrations before deployment:

.. code-block:: python

   lib.import_from_json("calibrations.json")
   
   if not lib.validate():
       raise RuntimeError("Calibrations are invalid")
   
   print(lib.summary())  # Inspect contents

**4. Testing**

Test with both execution modes:

.. code-block:: python

   # Test circuit
   circuit = Circuit(2)
   circuit.h(0)
   circuit.x(1)
   circuit.measure_z(0)
   circuit.measure_z(1)
   
   compiler = GateToPulsePass(defcal_library=lib)
   pulse_circuit = compiler.execute_plan(circuit, device_params={...})
   
   # Mode A: Realistic sampling
   result_sampling = pulse_circuit.device(provider="simulator").run(shots=1024)
   print(f"Sampling result: {result_sampling}")
   
   # Mode B: Ideal state vector
   state_ideal = pulse_circuit.state()
   print(f"Ideal state: {state_ideal}")

Quick Start: Chain API with Sampling
-------------------------------------

The simplest way to use DefcalLibrary in production is via the complete chain API:

.. code-block:: python

   from tyxonq import Circuit, waveforms
   from tyxonq.compiler.pulse_compile_engine import DefcalLibrary
   from tyxonq.compiler.pulse_compile_engine.native.gate_to_pulse import GateToPulsePass

   # 1. Setup calibrations
   lib = DefcalLibrary(hardware="Homebrew_S2")
   lib.add_calibration("x", (0,), waveforms.Drag(amp=0.8, duration=40, ...), {...})

   # 2. Build circuit
   circuit = Circuit(2)
   circuit.h(0)
   circuit.x(1)
   circuit.measure_z(0)

   # 3. Compile with defcal
   compiler = GateToPulsePass(defcal_library=lib)
   pulse_circuit = compiler.execute_plan(
       circuit,
       device_params={"qubit_freq": [5.0e9, 5.05e9]},
       mode="pulse_only"
   )

   # 4. Execute (chain API) - realistic sampling
   result = pulse_circuit.device(provider="simulator").run(shots=1024)

   # 5. Get results
   counts = result[0].get('result', {})
   for state, count in sorted(counts.items()):
       print(f"|{state}⟩: {count/1024:.4f}")

This single chain demonstrates the complete workflow:

::

   Circuit (gates)
       ↓
   .compile(defcal)  ← Hardware calibrations applied
       ↓
   .device()         ← Select simulator
       ↓
   .run(shots=1024)  ← Realistic sampling (NOT ideal)
       ↓
   Results with optimized pulses

Examples
--------

See the following example files for complete working code:

- ``examples/defcal_hardware_calibration.py`` - Hardware characterization workflow
- ``examples/defcal_circuit_compilation.py`` - Circuit compilation with defcal
- ``examples/defcal_performance_comparison.py`` - Performance benchmarks
- ``examples/defcal_integration_in_workflow.py`` - Complete integration examples with chain API

Related Topics
--------------

- :doc:`../pulse/index` - Pulse programming overview
- :doc:`../../api/compiler` - Compilation API
- :doc:`../../tutorials/advanced/pulse_defcal_integration` - Integration tutorial with detailed steps
