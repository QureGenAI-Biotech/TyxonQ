==============
Cloud Examples
==============

This page demonstrates how to execute quantum circuits on cloud platforms and real quantum hardware using TyxonQ.

.. contents:: Contents
   :depth: 3
   :local:

.. note::
   Cloud features require proper authentication and account setup with supported providers.

Getting Started with Cloud
===========================

Basic Cloud Execution
---------------------

Running a simple circuit on a cloud simulator:

.. code-block:: python

   import tyxonq as tq

   # Create circuit
   circuit = tq.Circuit(2)
   circuit.h(0)
   circuit.cnot(0, 1)
   circuit.measure_z(0)
   circuit.measure_z(1)

   # Execute on cloud simulator
   result = circuit.provider('cloud').device('simulator').run(shots=1000)
   
   print(f"Cloud execution results: {result}")
   # Expected: {'00': ~500, '11': ~500}

**Key Concepts**:

- ``provider('cloud')``: Specify cloud provider
- ``device('simulator')``: Select cloud simulator
- Cloud execution is asynchronous - may take longer than local

Authentication Setup
--------------------

Configuring cloud credentials:

.. code-block:: python

   import tyxonq as tq
   from tyxonq.cloud import set_credentials

   # Method 1: Set credentials programmatically
   set_credentials(
       provider='cloud_provider_name',
       api_token='your_api_token_here'
   )

   # Method 2: Load from environment variables
   # Set TYXONQ_CLOUD_TOKEN in your environment
   
   # Method 3: Load from config file
   # Create ~/.tyxonq/config.json with credentials

**Security Best Practices**:

- Never hardcode tokens in source code
- Use environment variables or config files
- Rotate tokens regularly
- Keep credentials in .gitignore

Device Selection and Management
================================

Listing Available Devices
-------------------------

.. code-block:: python

   from tyxonq.cloud import list_devices

   # List all available cloud devices
   devices = list_devices(provider='cloud')
   
   for device in devices:
       print(f"Device: {device['name']}")
       print(f"  Type: {device['type']}")
       print(f"  Qubits: {device['num_qubits']}")
       print(f"  Status: {device['status']}")
       print(f"  Queue length: {device.get('queue_length', 'N/A')}")
       print()

**Expected Output**:

.. code-block:: text

   Device: cloud_simulator
     Type: simulator
     Qubits: 32
     Status: online
     Queue length: N/A
   
   Device: quantum_processor_1
     Type: hardware
     Qubits: 20
     Status: online
     Queue length: 5

Device Properties and Capabilities
----------------------------------

.. code-block:: python

   from tyxonq.cloud import get_device_properties

   # Get detailed device information
   props = get_device_properties(
       provider='cloud',
       device='quantum_processor_1'
   )
   
   print(f"Device Properties:")
   print(f"  Native gates: {props['native_gates']}")
   print(f"  Connectivity: {props['coupling_map']}")
   print(f"  T1 times: {props['t1_times']}")
   print(f"  T2 times: {props['t2_times']}")
   print(f"  Gate fidelities: {props['gate_fidelities']}")

Task Submission and Management
===============================

Submitting Quantum Jobs
-----------------------

.. code-block:: python

   from tyxonq.cloud import submit_job, get_job_status

   # Create and submit job
   circuit = tq.Circuit(3)
   circuit.h(0)
   circuit.h(1)
   circuit.h(2)
   circuit.measure_all()

   # Submit to cloud
   job = submit_job(
       circuit=circuit,
       provider='cloud',
       device='quantum_processor_1',
       shots=2048,
       priority='normal'
   )
   
   print(f"Job submitted: {job.id}")
   print(f"Status: {job.status}")

**Job Lifecycle**:

1. ``QUEUED``: Waiting in queue
2. ``VALIDATING``: Checking circuit compatibility
3. ``RUNNING``: Executing on device
4. ``COMPLETED``: Results available
5. ``FAILED``: Error occurred

Monitoring Job Status
---------------------

.. code-block:: python

   import time

   # Poll for job completion
   while True:
       status = get_job_status(job.id)
       print(f"Job {job.id}: {status}")
       
       if status in ['COMPLETED', 'FAILED', 'CANCELLED']:
           break
       
       time.sleep(5)  # Wait 5 seconds

   # Retrieve results
   if status == 'COMPLETED':
       result = job.get_result()
       print(f"Results: {result}")
   else:
       print(f"Job failed: {job.get_error()}")

Batch Job Submission
--------------------

Submit multiple circuits efficiently:

.. code-block:: python

   from tyxonq.cloud import submit_batch

   # Create multiple circuits
   circuits = []
   for i in range(10):
       c = tq.Circuit(2)
       c.ry(0, i * 0.1)
       c.cnot(0, 1)
       c.measure_all()
       circuits.append(c)

   # Submit as batch
   batch = submit_batch(
       circuits=circuits,
       provider='cloud',
       device='quantum_processor_1',
       shots=1024
   )
   
   print(f"Batch submitted: {batch.id}")
   print(f"Total jobs: {len(batch.jobs)}")
   
   # Wait for all jobs
   batch.wait_for_completion(timeout=3600)
   
   # Get all results
   results = batch.get_results()
   for i, result in enumerate(results):
       print(f"Circuit {i}: {result}")

Real Hardware Execution
=======================

Preparing Circuits for Hardware
--------------------------------

.. code-block:: python

   from tyxonq.compiler import compile_for_device
   from tyxonq.cloud import get_device_properties

   # Get device constraints
   device_props = get_device_properties(
       provider='cloud',
       device='quantum_processor_1'
   )
   
   # Create circuit
   circuit = tq.Circuit(4)
   circuit.h(0)
   circuit.h(1)
   circuit.cnot(0, 2)
   circuit.cnot(1, 3)
   circuit.measure_all()
   
   # Compile for hardware
   compiled_circuit = compile_for_device(
       circuit,
       device_properties=device_props,
       optimization_level=3
   )
   
   print(f"Original gates: {len(circuit.ops)}")
   print(f"Compiled gates: {len(compiled_circuit.ops)}")
   print(f"Native gates used: {compiled_circuit.gate_types()}")

**Compilation Benefits**:

- Converts to native gate set
- Optimizes for device topology
- Reduces circuit depth
- Improves fidelity

Error Mitigation on Hardware
----------------------------

.. code-block:: python

   from tyxonq.postprocessing import apply_readout_mitigation

   # Run calibration circuits
   calibration_results = run_calibration(
       provider='cloud',
       device='quantum_processor_1'
   )
   
   # Execute main circuit
   raw_result = circuit.provider('cloud').device('quantum_processor_1').run(
       shots=4096
   )
   
   # Apply error mitigation
   mitigated_result = apply_readout_mitigation(
       raw_result,
       calibration_data=calibration_results
   )
   
   print(f"Raw results: {raw_result}")
   print(f"Mitigated results: {mitigated_result}")

Cloud-Based VQE Example
=======================

Running VQE on Real Hardware
----------------------------

.. code-block:: python

   from tyxonq.applications.chem import HEA
   from pyscf import gto

   # Define molecule
   mol = gto.Mole()
   mol.atom = 'H 0 0 0; H 0 0 0.74'
   mol.basis = 'sto-3g'
   mol.build()

   # Create HEA with cloud execution
   hea = HEA(
       molecule=mol,
       layers=2,
       runtime='device'  # Use device runtime for cloud
   )
   
   # Configure for cloud hardware
   hea.provider = 'cloud'
   hea.device = 'quantum_processor_1'
   hea.shots = 2048
   
   # Run optimization
   energy = hea.kernel(method="COBYLA")
   
   print(f"Ground state energy (hardware): {energy:.6f}")
   print(f"Optimization iterations: {hea.opt_res['nit']}")

**Hardware Considerations**:

- Use fewer shots initially (1024) for exploration
- Increase shots (4096+) for final refinement
- Expect longer queue times
- Budget for cloud credits/costs
- Apply error mitigation for accuracy

Hybrid Cloud-Local Workflow
----------------------------

.. code-block:: python

   # 1. Develop and test locally
   hea_local = HEA(
       molecule=mol,
       layers=2,
       runtime='device'
   )
   hea_local.provider = 'simulator'
   hea_local.device = 'statevector'
   hea_local.shots = 0  # Exact simulation
   
   # Local optimization (fast)
   energy_local = hea_local.kernel(method="COBYLA")
   optimal_params = hea_local.params
   
   print(f"Local optimization: {energy_local:.6f}")
   
   # 2. Fine-tune on cloud hardware
   hea_cloud = HEA(
       molecule=mol,
       layers=2,
       runtime='device'
   )
   hea_cloud.provider = 'cloud'
   hea_cloud.device = 'quantum_processor_1'
   hea_cloud.shots = 4096
   hea_cloud.init_guess = optimal_params  # Use local result as starting point
   
   # Cloud refinement (accurate)
   energy_cloud = hea_cloud.kernel(
       method="COBYLA",
       maxiter=50  # Limited iterations on hardware
   )
   
   print(f"Cloud refinement: {energy_cloud:.6f}")

Cost Management
===============

Estimating Job Costs
--------------------

.. code-block:: python

   from tyxonq.cloud import estimate_cost

   # Estimate cost before submission
   cost_estimate = estimate_cost(
       circuit=circuit,
       provider='cloud',
       device='quantum_processor_1',
       shots=2048
   )
   
   print(f"Estimated cost: ${cost_estimate:.2f}")
   print(f"Estimated queue time: {cost_estimate.queue_time} minutes")
   
   # Proceed if acceptable
   if cost_estimate.cost < 10.0:
       job = submit_job(circuit, provider='cloud', device='quantum_processor_1')

Optimizing Cloud Usage
----------------------

**Tips for Cost Reduction**:

1. **Test locally first**: Use simulators for development
2. **Batch jobs**: Group circuits to reduce overhead
3. **Optimize circuits**: Fewer gates = lower cost
4. **Use appropriate shots**: Don't over-sample
5. **Monitor usage**: Track spending regularly

.. code-block:: python

   from tyxonq.cloud import get_usage_stats

   # Check monthly usage
   stats = get_usage_stats(provider='cloud', period='month')
   
   print(f"Monthly Usage:")
   print(f"  Jobs submitted: {stats['total_jobs']}")
   print(f"  Total shots: {stats['total_shots']:,}")
   print(f"  Cost: ${stats['total_cost']:.2f}")
   print(f"  Remaining credits: ${stats['remaining_credits']:.2f}")

Advanced Cloud Features
=======================

Circuit Caching
---------------

Reuse compiled circuits:

.. code-block:: python

   from tyxonq.cloud import cache_compiled_circuit

   # Compile and cache
   compiled = compile_for_device(circuit, device_props)
   cache_id = cache_compiled_circuit(
       compiled,
       provider='cloud',
       device='quantum_processor_1'
   )
   
   # Reuse cached circuit
   for params in parameter_sets:
       result = run_cached_circuit(
           cache_id,
           parameters=params,
           shots=1024
       )

Priority Queuing
----------------

.. code-block:: python

   # Submit high-priority job (may cost more)
   urgent_job = submit_job(
       circuit=critical_circuit,
       provider='cloud',
       device='quantum_processor_1',
       priority='high',  # Options: 'low', 'normal', 'high'
       shots=4096
   )
   
   print(f"High-priority job {urgent_job.id} submitted")
   print(f"Estimated wait time: {urgent_job.estimated_wait} minutes")

Result Persistence
------------------

.. code-block:: python

   from tyxonq.cloud import save_results, load_results

   # Save results for later analysis
   job_id = 'job_12345'
   result = get_job_result(job_id)
   
   save_results(
       result,
       filename=f'results_{job_id}.json',
       metadata={
           'circuit': 'H2_VQE',
           'device': 'quantum_processor_1',
           'date': '2025-10-11'
       }
   )
   
   # Load results later
   loaded_result = load_results(f'results_{job_id}.json')

Troubleshooting Cloud Execution
================================

Common Issues
-------------

**Issue 1: Authentication failed**

.. code-block:: python

   # Solution: Verify credentials
   from tyxonq.cloud import test_connection
   
   status = test_connection(provider='cloud')
   if not status.success:
       print(f"Connection failed: {status.error}")
       print("Please check your API token")

**Issue 2: Job timeout**

.. code-block:: python

   # Solution: Extend timeout and retry
   try:
       result = job.wait_for_completion(timeout=1800)  # 30 minutes
   except TimeoutError:
       print("Job still running, checking status...")
       status = job.get_status()
       if status == 'RUNNING':
           # Wait longer
           result = job.wait_for_completion(timeout=3600)

**Issue 3: Circuit rejected**

.. code-block:: python

   # Solution: Validate circuit before submission
   from tyxonq.cloud import validate_circuit
   
   validation = validate_circuit(
       circuit,
       provider='cloud',
       device='quantum_processor_1'
   )
   
   if not validation.valid:
       print(f"Circuit validation failed:")
       for error in validation.errors:
           print(f"  - {error}")
       
       # Fix issues
       if 'unsupported_gates' in validation.errors:
           circuit = compile_for_device(circuit, device_props)

Best Practices
==============

1. **Development Workflow**:
   
   - Develop on local simulators
   - Test on cloud simulators
   - Deploy to real hardware
   - Validate results

2. **Resource Management**:
   
   - Monitor queue lengths
   - Use batch submission
   - Cache compiled circuits
   - Track spending

3. **Quality Assurance**:
   
   - Apply error mitigation
   - Use sufficient shots
   - Validate against simulations
   - Check device calibration dates

4. **Performance**:
   
   - Optimize circuit depth
   - Use native gates when possible
   - Minimize qubit count
   - Consider device topology

See Also
========

- :doc:`basic_examples` - Local circuit execution
- :doc:`chemistry_examples` - Quantum chemistry on cloud
- :doc:`../user_guide/devices/index` - Device management guide
- :doc:`../cloud_services/index` - Cloud services documentation

Next Steps
==========

After mastering cloud execution:

1. Explore :doc:`advanced_examples` for hybrid algorithms
2. Study :doc:`../cloud_services/hardware_access` for device details
3. Learn :doc:`../postprocessing/index` for error mitigation
4. Check cloud provider documentation for specific features
