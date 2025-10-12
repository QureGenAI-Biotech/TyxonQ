=========
Optimizer
=========

The Optimizer library provides specialized optimization algorithms and integration tools for variational quantum algorithms (VQAs). It features the SOAP optimizer—a lightweight, gradient-free method designed specifically for noisy quantum objective functions—and seamless interoperability with SciPy's optimization ecosystem.

.. contents:: Table of Contents
   :depth: 3
   :local:

Overview
========

Variational quantum algorithms like VQE and QAOA require classical optimization of noisy, expensive-to-evaluate objective functions. The Optimizer library addresses these challenges by providing:

🚀 **SOAP Algorithm**
   Sequential Optimization with Approximate Parabola—a custom optimizer for noisy VQE landscapes

🔧 **SciPy Integration**
   Wrapper utilities for seamless use of SciPy optimizers with TyxonQ energy functions

📊 **Trajectory Tracking**
   Built-in support for recording optimization history and convergence analysis

⚡ **Minimal Dependencies**
   Lightweight implementation with only NumPy and SciPy requirements

🎯 **VQA-Optimized**
   Designed for the specific challenges of quantum variational algorithms

.. mermaid::

   graph TD
       A[Optimizer Library] --> B[SOAP]
       A --> C[Interop Layer]
       
       B --> D[Gradient-Free]
       B --> E[Noise-Tolerant]
       B --> F[Parabolic Fitting]
       
       C --> G[SciPy Wrapper]
       C --> H[Type Conversion]
       C --> I[Gradient Handling]
       
       D --> J[VQE Applications]
       E --> J
       G --> J

Library Architecture
====================

The optimizer library consists of two main components:

.. list-table:: Optimizer Components
   :header-rows: 1
   :widths: 30 40 30

   * - Component
     - Purpose
     - Key Features
   * - SOAP Optimizer
     - Custom gradient-free optimization
     - Parabolic fitting, direction management, noise tolerance
   * - Interop Layer
     - SciPy integration
     - Type safety, gradient wrapping, backend compatibility

SOAP Optimizer
==============

Sequential Optimization with Approximate Parabola (SOAP) is a specialized optimizer for variational quantum algorithms.

Algorithmic Foundation
----------------------

**Core Principle**: SOAP explores the parameter space along coordinate directions, fitting parabolas to estimate local curvature and determine optimal step sizes.

**Key Algorithm Steps**:

1. **Direction Selection**: Maintain a prioritized list of search directions
2. **Parabolic Sampling**: Evaluate objective at 3-4 points along each direction
3. **Optimal Step**: Fit parabola :math:`f(x) \approx ax^2 + bx + c`, step to minimum at :math:`x^* = -b/(2a)`
4. **Adaptive Directions**: Periodically update directions based on optimization history
5. **Convergence Check**: Monitor moving average of recent improvements

.. mermaid::

   flowchart TD
       Start([Initialize x₀]) --> InitDirs["Initialize direction list<br/>(sorted by |x₀|)"]
       InitDirs --> Loop{"nfev < maxfev?"}
       
       Loop -->|Yes| CheckDirs{"Directions<br/>available?"}
       CheckDirs -->|Yes| SelectDir["Select next<br/>direction"]
       CheckDirs -->|No| RebuildDirs["Rebuild directions<br/>(average direction)"]
       
       RebuildDirs --> CheckDirs
       
       SelectDir --> Sample3["Sample at -δ, 0, +δ"]
       Sample3 --> CheckMin{"Minimum<br/>at boundary?"}
       
       CheckMin -->|Yes| Sample4["Extend to 4 points<br/>(-4δ or +4δ)"]
       CheckMin -->|No| FitParabola
       
       Sample4 --> FitParabola["Fit parabola<br/>f(x) = ax² + bx + c"]
       FitParabola --> ComputeStep["Compute step:<br/>x* = -b/(2a)"]
       
       ComputeStep --> UpdateParams["Update parameters:<br/>x ← x - x*·direction"]
       UpdateParams --> EvalNew["Evaluate f(x_new)"]
       
       EvalNew --> Callback{"Callback?"}
       Callback -->|Yes| CallUser["callback(x_new)"]
       Callback -->|No| CheckConv
       CallUser --> CheckConv
       
       CheckConv{"Converged?<br/>Δf < atol"}
       CheckConv -->|Yes| Return["Return OptimizeResult"]
       CheckConv -->|No| Loop
       
       Loop -->|No| Return
       Return --> End([End])

**Convergence Criterion**: 

SOAP monitors the moving average of energy improvements over the last :math:`2n` iterations (where :math:`n` is the parameter count). Convergence is declared when:

.. math::

   \frac{1}{n}\sum_{i=k-2n}^{k-n} E_i - \frac{1}{n}\sum_{i=k-n}^{k} E_i < \text{atol}

API Reference
-------------

.. py:function:: soap(fun, x0, args=(), u=0.1, maxfev=2000, atol=1e-3, callback=None, ret_traj=False, **kwargs)

   Sequential Optimization with Approximate Parabola.

   A SciPy-compatible gradient-free optimizer designed for noisy objective functions typical in variational quantum algorithms.

   :param fun: Objective function to minimize. Signature: ``f(x, *args) -> float``
   :type fun: Callable[[np.ndarray, Any], float]
   :param x0: Initial parameters
   :type x0: np.ndarray
   :param args: Additional arguments passed to ``fun``
   :type args: Tuple[Any, ...], optional
   :param u: Initial step scale factor (default: 0.1)
   :type u: float, optional
   :param maxfev: Maximum number of function evaluations (default: 2000)
   :type maxfev: int, optional
   :param atol: Absolute tolerance for convergence (default: 1e-3)
   :type atol: float, optional
   :param callback: Called after each iteration with current parameters
   :type callback: Callable[[np.ndarray], None], optional
   :param ret_traj: If True, include full parameter trajectory in result
   :type ret_traj: bool, optional
   :param kwargs: Additional keyword arguments (ignored, for compatibility)
   :return: Optimization result with fields:
   
            - ``x``: Optimal parameters
            - ``fun``: Optimal function value
            - ``nit``: Number of iterations
            - ``nfev``: Number of function evaluations
            - ``fun_list``: Energy history (np.ndarray)
            - ``nfev_list``: Function evaluation counts at each iteration
            - ``trajectory``: Full parameter history (if ``ret_traj=True``)
            - ``success``: Always True
   :rtype: scipy.optimize.OptimizeResult

**Hyperparameter Guide**:

.. list-table:: SOAP Hyperparameters
   :header-rows: 1
   :widths: 20 15 65

   * - Parameter
     - Default
     - Guidance
   * - ``u``
     - 0.1
     - Initial step scale. Larger values for coarse landscapes, smaller for fine-tuning
   * - ``maxfev``
     - 2000
     - Budget for function evaluations. Increase for complex landscapes
   * - ``atol``
     - 1e-3
     - Convergence tolerance. Tighten for higher accuracy, relax for noisy functions
   * - ``ret_traj``
     - False
     - Enable for detailed convergence analysis (increases memory usage)

Usage Examples
--------------

**Example 1: Basic VQE Optimization**

.. code-block:: python

   from tyxonq.libs.optimizer import soap
   import numpy as np

   # Define a noisy energy function (simulating VQE)
   def noisy_energy(params, noise_level=0.01):
       """Simulated VQE energy landscape."""
       # True minimum at [π/4, π/4]
       true_energy = np.sum(np.cos(params))
       noise = np.random.normal(0, noise_level)
       return true_energy + noise

   # Initial parameters
   x0 = np.random.rand(4) * 2 * np.pi

   # Run SOAP optimization
   result = soap(
       fun=noisy_energy,
       x0=x0,
       u=0.2,           # Moderate step size
       maxfev=500,      # Function evaluation budget
       atol=1e-3,       # Convergence threshold
       ret_traj=True    # Track full trajectory
   )

   print(f"Optimal energy: {result.fun:.6f}")
   print(f"Optimal params: {result.x}")
   print(f"Iterations: {result.nit}")
   print(f"Function evals: {result.nfev}")

   # Plot convergence
   import matplotlib.pyplot as plt
   plt.plot(result.nfev_list, result.fun_list)
   plt.xlabel('Function Evaluations')
   plt.ylabel('Energy')
   plt.title('SOAP Convergence')
   plt.show()

**Example 2: VQE with Callback Monitoring**

.. code-block:: python

   from tyxonq.libs.optimizer import soap
   from tyxonq.libs.circuits_library import hardware_efficient
   import numpy as np

   # Store optimization history
   history = {'params': [], 'energies': []}

   def vqe_energy(params):
       """Compute VQE energy (pseudo-code)."""
       circuit = hardware_efficient(num_qubits=4, depth=2, params=params)
       energy = circuit.expectation(hamiltonian)  # Simulated
       return energy

   def monitor_progress(params):
       """Callback to track progress."""
       history['params'].append(params.copy())
       energy = vqe_energy(params)
       history['energies'].append(energy)
       print(f"Iteration {len(history['energies'])}: E = {energy:.6f}")

   # Optimize with monitoring
   x0 = np.random.rand(16) * 0.1  # Small initial angles
   result = soap(
       fun=vqe_energy,
       x0=x0,
       u=0.1,
       maxfev=1000,
       atol=5e-4,
       callback=monitor_progress
   )

   print(f"\nFinal ground state energy: {result.fun:.6f}")

**Example 3: Comparing with SciPy Optimizers**

.. code-block:: python

   from tyxonq.libs.optimizer import soap
   from scipy.optimize import minimize
   import numpy as np
   import time

   def benchmark_function(params):
       """Rosenbrock function (noisy)."""
       noise = np.random.normal(0, 0.01)
       return sum(100*(params[1:]-params[:-1]**2)**2 + (1-params[:-1])**2) + noise

   x0 = np.random.rand(6)

   # SOAP
   t0 = time.time()
   result_soap = soap(benchmark_function, x0, maxfev=500)
   time_soap = time.time() - t0

   # SciPy COBYLA (gradient-free)
   t0 = time.time()
   result_cobyla = minimize(
       benchmark_function, x0, 
       method='COBYLA', 
       options={'maxiter': 500}
   )
   time_cobyla = time.time() - t0

   # Compare results
   print("\n=== Optimization Comparison ===")
   print(f"SOAP:    E={result_soap.fun:.4f}, t={time_soap:.2f}s, nfev={result_soap.nfev}")
   print(f"COBYLA:  E={result_cobyla.fun:.4f}, t={time_cobyla:.2f}s, nfev={result_cobyla.nfev}")

Performance Characteristics
---------------------------

**Strengths**:

- ✅ **Noise Tolerance**: Robust to measurement shot noise and statistical fluctuations
- ✅ **No Gradient Required**: Suitable for non-differentiable or discrete objectives
- ✅ **Adaptive Directions**: Automatically adjusts search directions based on landscape
- ✅ **Lightweight**: Minimal memory footprint and computational overhead

**Limitations**:

- ❌ **Convergence Speed**: Slower than gradient-based methods on smooth landscapes
- ❌ **Scaling**: Efficiency decreases with high-dimensional parameter spaces (n > 50)
- ❌ **Local Optima**: Like most local optimizers, can get trapped in local minima

**Typical Performance**:

.. list-table:: SOAP Performance Benchmarks
   :header-rows: 1
   :widths: 30 25 45

   * - Problem Size
     - Function Evals
     - Typical Use Case
   * - 4-8 params
     - 200-500
     - Small VQE ansätze, QAOA p=1-2
   * - 10-20 params
     - 500-1500
     - Medium VQE, HEA depth 3-4
   * - 30-50 params
     - 1500-3000
     - Large ansätze, deep circuits

SciPy Integration Layer
=======================

The interop layer provides seamless integration between TyxonQ energy functions and SciPy's optimization ecosystem.

API Reference
-------------

.. py:function:: scipy_opt_wrap(f, gradient=True)

   Wrap a TyxonQ energy function for use with SciPy optimizers.

   This wrapper handles:
   
   - Type conversion to NumPy float64
   - Backend-agnostic array handling
   - Gradient unpacking for gradient-based optimizers
   - Return value standardization

   :param f: Energy function to wrap. Signature:
   
            - If ``gradient=True``: ``f(params, *args) -> (value, grad)``
            - If ``gradient=False``: ``f(params, *args) -> value``
   :type f: Callable
   :param gradient: Whether function returns gradients (default: True)
   :type gradient: bool
   :return: Wrapped function compatible with SciPy optimizers
   :rtype: Callable

**Wrapped Function Behavior**:

- **Input**: Accepts NumPy arrays (converted to float64)
- **Output**: 
  
  - If ``gradient=True``: Returns ``(float, np.ndarray)`` for value and gradient
  - If ``gradient=False``: Returns ``float``

- **Backend Conversion**: Automatically converts TyxonQ backend arrays to NumPy

Usage Examples
--------------

**Example 1: Gradient-Free Optimization**

.. code-block:: python

   from tyxonq.libs.optimizer.interop import scipy_opt_wrap
   from scipy.optimize import minimize
   import numpy as np

   # TyxonQ energy function (no gradients)
   def vqe_energy_nograd(params):
       """Compute VQE energy using shot-based measurement."""
       # ... circuit construction and measurement ...
       return energy  # Single value

   # Wrap for SciPy
   wrapped = scipy_opt_wrap(vqe_energy_nograd, gradient=False)

   # Use with SciPy's gradient-free methods
   x0 = np.random.rand(8) * 0.1
   result = minimize(
       wrapped,
       x0,
       method='COBYLA',
       options={'maxiter': 300}
   )

   print(f"Optimized energy: {result.fun}")
   print(f"Optimal parameters: {result.x}")

**Example 2: Gradient-Based Optimization**

.. code-block:: python

   from tyxonq.libs.optimizer.interop import scipy_opt_wrap
   from scipy.optimize import minimize
   from tyxonq.numerics import get_backend
   import numpy as np

   backend = get_backend('pytorch')  # Supports auto-diff

   def vqe_energy_with_grad(params):
       """Compute VQE energy and gradients."""
       # Convert to backend tensor
       params_tensor = backend.asarray(params)
       params_tensor.requires_grad = True
       
       # ... circuit construction ...
       energy = compute_expectation(params_tensor)
       
       # Compute gradients
       energy.backward()
       grad = params_tensor.grad.numpy()
       
       return float(energy), grad

   # Wrap for SciPy
   wrapped = scipy_opt_wrap(vqe_energy_with_grad, gradient=True)

   # Use with gradient-based methods
   x0 = np.random.rand(12) * 0.2
   result = minimize(
       wrapped,
       x0,
       method='L-BFGS-B',
       jac=True,  # Tell SciPy we provide gradients
       options={'maxiter': 100}
   )

   print(f"Optimized energy: {result.fun}")
   print(f"Convergence: {result.success}")

**Example 3: Integration with Parameter-Shift Rule**

.. code-block:: python

   from tyxonq.libs.optimizer.interop import scipy_opt_wrap
   from scipy.optimize import minimize
   import numpy as np

   def parameter_shift_gradient(circuit_fn, params, shift=np.pi/2):
       """Compute gradients using parameter-shift rule."""
       grad = np.zeros_like(params)
       for i in range(len(params)):
           params_plus = params.copy()
           params_minus = params.copy()
           params_plus[i] += shift
           params_minus[i] -= shift
           
           grad[i] = (circuit_fn(params_plus) - circuit_fn(params_minus)) / 2
       return grad

   def vqe_with_param_shift(params):
       """VQE energy with parameter-shift gradients."""
       energy = circuit_evaluate(params)  # Your circuit function
       grad = parameter_shift_gradient(circuit_evaluate, params)
       return energy, grad

   # Wrap and optimize
   wrapped = scipy_opt_wrap(vqe_with_param_shift, gradient=True)
   
   x0 = np.zeros(10)
   result = minimize(
       wrapped,
       x0,
       method='L-BFGS-B',
       jac=True,
       options={'maxiter': 50, 'ftol': 1e-6}
   )

**Example 4: Multi-Method Comparison**

.. code-block:: python

   from tyxonq.libs.optimizer import soap
   from tyxonq.libs.optimizer.interop import scipy_opt_wrap
   from scipy.optimize import minimize
   import numpy as np

   # Define energy function
   def energy_function(params):
       # Simulated VQE energy
       return np.sum(params**2) + 0.1 * np.random.randn()

   x0 = np.random.rand(10)
   methods_to_test = [
       ('SOAP', lambda: soap(energy_function, x0, maxfev=500)),
       ('COBYLA', lambda: minimize(scipy_opt_wrap(energy_function, False), x0, method='COBYLA')),
       ('Powell', lambda: minimize(scipy_opt_wrap(energy_function, False), x0, method='Powell')),
       ('Nelder-Mead', lambda: minimize(scipy_opt_wrap(energy_function, False), x0, method='Nelder-Mead')),
   ]

   print("\n=== Optimizer Comparison ===")
   for name, optimizer in methods_to_test:
       result = optimizer()
       print(f"{name:15s}: E={result.fun:8.5f}, nfev={result.nfev:4d}")

Integration Workflow
--------------------

.. mermaid::

   sequenceDiagram
       participant User
       participant SciPy as SciPy Optimizer
       participant Wrapper as scipy_opt_wrap
       participant TyxonQ as TyxonQ Energy Function
       participant Backend as Numeric Backend
       
       User->>SciPy: minimize(wrapped_fn, x0)
       SciPy->>Wrapper: Call with NumPy params
       Wrapper->>Wrapper: Convert to float64
       Wrapper->>TyxonQ: Forward to energy function
       TyxonQ->>Backend: Compute with backend arrays
       Backend-->>TyxonQ: Return (energy, gradient)
       TyxonQ-->>Wrapper: Return result
       Wrapper->>Wrapper: Convert backend arrays to NumPy
       Wrapper-->>SciPy: Return (float, np.ndarray)
       SciPy->>SciPy: Update parameters
       
       loop Until convergence
           SciPy->>Wrapper: Evaluate with new params
           Wrapper->>TyxonQ: Forward call
           TyxonQ-->>Wrapper: Return energy/grad
           Wrapper-->>SciPy: Return NumPy types
       end
       
       SciPy-->>User: Return OptimizeResult

Best Practices
==============

Choosing an Optimizer
---------------------

**Decision Tree**:

.. mermaid::

   graph TD
       Start[Choose Optimizer] --> HasGrad{"Gradients<br/>available?"}
       
       HasGrad -->|Yes| GradCost{"Gradient<br/>cost?"}
       HasGrad -->|No| NoiseLevel{"Noise<br/>level?"}
       
       GradCost -->|Cheap| UseLBFGS["L-BFGS-B<br/>(SciPy)"]
       GradCost -->|Expensive| UseAdam["Adam<br/>(Custom)"]
       
       NoiseLevel -->|High| UseSOAP["SOAP<br/>(Noise-tolerant)"]
       NoiseLevel -->|Low| ParamCount{"Param<br/>count?"}
       
       ParamCount -->|< 20| UseCOBYLA["COBYLA<br/>(SciPy)"]
       ParamCount -->|≥ 20| UseSOAP2["SOAP or<br/>Powell"]
       
       UseLBFGS --> ValidateConv["Validate Convergence"]
       UseAdam --> ValidateConv
       UseSOAP --> ValidateConv
       UseCOBYLA --> ValidateConv
       UseSOAP2 --> ValidateConv

**Recommendation Matrix**:

.. list-table:: Optimizer Selection Guide
   :header-rows: 1
   :widths: 25 25 25 25

   * - Scenario
     - Gradients?
     - Noise?
     - Recommended
   * - Small VQE (n<10)
     - No
     - Low
     - COBYLA
   * - Small VQE (n<10)
     - No
     - High
     - SOAP
   * - Large VQE (n>20)
     - No
     - Any
     - SOAP
   * - Any size
     - Yes (cheap)
     - Low
     - L-BFGS-B
   * - Any size
     - Yes (param-shift)
     - Medium
     - Adam or SOAP
   * - QAOA
     - No
     - High
     - SOAP
   * - Chemistry VQE
     - Yes (auto-diff)
     - Low
     - L-BFGS-B

Optimization Tips
-----------------

**1. Initialization Strategy**:

.. code-block:: python

   # Good: Small random initialization near identity
   x0 = np.random.rand(n_params) * 0.1

   # Better: Zero initialization for symmetric Hamiltonians
   x0 = np.zeros(n_params)

   # Best: Problem-specific heuristics
   x0 = initialize_from_classical_solution()

**2. Hyperparameter Tuning**:

.. code-block:: python

   # Start with loose tolerance for exploration
   result1 = soap(energy, x0, atol=1e-2, maxfev=200)

   # Refine with tighter tolerance
   result2 = soap(energy, result1.x, atol=1e-4, maxfev=300)

**3. Convergence Monitoring**:

.. code-block:: python

   import matplotlib.pyplot as plt

   result = soap(energy, x0, ret_traj=True)

   # Plot energy convergence
   plt.figure(figsize=(12, 4))
   
   plt.subplot(1, 2, 1)
   plt.plot(result.nfev_list, result.fun_list)
   plt.xlabel('Function Evaluations')
   plt.ylabel('Energy')
   plt.title('Energy Convergence')
   
   plt.subplot(1, 2, 2)
   plt.plot(np.gradient(result.fun_list))
   plt.xlabel('Iteration')
   plt.ylabel('Energy Gradient')
   plt.title('Convergence Rate')
   
   plt.tight_layout()
   plt.show()

**4. Multi-Start Strategy**:

.. code-block:: python

   # Run multiple optimizations from different initializations
   n_starts = 5
   results = []
   
   for i in range(n_starts):
       x0 = np.random.rand(n_params) * 0.2 - 0.1
       result = soap(energy, x0, maxfev=500)
       results.append(result)
   
   # Select best result
   best_result = min(results, key=lambda r: r.fun)
   print(f"Best energy from {n_starts} starts: {best_result.fun}")

Common Pitfalls
---------------

**❌ Pitfall 1: Ignoring Noise**

.. code-block:: python

   # BAD: Using tight tolerance with noisy function
   result = soap(noisy_vqe, x0, atol=1e-8)  # Will never converge!
   
   # GOOD: Match tolerance to noise level
   noise_std = 0.01
   result = soap(noisy_vqe, x0, atol=10*noise_std)

**❌ Pitfall 2: Insufficient Budget**

.. code-block:: python

   # BAD: Too few evaluations for complex landscape
   result = soap(energy, x0, maxfev=50)  # Likely incomplete
   
   # GOOD: Generous budget, check convergence
   result = soap(energy, x0, maxfev=2000)
   if result.nfev >= 2000:
       print("Warning: Reached maxfev, may not have converged")

**❌ Pitfall 3: Mismatched Wrapper**

.. code-block:: python

   # BAD: Wrong gradient flag
   def energy_with_grad(params):
       return value, grad
   
   wrapped = scipy_opt_wrap(energy_with_grad, gradient=False)  # Wrong!
   
   # GOOD: Correct gradient flag
   wrapped = scipy_opt_wrap(energy_with_grad, gradient=True)

See Also
========

- :doc:`../circuits_library/index` - Circuit templates for VQE and QAOA
- :doc:`../../user_guide/core/index` - Quantum circuits and gates
- :doc:`../../tutorials/index` - Tutorial: VQE optimization workflow
- :doc:`../../examples/index` - Complete VQE examples
- :doc:`../../api/libs/optimizer` - Complete API reference

Further Reading
===============

**Variational Quantum Algorithms**

.. [Peruzzo2014] A. Peruzzo et al.,  
   "A variational eigenvalue solver on a photonic quantum processor",  
   Nature Communications, 5, 4213 (2014)

.. [McClean2016] J. R. McClean et al.,  
   "The theory of variational hybrid quantum-classical algorithms",  
   New Journal of Physics, 18, 023023 (2016)

**Optimization Methods**

.. [Spall1998] J. C. Spall,  
   "Implementation of the simultaneous perturbation algorithm for stochastic optimization",  
   IEEE Transactions on Aerospace and Electronic Systems, 34, 817 (1998)

.. [Nelder1965] J. A. Nelder and R. Mead,  
   "A simplex method for function minimization",  
   The Computer Journal, 7, 308 (1965)

.. [Powell1964] M. J. D. Powell,  
   "An efficient method for finding the minimum of a function",  
   The Computer Journal, 7, 155 (1964)
