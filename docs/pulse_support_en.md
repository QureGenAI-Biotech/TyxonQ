# TyxonQ Pulse Interface User Guide

## Table of Contents

- [Overview](#overview)
- [TQASM 0.2 Syntax Specification](#tqasm-02-syntax-specification)
- [Core Components](#core-components)
- [Waveform Parameter Overview](#waveform-parameter-overview)
- [Detailed Waveform Parameter Description](#detailed-waveform-parameter-description)
- [Usage Methods](#usage-methods)
- [TQASM Output Format](#tqasm-output-format)
- [Advanced Features](#advanced-features)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)
- [Practical Application Examples](#practical-application-examples)
- [Summary](#summary)

---

## Overview

TyxonQ provides a powerful pulse-level control interface that allows users to directly manipulate quantum bit pulse signals for precise quantum control. Through the Pulse interface, you can:

- Define custom pulse waveforms
- Create quantum bit calibration programs
- Implement advanced quantum control algorithms
- Generate TQASM 0.2 format pulse-level circuits

### Supported Waveform Types

Currently supports the following four main waveform types:
- **cosine_drag** - Cosine DRAG waveform for suppressing leakage state transitions
- **flattop** - Flat-top waveform suitable for quantum state preparation
- **gaussian** - Gaussian waveform providing smooth pulse transitions
- **sine** - Sine waveform for periodic oscillation experiments

For detailed parameter definitions and mathematical expressions, please refer to the waveform parameter description section below.

## TQASM 0.2 Syntax Specification

### Syntax Definition

TQASM 0.2 uses Backus-Naur Form (BNF) to define syntax structure:

```
<pulse> ::= <defcal>

<defcal> ::= "defcal" <id> <idlist> { <calgrammar> }

<calgrammar> ::= <calstatement>
               | <calgrammar> <calstatement>

<calstatement> ::= <framedecl>
                | <waveformplay>

<framedecl> ::= "frame" <id> "=" "newframe" (<idlist>);

<waveformplay> ::= "play" (<id>, <waveform>);

<waveform> ::= <id> (<explist>)
```

### Keyword Description

| Keyword | Function Description | Syntax Format |
|---------|---------------------|---------------|
| `defcal` | Define custom parameterized waveform quantum gates | `defcal <gate_name> <parameter_list> { <calibration_statements> }` |
| `frame` | Declare a variable as frame type | `frame <frame_name> = newframe(<qubit>);` |
| `newframe` | Create a new frame on target qubit for carrying waveforms | `newframe(<qubit_identifier>)` |
| `play` | Play waveform on specified frame | `play(<frame_name>, <waveform_function>(<parameters>));` |

### Supported Waveform Types

Currently supported waveform functions include:
- `cosine_drag(duration, amp, phase, alpha)` - Cosine DRAG waveform
- `flattop(duration, amp, width)` - Flat-top waveform
- `gaussian(duration, amp, sigma, angle)` - Gaussian waveform
- `sin(duration, amp, phase, freq, angle)` - Sine waveform

### Complete Example

Below is a complete TQASM 0.2 code example showing how to define and use parameterized waveforms:

```tqasm
TQASM 0.2;
QREG q[1];

defcal hello_world a {
  frame drive_frame = newframe(a);
  play(drive_frame, cosine_drag(50, 0.2, 0.0, 0.0));
}

hello_world q[0];
```

### Code Analysis

1. **TQASM 0.2;** - Declare use of TQASM 0.2 version
2. **QREG q[1];** - Define quantum register with 1 qubit
3. **defcal hello_world a { ... }** - Define calibration program named "hello_world" with parameter "a"
4. **frame drive_frame = newframe(a);** - Create frame named "drive_frame" on qubit "a"
5. **play(drive_frame, cosine_drag(50, 0.2, 0.0, 0.0));** - Play cosine DRAG waveform on frame
6. **hello_world q[0];** - Call calibration program on qubit q[0]

### Waveform Parameter Description

Parameters in `cosine_drag(50, 0.2, 0.0, 0.0)`:
- `50` - Pulse duration (sampling periods)
- `0.2` - Waveform amplitude
- `0.0` - Phase angle (radians)
- `0.0` - DRAG coefficient

## Core Components

TyxonQ Pulse interface core components include waveform types, parameterization support, and calibration builders. These components work together to provide users with complete pulse control capabilities.

---

### 1. Waveform Types

TyxonQ supports multiple predefined pulse waveform types, each with specific parameters:

#### Gaussian Waveform
```python
from tyxonq import waveforms

# Create Gaussian waveform: amplitude, duration, standard deviation
gaussian_wf = waveforms.Gaussian(amp=0.5, duration=100, sigma=20)
```

#### Gaussian Square Waveform
```python
# Create Gaussian square waveform: amplitude, duration, standard deviation, width
gaussian_square_wf = waveforms.GaussianSquare(amp=0.5, duration=100, sigma=20, width=60)
```

#### DRAG Waveform
```python
# Create DRAG waveform: amplitude, duration, standard deviation, beta parameter
drag_wf = waveforms.Drag(amp=0.5, duration=100, sigma=20, beta=0.5)
```

#### Constant Waveform
```python
# Create constant waveform: amplitude, duration
constant_wf = waveforms.Constant(amp=0.5, duration=100)
```

#### Sine Waveform
```python
# Create sine waveform: amplitude, frequency, duration
sine_wf = waveforms.Sine(amp=0.5, frequency=0.1, duration=100)
```

#### Cosine Waveform
```python
# Create cosine waveform: amplitude, frequency, duration
cosine_wf = waveforms.Cosine(amp=0.5, frequency=0.1, duration=100)
```

#### Cosine DRAG Waveform
```python
# Create cosine DRAG waveform: amplitude, duration, phase, alpha parameter
cosine_drag_wf = waveforms.CosineDrag(amp=0.5, duration=100, phase=0.0, alpha=0.2)
```

#### Flat-top Waveform
```python
# Create flat-top waveform: amplitude, width, duration
flattop_wf = waveforms.Flattop(amp=0.5, width=60, duration=100)
```

### 2. Parameterization Support

All waveforms support parameterization using the `Param` class:

```python
from tyxonq import Param

# Create parameterized waveform
param_t = Param("t")
parametric_wf = waveforms.CosineDrag(param_t, 0.2, 0.0, 0.0)
```

### 3. Calibration Builder

`DefcalBuilder` is the core tool for creating quantum bit calibration programs:

```python
from tyxonq import Circuit, Param

# Create circuit and enable pulse mode
qc = Circuit(1)
qc.use_pulse()

# Create parameters
param0 = Param("a")

# Start building calibration program
builder = qc.calibrate("calibration_name", [param0])

# Define frame
builder.new_frame("drive_frame", param0)

# Play waveform
builder.play("drive_frame", waveforms.CosineDrag(param0, 0.2, 0.0, 0.0))

# Build calibration program
builder.build()
```

## Waveform Parameter Overview

The table below provides a quick reference for all supported waveforms, including parameter formats and main application scenarios:

| No. | Waveform Type | Waveform Parameters | Main Purpose |
|-----|---------------|-------------------|--------------|
| 1 | `cosine_drag` | `CosineDrag(duration, amp, phase, alpha)` | Precise control for suppressing leakage state transitions |
| 2 | `flattop` | `Flattop(duration, amp, width)` | Flat-top pulse for quantum state preparation |
| 3 | `gaussian` | `Gaussian(duration, amp, sigma, angle)` | Gaussian pulse for smooth transitions |
| 4 | `sin` | `Sin(duration, amp, phase, freq, angle)` | Sine pulse for periodic oscillation |
| 5 | `drag` | `Drag(duration, amp, sigma, beta)` | DRAG protocol for superconducting qubit control |
| 6 | `constant` | `Constant(duration, amp)` | Constant pulse for DC bias |
| 7 | `gaussian_square` | `GaussianSquare(duration, amp, sigma, width)` | Gaussian-edged square wave |
| 8 | `cosine` | `Cosine(duration, amp, freq, phase)` | Cosine pulse |

## Detailed Waveform Parameter Description

This section provides detailed parameter definitions, mathematical expressions, and physical meanings for each waveform. Each waveform has specific application scenarios and parameter constraints.

---

### 1. CosineDrag Waveform Parameters

| Parameter | Type | Description | Constraints |
|-----------|------|-------------|-------------|
| `amp` | real value | Waveform amplitude | \|amp\| ≤ 2 |
| `duration` | int | Pulse length (sampling periods) | 0 < duration < 10000 |
| `phase` | real value | Phase angle (radians) | No special restrictions |
| `alpha` | real value | DRAG coefficient | \|alpha\| ≤ 10 |

**Mathematical Expression**: 
- `g(x) = (Amp / 2) × e^(i × phase) × [cos((2πx / duration) - π) + 1]`
- `output(x) = g(x) + i × alpha × g'(x)`
- Domain: `x ∈ [0, duration)`

**Parameter Description**: 
- `amp`: Waveform amplitude, controls waveform intensity
- `duration`: Pulse duration in sampling periods
- `phase`: Phase angle, controls waveform phase offset
- `alpha`: DRAG coefficient for suppressing leakage state transitions

### 2. Flattop Waveform Parameters

| Parameter | Type | Description | Constraints |
|-----------|------|-------------|-------------|
| `amp` | real value | Waveform amplitude | amp ≤ 2 |
| `width` | real value | FWHM of Gaussian component | width ≤ 100 |
| `duration` | int | Pulse length (sampling periods) | duration ≤ 100,000 |

**Mathematical Expression**: 
- `w = width` (FWHM of Gaussian component)
- `σ = w / √(4 log 2)` (standard deviation)
- `A = amp` (amplitude)
- `T = duration` (duration)
- `output(x) = (A / 2) × [erf((w + T - x) / σ) - erf((w - x) / σ)]`
- Domain: `x ∈ [0, T + 2w)`

**Parameter Description**: 
- `amp`: Waveform amplitude, controls overall waveform intensity
- `width`: FWHM of Gaussian component, controls Gaussian edge width
- `duration`: Pulse duration, controls flat-top section length

### 3. Gaussian Waveform Parameters

| Parameter | Type | Description | Constraints |
|-----------|------|-------------|-------------|
| `amp` | real value | Waveform amplitude | \|amp\| ≤ 2 |
| `duration` | int | Pulse length (sampling periods) | 0 < duration < 10000 |
| `sigma` | real value | Gaussian waveform standard deviation | No special restrictions |
| `angle` | real value | Complex phase factor angle (radians) | No special restrictions |

**Mathematical Expression**: 
- `f'(x) = exp(- (1/2) × ((x - duration/2)² / sigma²))`
- `f(x) = A × f'(x)` when `0 ≤ x < duration`
- `A = amp × exp(i × angle)`

**Parameter Description**: 
- `amp`: Waveform amplitude, controls waveform intensity
- `duration`: Pulse duration in sampling periods
- `sigma`: Gaussian distribution standard deviation, controls waveform width
- `angle`: Complex phase factor, controls waveform phase

### 4. Sine Waveform Parameters

| Parameter | Type | Description | Constraints |
|-----------|------|-------------|-------------|
| `amp` | real value | Sine wave amplitude, waveform range [-amp, amp] | \|amp\| ≤ 2 |
| `phase` | real value | Sine wave phase (radians) | No special restrictions |
| `freq` | real value | Sine wave frequency (reciprocal of sampling period) | No special restrictions |
| `angle` | real value | Complex phase factor angle (radians) | No special restrictions |
| `duration` | int | Pulse length (sampling periods) | 0 < duration < 10000 |

**Mathematical Expression**: 
- `f(x) = A sin(2π × freq × x + phase)` when `0 ≤ x < duration`
- `A = amp × exp(i × angle)`

**Parameter Description**: 
- `amp`: Sine wave amplitude, controls waveform intensity, range [-amp, amp]
- `phase`: Sine wave phase, controls waveform phase offset
- `freq`: Sine wave frequency in reciprocal of sampling period
- `angle`: Complex phase factor, controls complex waveform phase
- `duration`: Pulse duration in sampling periods

### 5. Other Waveform Parameters

#### GaussianSquare Waveform
| Parameter | Type | Description | Constraints |
|-----------|------|-------------|-------------|
| `amp` | real value | Waveform amplitude | \|amp\| ≤ 2 |
| `duration` | int | Pulse length (sampling periods) | 0 < duration < 10000 |
| `sigma` | real value | Gaussian component standard deviation | No special restrictions |
| `width` | real value | Square wave section width | width ≤ duration |

#### Drag Waveform
| Parameter | Type | Description | Constraints |
|-----------|------|-------------|-------------|
| `amp` | real value | Waveform amplitude | \|amp\| ≤ 2 |
| `duration` | int | Pulse length (sampling periods) | 0 < duration < 10000 |
| `sigma` | real value | Gaussian component standard deviation | No special restrictions |
| `beta` | real value | DRAG parameter for suppressing leakage state transitions | No special restrictions |

**Mathematical Expression**: 
- `f(x) = A × exp(-(x - duration/2)² / (2 × sigma²))` when `0 ≤ x < duration`
- `A = amp × exp(i × angle)` (if angle parameter is supported)

#### Constant Waveform
| Parameter | Type | Description | Constraints |
|-----------|------|-------------|-------------|
| `amp` | real value | Constant amplitude | \|amp\| ≤ 2 |
| `duration` | int | Pulse length (sampling periods) | 0 < duration < 10000 |

**Mathematical Expression**: 
- `f(x) = amp` when `0 ≤ x < duration`
- Domain: `x ∈ [0, duration)`

**Parameter Description**: 
- `amp`: Constant amplitude, remains constant throughout duration
- `duration`: Pulse duration in sampling periods

## Usage Methods

TyxonQ Pulse interface provides an intuitive and easy-to-use API, allowing users to easily create complex pulse control programs.

---

### Basic Workflow

1. **Enable Pulse Mode**
```python
qc = Circuit(n_qubits)
qc.use_pulse()
```

2. **Define Calibration Program**
```python
# Use DefcalBuilder to build calibration program
builder = qc.calibrate("cal_name", [param1, param2])
builder.new_frame("frame_name", qubit_param)
builder.play("frame_name", waveform)
builder.build()
```

3. **Call Calibration Program**
```python
# Call calibration program in circuit
qc.add_calibration('cal_name', ['q[0]'])
```

4. **Generate TQASM Code**
```python
tqasm_code = qc.to_tqasm()
```

### Complete Examples

#### Example 1: Simple Rabi Oscillation Experiment

```python
import sys
import os
sys.path.insert(0, "..")

from tyxonq import Circuit, Param, waveforms
from tyxonq.cloud import apis

def create_rabi_circuit(t):
    """Create Rabi oscillation experiment circuit"""
    qc = Circuit(1)
    qc.use_pulse()
    
    # Create parameters
    param_t = Param("t")
    
    # Build calibration program
    builder = qc.calibrate("rabi_experiment", [param_t])
    builder.new_frame("drive_frame", param_t)
    builder.play("drive_frame", waveforms.CosineDrag(param_t, 0.2, 0.0, 0.0))
    builder.build()
    
    # Call calibration program
    qc.add_calibration('rabi_experiment', ['q[0]'])
    
    return qc

# Create circuits with different time parameters
for t in [10, 30, 50, 70, 90]:
    qc = create_rabi_circuit(t)
    print(f"TQASM for t={t}:")
    print(qc.to_tqasm())
    print("-" * 50)
```

## TQASM Output Format

Generated TQASM code follows TQASM 0.2 standard:

```tqasm
TQASM 0.2;
QREG q[1];

defcal rabi_experiment a {
  frame drive_frame = newframe(a);
  play(drive_frame, cosine_drag(a, 0.2, 0.0, 0.0));
}

rabi_experiment q[0];
```

## Advanced Features

### 1. Time Control

Can add start time parameter to waveforms:

```python
builder.play("frame_name", waveform, start_time=50)
```

### 2. Complex Calibration Programs

Can build complex calibration programs with multiple instructions:

```python
builder = qc.calibrate("complex_cal", [param])
builder.new_frame("frame1", param)
builder.play("frame1", waveform1)
builder.new_frame("frame2", param)
builder.play("frame2", waveform2)
builder.build()
```

### 3. Cloud API Integration

```python
from tyxonq.cloud import apis

# Set authentication
apis.set_token("your_token")
apis.set_provider("tyxonq")

# Submit pulse circuit task
task = apis.submit_task(
    circuit=qc,
    shots=1000,
    device="homebrew_s2",
    enable_qos_gate_decomposition=False,
    enable_qos_qubit_mapping=False,
)

# Get results
result = task.results()
```

## Best Practices

1. **Parameter Naming**: Use meaningful parameter names for easy understanding and debugging
2. **Waveform Selection**: Choose appropriate waveform types based on physical requirements
3. **Time Units**: Pay attention to time parameter units (usually nanoseconds)
4. **Error Handling**: Verify TQASM code correctness before submitting to hardware
5. **Documentation**: Add comments and explanations for complex calibration programs
6. **Complex Phase**: Some waveforms support complex phase factors (angle parameter) for fine phase control
7. **Domain**: Pay attention to each waveform's domain range, ensure reasonable parameter settings

## Waveform Selection Guide

### Choose Waveforms Based on Application Scenarios

- **CosineDrag**: Suitable for precise control requiring leakage state transition suppression, such as single-qubit gate operations
- **Flattop**: Suitable for applications requiring flat-top pulses, such as quantum state preparation
- **Gaussian**: Suitable for pulses requiring smooth transitions, such as adiabatic evolution
- **Sine**: Suitable for applications requiring periodic oscillations, such as Rabi oscillation experiments
- **Drag**: Suitable for precise control of superconducting qubits
- **Constant**: Suitable for simple constant pulses, such as DC bias
- **GaussianSquare**: Suitable for square wave pulses with Gaussian edges

## Troubleshooting

### Common Issues

1. **Unsupported Waveform Type**: Ensure use of predefined waveform types
2. **Parameter Type Error**: Check if parameters are `Param` type or numerical values
3. **Frame Not Defined**: Ensure frame is defined before playing waveform
4. **TQASM Generation Failure**: Check calibration program build order

## Practical Application Examples

### Example: Precise Rabi Oscillation Experiment

Based on mathematical definition for precise parameter setting:

```python
from tyxonq import Circuit, Param, waveforms

def create_precise_rabi_circuit(t_duration, amplitude, frequency):
    """
    Create precise Rabi oscillation experiment circuit
    
    Parameters:
    - t_duration: Pulse duration (sampling periods)
    - amplitude: Sine wave amplitude (|amp| ≤ 2)
    - frequency: Sine wave frequency (reciprocal of sampling period)
    """
    qc = Circuit(1)
    qc.use_pulse()
    
    # Create parameterized waveform
    param_t = Param("t")
    
    sine_wave = waveforms.Sine(
        duration=t_duration,      # Duration
        amp=amplitude,            # Amplitude
        frequency=frequency,      # Frequency
    )

    
    # Build calibration program
    builder = qc.calibrate("precise_rabi", [param_t])
    builder.new_frame("drive_frame", param_t)
    builder.play("drive_frame", sine_wave)
    builder.build()
    
    # Call calibration program
    qc.add_calibration('precise_rabi', ['q[0]'])
    
    return qc

# Create circuits with different parameters for parameter scanning
frequencies = [0.01, 0.02, 0.05, 0.1]  # Different frequencies
amplitudes = [0.5, 1.0, 1.5]            # Different amplitudes

for freq in frequencies:
    for amp in amplitudes:
        qc = create_precise_rabi_circuit(
            t_duration=100,    # 100 sampling periods
            amplitude=amp,      # Amplitude
            frequency=freq      # Frequency
        )
        print(f"Frequency: {freq}, Amplitude: {amp}")
        print(qc.to_tqasm())
        print("-" * 50)
```

### Example: DRAG Pulse Optimization

```python
def create_optimized_drag_pulse():
    """Create optimized DRAG pulse"""
    qc = Circuit(1)
    qc.use_pulse()
    
    param_qubit = Param("q")
    
    # Use DRAG protocol to suppress leakage state transitions
    # f(x) = A × exp(-(x - duration/2)² / (2 × sigma²))
    # A = amp × exp(i × angle)
    drag_wave = waveforms.Drag(
        duration=100,    
        amp=1.0,         # Amplitude
        sigma=20,        # Gaussian standard deviation
        beta=0.5         # DRAG parameter for suppressing leakage states
    )
    
    builder = qc.calibrate("optimized_drag", [param_qubit])
    builder.new_frame("drive_frame", param_qubit)
    builder.play("drive_frame", drag_wave)
    builder.build()
    
    qc.add_calibration('optimized_drag', ['q[0]'])
    return qc
```

## Summary

TyxonQ's Pulse interface provides powerful microwave pulse control capabilities. Through reasonable use of waveform types, parameterization, and calibration builders, you can implement various complex quantum control experiments. It's recommended to start with simple examples and gradually build more complex applications.

---

### Key Points

1. **Mathematical Precision**: Each waveform has precise mathematical definitions ensuring accurate physical implementation
2. **Parameter Constraints**: Strictly adhere to parameter constraint conditions to avoid hardware errors
3. **Application-Oriented**: Choose appropriate waveform types based on specific quantum control requirements
4. **Experimental Design**: Utilize parameterization features for systematic parameter scanning and optimization 