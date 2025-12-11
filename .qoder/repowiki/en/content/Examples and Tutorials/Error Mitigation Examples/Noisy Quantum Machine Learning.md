# Noisy Quantum Machine Learning

<cite>
**Referenced Files in This Document**   
- [noisy_qml.py](file://examples-ng/noisy_qml.py)
- [mcnoise_boost.py](file://examples-ng/mcnoise_boost.py)
- [mcnoise_boost_v2.py](file://examples-ng/mcnoise_boost_v2.py)
- [vqe_noisyopt.py](file://examples/vqe_noisyopt.py)
</cite>

## Table of Contents
1. [Introduction](#introduction)
2. [Project Structure](#project-structure)
3. [Core Components](#core-components)
4. [Architecture Overview](#architecture-overview)
5. [Detailed Component Analysis](#detailed-component-analysis)
6. [Dependency Analysis](#dependency-analysis)
7. [Performance Considerations](#performance-considerations)
8. [Troubleshooting Guide](#troubleshooting-guide)
9. [Conclusion](#conclusion)

## Introduction
This document provides a comprehensive analysis of Noisy Quantum Machine Learning (QML) using variational quantum circuits under depolarizing noise. The implementation integrates Monte Carlo noise simulation within the QML training loop, leveraging vmapped noise trajectories for efficient gradient computation. The workflow is built on PyTorch for optimization and includes key components such as circuit ansatz design, parameterized noise injection via status flags, and a sigmoid-based classification loss function. The system processes MNIST data through custom preprocessing and batched training using a PyTorch Dataset. Vectorized value and gradient computation (vvag) enables high-performance training under varying noise levels, with performance insights and convergence behavior analyzed throughout.

## Project Structure
The project is organized into modular components that support quantum machine learning with noise simulation. Key directories include `examples-ng` for advanced QML workflows and `examples` for foundational demonstrations. The noisy QML implementation resides in `examples-ng/noisy_qml.py`, which orchestrates the full training pipeline. Supporting noise simulation techniques are demonstrated in `mcnoise_boost.py` and `mcnoise_boost_v2.py`, while `vqe_noisyopt.py` provides comparative insights into noisy variational quantum eigensolvers.

```mermaid
graph TB
subgraph "Examples"
A[noisy_qml.py] --> B[MNIST Dataset]
A --> C[Monte Carlo Noise Simulation]
A --> D[PyTorch Optimization]
E[mcnoise_boost.py] --> F[Layerwise Slicing]
G[mcnoise_boost_v2.py] --> H[Precompute State]
I[vqe_noisyopt.py] --> J[Parameter Shift Gradients]
end
```

**Diagram sources**
- [noisy_qml.py](file://examples-ng/noisy_qml.py)
- [mcnoise_boost.py](file://examples-ng/mcnoise_boost.py)
- [mcnoise_boost_v2.py](file://examples-ng/mcnoise_boost_v2.py)
- [vqe_noisyopt.py](file://examples/vqe_noisyopt.py)

**Section sources**
- [noisy_qml.py](file://examples-ng/noisy_qml.py)
- [mcnoise_boost.py](file://examples-ng/mcnoise_boost.py)
- [mcnoise_boost_v2.py](file://examples-ng/mcnoise_boost_v2.py)
- [vqe_noisyopt.py](file://examples/vqe_noisyopt.py)

## Core Components
The core components of the noisy QML system include the quantum circuit ansatz, Monte Carlo noise simulation, vectorized gradient computation, and classification loss function. The circuit processes MNIST images encoded into quantum states, applies a layered variational ansatz with depolarizing noise, and measures expectations for classification. Noise is simulated using random seeds passed through vmapped trajectories, enabling differentiable noise modeling. The loss function uses a sigmoid transformation for binary classification, and optimization is performed using Adam with separate learning rates for circuit parameters and scaling factors.

**Section sources**
- [noisy_qml.py](file://examples-ng/noisy_qml.py#L0-L227)
- [vqe_noisyopt.py](file://examples/vqe_noisyopt.py#L0-L288)

## Architecture Overview
The architecture integrates classical deep learning frameworks with quantum circuit simulation under noise. The workflow begins with MNIST data preprocessing, followed by quantum state encoding and variational circuit execution. Noise is injected via depolarizing channels controlled by status flags derived from random seeds. Multiple noise trajectories are processed in parallel using `vmap`, and gradients are computed using `vvag` for end-to-end training. The PyTorch backend enables seamless integration of quantum operations within a classical optimization loop.

```mermaid
graph TD
A[MNIST Data] --> B[Preprocessing]
B --> C[Quantum Encoding]
C --> D[Variational Circuit]
D --> E[Depolarizing Noise]
E --> F[Expectation Measurement]
F --> G[Sigmoid Classification]
G --> H[Cross-Entropy Loss]
H --> I[PyTorch Adam Optimizer]
I --> J[Parameter Update]
J --> D
K[Random Seeds] --> E
L[vmapped Trajectories] --> E
M[vvag Gradient Computation] --> I
```

**Diagram sources**
- [noisy_qml.py](file://examples-ng/noisy_qml.py#L0-L227)

## Detailed Component Analysis

### Quantum Circuit Ansatz and Noise Injection
The variational quantum circuit consists of a data encoding layer followed by multiple entangling and rotation layers. Each layer applies CNOT gates for entanglement and parameterized RX/RZ rotations. Depolarizing noise is injected after each CNOT gate using status flags derived from random seeds, allowing precise control over noise realization per trajectory.

```mermaid
flowchart TD
Start([Circuit Initialization]) --> DataEncoding["Data Encoding via RX Gates"]
DataEncoding --> LayerLoop["For Each Layer j"]
LayerLoop --> Entangle["Apply CNOT Chain"]
Entangle --> NoiseX["Depolarizing Channel on Qubit i"]
Entangle --> NoiseY["Depolarizing Channel on Qubit i+1"]
NoiseX --> RotateZ["Parameterized RZ Rotations"]
NoiseY --> RotateZ
RotateZ --> RotateX["Parameterized RX Rotations"]
RotateX --> LayerLoop
LayerLoop --> Measure["Expectation Measurement"]
Measure --> Output["Mean of Z-Basis Expectations"]
```

**Diagram sources**
- [noisy_qml.py](file://examples-ng/noisy_qml.py#L86-L109)

**Section sources**
- [noisy_qml.py](file://examples-ng/noisy_qml.py#L86-L109)

### Training Workflow and Optimization
The training loop integrates noise simulation, forward pass, loss computation, and gradient-based optimization. It uses two Adam optimizers: one for circuit parameters and another for a scaling factor that controls the sigmoid nonlinearity. The loop supports variable noise levels and number of noise trajectories (noc), with periodic inference to monitor convergence.

```mermaid
sequenceDiagram
participant Trainer as "train()"
participant Loss as "loss()"
participant Circuit as "Quantum Circuit"
participant Optimizer as "Adam Optimizer"
Trainer->>Trainer : Initialize parameters and optimizers
loop For each iteration
Trainer->>Trainer : Sample batch from MNISTDataset
Trainer->>Trainer : Generate random noise seeds
Trainer->>Loss : Call loss(param, scale, seeds, x, y, pn)
Loss->>Circuit : Execute vmapped circuit evaluations
Circuit-->>Loss : Return expectations
Loss->>Loss : Apply sigmoid and compute cross-entropy
Loss-->>Trainer : Return loss and predictions
Trainer->>Optimizer : Backward pass via loss.backward()
Optimizer->>Trainer : Update parameters
Trainer->>Trainer : Optional inference every val_step
end
Trainer->>Trainer : Save trained parameters
```

**Diagram sources**
- [noisy_qml.py](file://examples-ng/noisy_qml.py#L142-L195)

**Section sources**
- [noisy_qml.py](file://examples-ng/noisy_qml.py#L142-L195)

### Dataset Preprocessing and Batching
The MNIST dataset is preprocessed using either bilinear interpolation to a √n×√n grid or PCA dimensionality reduction. A custom PyTorch Dataset class enables batched training with random sampling, where each iteration returns a new random batch of fixed size. This design supports stochastic training while maintaining compatibility with quantum circuit input requirements.

```mermaid
classDiagram
class MNISTDataset {
+x : torch.Tensor
+y : torch.Tensor
+maxiter : int
+__init__(x, y, maxiter)
+__len__()
+__getitem__(idx)
}
class DataLoader {
+dataset : MNISTDataset
+batch_size : int
+shuffle : bool
}
MNISTDataset --> DataLoader : "used by"
```

**Diagram sources**
- [noisy_qml.py](file://examples-ng/noisy_qml.py#L69-L81)

**Section sources**
- [noisy_qml.py](file://examples-ng/noisy_qml.py#L69-L81)

## Dependency Analysis
The system relies on several key dependencies for functionality. The primary dependency is PyTorch for tensor operations and automatic differentiation. The TyxonQ framework provides quantum circuit simulation capabilities, including noise channels and expectation computation. Additional dependencies include torchvision for MNIST data loading and NumPy for numerical operations. The integration of `vvag` for vectorized gradients and `vmap` for parallel noise trajectory simulation creates a tightly coupled computational graph that enables efficient training under noise.

```mermaid
graph TD
A[PyTorch] --> B[Autograd]
A --> C[Adam Optimizer]
A --> D[Tensor Operations]
E[TyxonQ] --> F[Quantum Circuit]
E --> G[Depolarizing Channel]
E --> H[Expectation Computation]
I[torchvision] --> J[MNIST Dataset]
K[NumPy] --> L[Data Preprocessing]
B --> M[Loss Backward]
D --> N[vmapped Execution]
F --> N
J --> O[Data Loading]
O --> P[Custom Dataset]
P --> N
```

**Diagram sources**
- [noisy_qml.py](file://examples-ng/noisy_qml.py)
- [vqe_noisyopt.py](file://examples/vqe_noisyopt.py)

**Section sources**
- [noisy_qml.py](file://examples-ng/noisy_qml.py)
- [vqe_noisyopt.py](file://examples/vqe_noisyopt.py)

## Performance Considerations
The implementation demonstrates significant performance considerations in noisy QML training. The use of `K.jit` and `vmap` enables efficient execution of multiple noise trajectories, reducing both staging and running times. The `mcnoise_boost` examples show that layerwise slicing and state precomputation can dramatically reduce compilation overhead. Training performance is affected by noise level, number of trajectories, and circuit depth, with deeper circuits requiring more computational resources. The separation of parameter and scale optimization allows for adaptive learning but increases memory usage due to multiple optimizer states.

**Section sources**
- [mcnoise_boost.py](file://examples-ng/mcnoise_boost.py)
- [mcnoise_boost_v2.py](file://examples-ng/mcnoise_boost_v2.py)
- [noisy_qml.py](file://examples-ng/noisy_qml.py)

## Troubleshooting Guide
Common issues in this implementation include gradient computation failures under high noise, memory exhaustion with large batch sizes, and convergence problems due to improper scaling. Ensure that the `PYTORCH_ENABLE_MPS_FALLBACK` environment variable is set for GPU compatibility. When training fails to converge, try reducing the learning rate or increasing the number of noise trajectories. For memory issues, reduce batch size or use fewer noise trajectories. Debugging can be enabled in the inference function to inspect prediction values.

**Section sources**
- [noisy_qml.py](file://examples-ng/noisy_qml.py#L198-L220)
- [noisy_qml.py](file://examples-ng/noisy_qml.py#L138-L138)

## Conclusion
This analysis demonstrates a complete framework for Noisy Quantum Machine Learning using variational circuits with Monte Carlo noise simulation. The integration of PyTorch optimization with vmapped noise trajectories enables efficient training under depolarizing noise. Key components including the circuit ansatz, parameterized noise injection, and sigmoid-based classification work together to form a robust QML pipeline. The use of `vvag` for vectorized gradients and custom dataset handling for MNIST preprocessing showcases the flexibility of the framework. Performance can be further optimized using techniques like layerwise slicing, making this approach suitable for large-scale noisy quantum machine learning experiments.