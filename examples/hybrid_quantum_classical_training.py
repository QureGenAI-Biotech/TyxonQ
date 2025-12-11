"""
Hybrid Quantum-Classical Training Pipeline with GPU Acceleration

This example demonstrates TyxonQ's seamless integration with PyTorch for
hybrid quantum-classical machine learning. Both quantum circuit simulation
and classical neural network layers run on GPU with automatic differentiation.

Key Features:
1. End-to-end PyTorch integration (quantum + classical)
2. GPU acceleration via PyTorch backend
3. Automatic differentiation through quantum layers
4. Practical MNIST classification task
5. Manual batch processing for quantum circuits

Architecture:
  Input → Quantum PQC Layer → Classical Linear → Sigmoid → Binary Classification

Note:
  This example uses manual batching for quantum circuits. For production use,
  consider implementing custom autograd functions or using vectorized operations.

Performance:
  - GPU: ~10x faster than CPU for classical layers
  - Quantum circuit evaluation happens per-sample

Migrated from: examples-ng/hybrid_gpu_pipeline.py
Reference: Quantum machine learning best practices
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

import tyxonq as tq
from tyxonq.libs.quantum_library.kernels.statevector import (
    init_statevector,
    apply_1q_statevector,
    apply_2q_statevector,
    expect_z_statevector,
)
from tyxonq.libs.quantum_library.kernels.gates import (
    gate_rx,
    gate_ry,
    gate_cx_4x4,
)


# ==================== Configuration ====================

# Set PyTorch device
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("✓ CUDA GPU available and selected")
else:
    DEVICE = torch.device("cpu")
    print("! CUDA not available, using CPU")

# Quantum circuit parameters
N_QUBITS = 9  # Reduced from 28x28 pixels to 3x3
N_LAYERS = 2  # PQC depth
N_TRAIN_SAMPLES = 100  # Limited for demo (use full dataset in production)
BATCH_SIZE = 32
N_EPOCHS = 10
LEARNING_RATE = 1e-2


# ==================== Data Preparation ====================

def load_and_preprocess_mnist():
    """Load MNIST and preprocess to binary classification (1 vs 5)"""
    print("\n" + "=" * 60)
    print("Loading MNIST Dataset...")
    print("=" * 60)
    
    # Download MNIST
    train_dataset = torchvision.datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transforms.ToTensor()
    )
    
    # Convert to numpy for preprocessing
    x_train = train_dataset.data.numpy()
    y_train = train_dataset.targets.numpy()
    
    # Normalize to [0, 1]
    x_train = x_train[..., np.newaxis] / 255.0
    
    # Filter to binary classification: 1 vs 5
    def filter_classes(x, y, class_a, class_b):
        keep = (y == class_a) | (y == class_b)
        x, y = x[keep], y[keep]
        y = (y == class_a).astype(np.float32)  # 1 → label 1, 5 → label 0
        return x, y
    
    x_train, y_train = filter_classes(x_train, y_train, 1, 5)
    
    # Downsample images to 3x3 using bilinear interpolation
    x_train_tensor = torch.from_numpy(x_train).float()
    x_train_small = F.interpolate(
        x_train_tensor.permute(0, 3, 1, 2),  # [N, 1, 28, 28]
        size=(3, 3),
        mode='bilinear',
        align_corners=False
    ).permute(0, 2, 3, 1)  # [N, 3, 3, 1]
    
    # Binarize: threshold at 0.5
    x_train_bin = (x_train_small > 0.5).float().numpy()
    x_train_bin = np.squeeze(x_train_bin).reshape([-1, 9])  # Flatten to 9 features
    
    # Convert to PyTorch tensors and move to device
    x_train_torch = torch.tensor(x_train_bin[:N_TRAIN_SAMPLES], dtype=torch.float32).to(DEVICE)
    y_train_torch = torch.tensor(y_train[:N_TRAIN_SAMPLES], dtype=torch.float32).to(DEVICE)
    
    print(f"✓ Preprocessed {len(x_train_torch)} samples")
    print(f"  Input shape: {x_train_torch.shape}")
    print(f"  Label shape: {y_train_torch.shape}")
    print(f"  Class distribution: 1s={y_train_torch.sum().item():.0f}, 0s={(1-y_train_torch).sum().item():.0f}")
    
    return x_train_torch, y_train_torch


# ==================== Quantum Circuit Definition ====================

def quantum_circuit_forward(x, weights):
    """Parameterized quantum circuit for binary classification
    
    Uses direct statevector construction for gradient compatibility.
    
    Args:
        x: Input features (9 angles) - shape [9]
        weights: Trainable parameters - shape [2*nlayers, n_qubits]
    
    Returns:
        Expectation values for each qubit - shape [9]
    """
    K = tq.get_backend()
    
    # Initialize statevector |00...0>
    psi = init_statevector(N_QUBITS, backend=K)
    
    # Encoding layer: encode classical data as RX rotation angles
    for i in range(N_QUBITS):
        angle = x[i]
        psi = apply_1q_statevector(K, psi, gate_rx(angle), i, N_QUBITS)
    
    # Variational layers
    for j in range(N_LAYERS):
        # Entangling layer: CNOT ladder
        for i in range(N_QUBITS - 1):
            psi = apply_2q_statevector(K, psi, gate_cx_4x4(), i, i + 1, N_QUBITS)
        
        # Rotation layer
        for i in range(N_QUBITS):
            theta_x = weights[2 * j, i]
            theta_y = weights[2 * j + 1, i]
            psi = apply_1q_statevector(K, psi, gate_rx(theta_x), i, N_QUBITS)
            psi = apply_1q_statevector(K, psi, gate_ry(theta_y), i, N_QUBITS)
    
    # Measurement layer: Pauli-Z expectations
    expectations = []
    for i in range(N_QUBITS):
        exp_val = expect_z_statevector(psi, i, N_QUBITS, backend=K)
        # Convert to real value
        if hasattr(K, 'real'):
            exp_val = K.real(exp_val)
        expectations.append(exp_val)
    
    # Stack into tensor
    if hasattr(K, 'name') and K.name == 'pytorch':
        import torch
        return torch.stack([torch.as_tensor(e, dtype=torch.float32) for e in expectations])
    else:
        import numpy as np
        return np.array(expectations, dtype=np.float32)


# ==================== Hybrid Model Definition ====================

class HybridQuantumClassicalModel(nn.Module):
    """Hybrid model: Quantum PQC → Classical Linear → Sigmoid"""
    
    def __init__(self, n_qubits, n_layers):
        super().__init__()
        
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        
        # Trainable quantum circuit parameters
        self.quantum_weights = nn.Parameter(
            torch.randn(2 * n_layers, n_qubits, dtype=torch.float32) * 0.1
        )
        
        # Classical layers
        self.fc = nn.Linear(n_qubits, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """
        Args:
            x: Input batch - shape [batch_size, n_qubits]
        
        Returns:
            predictions: Binary probabilities - shape [batch_size, 1]
        """
        batch_size = x.shape[0]
        
        # Quantum layer: process each sample in the batch
        quantum_outputs = []
        for i in range(batch_size):
            output = quantum_circuit_forward(x[i], self.quantum_weights)
            quantum_outputs.append(output)
        
        quantum_output = torch.stack(quantum_outputs)  # [batch_size, n_qubits]
        
        # Classical post-processing
        logits = self.fc(quantum_output)  # [batch_size, 1]
        predictions = self.sigmoid(logits)  # [batch_size, 1]
        
        return predictions


# ==================== Training Loop ====================

def train_hybrid_model(x_train, y_train):
    """Train the hybrid quantum-classical model"""
    print("\n" + "=" * 60)
    print("Training Hybrid Model...")
    print("=" * 60)
    
    # Set TyxonQ backend to PyTorch
    tq.set_backend("pytorch")
    
    # Initialize model
    model = HybridQuantumClassicalModel(N_QUBITS, N_LAYERS)
    model = model.to(DEVICE)
    
    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training metrics
    epoch_times = []
    epoch_losses = []
    
    print(f"\nConfiguration:")
    print(f"  Device: {DEVICE}")
    print(f"  Qubits: {N_QUBITS}, Layers: {N_LAYERS}")
    print(f"  Batch size: {BATCH_SIZE}, Epochs: {N_EPOCHS}")
    print(f"  Learning rate: {LEARNING_RATE}")
    
    print(f"\n{'Epoch':<8} {'Loss':<12} {'Accuracy':<12} {'Time (s)':<10}")
    print("-" * 60)
    
    for epoch in range(N_EPOCHS):
        epoch_start = time.time()
        
        # Random batch sampling
        indices = np.random.randint(
            low=0, 
            high=min(N_TRAIN_SAMPLES, x_train.shape[0]), 
            size=BATCH_SIZE
        )
        inputs = x_train[indices]
        labels = y_train[indices]
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)
        
        # Compute loss
        loss = criterion(
            outputs.reshape(BATCH_SIZE, 1), 
            labels.reshape(BATCH_SIZE, 1)
        )
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Compute accuracy
        predictions = (outputs > 0.5).float()
        accuracy = (predictions.reshape(-1) == labels).float().mean()
        
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        epoch_losses.append(loss.item())
        
        if epoch % 5 == 0 or epoch == N_EPOCHS - 1:
            print(f"{epoch:<8} {loss.item():<12.6f} {accuracy.item():<12.4f} {epoch_time:<10.3f}")
    
    print("-" * 60)
    print(f"✓ Training completed!")
    print(f"  Final loss: {epoch_losses[-1]:.6f}")
    print(f"  Avg time/epoch: {np.mean(epoch_times[1:]):.3f}s (excluding first)")
    
    return model, epoch_losses, epoch_times


# ==================== Main Execution ====================

def main():
    """Main execution function"""
    print("\n" + "=" * 60)
    print("TyxonQ Hybrid Quantum-Classical Training Demo")
    print("=" * 60)
    
    # Load data
    x_train, y_train = load_and_preprocess_mnist()
    
    # Train model
    model, losses, times = train_hybrid_model(x_train, y_train)
    
    # Summary
    print("\n" + "=" * 60)
    print("Training Summary")
    print("=" * 60)
    print(f"Initial loss: {losses[0]:.6f}")
    print(f"Final loss: {losses[-1]:.6f}")
    print(f"Improvement: {(losses[0] - losses[-1]):.6f}")
    print(f"Average epoch time: {np.mean(times[1:]):.3f}s")
    
    print("\n" + "=" * 60)
    print("Key Takeaways")
    print("=" * 60)
    print("1. Seamless PyTorch integration: quantum circuits as nn.Module layers")
    print("2. GPU acceleration: both quantum simulation and classical layers")
    print("3. Automatic differentiation: end-to-end gradient flow")
    print("4. Practical workflow: demonstrates real quantum machine learning")
    print("5. Manual batching: explicit control over quantum circuit evaluation")
    print("\nThis hybrid paradigm enables practical quantum machine learning!")
    print("For production, consider implementing custom vmap or batch operations.")
    print("=" * 60)


if __name__ == "__main__":
    main()
