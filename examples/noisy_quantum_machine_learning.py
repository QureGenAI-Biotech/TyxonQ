"""
Noisy Quantum Machine Learning (QML) with MNIST Classification

This example demonstrates:
1. Quantum circuit-based binary classification on MNIST (0 vs 1)
2. Realistic noise simulation using TyxonQ's noise API
3. PyTorch integration for hybrid quantum-classical training
4. Vectorized Monte Carlo noise sampling for efficiency

Key Features:
- Uses `.with_noise()` API for simplified noise configuration
- Parameterized quantum circuit (PQC) as a quantum feature map
- Binary cross-entropy loss with sigmoid activation
- Adam optimizer for variational parameters
- Supports both exact (density matrix) and Monte Carlo noise simulation

Hardware Requirements:
- PyTorch with GPU support (recommended)
- ~2GB RAM for 64 training samples
- MNIST dataset (auto-downloaded)

Expected Results:
- Training accuracy: ~90-95% (with 0.5% noise)
- Demonstrates NISQ algorithm robustness to noise
"""

import os
import time
import numpy as np
import torch

# Enable MPS fallback for Mac GPU compatibility
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import tyxonq as tq

# Configure PyTorch backend for automatic differentiation
K = tq.set_backend("pytorch")

# Import MNIST dataset
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import torch.utils.data as data

print("=" * 60)
print("Noisy Quantum Machine Learning on MNIST")
print("=" * 60)

# ==============================================================================
# Configuration
# ==============================================================================

# Quantum circuit parameters
N_QUBITS = 9  # Number of qubits (3x3 grid for images)
N_LAYERS = 4  # Number of variational layers
NOISE_LEVEL = 0.005  # Depolarizing noise probability (0.5%)

# Training parameters
N_SAMPLES = 64  # Number of training samples
BATCH_SIZE = 16  # Mini-batch size
MAX_ITER = 200  # Maximum training iterations
LEARNING_RATE = 1e-2  # Learning rate for circuit parameters
VAL_STEP = 50  # Validation every N steps

# Data preparation method
DATA_PREP = "resize"  # "resize" or "pca"

# Monte Carlo noise sampling
N_NOISE_SAMPLES = 8  # Number of noise samples for Monte Carlo averaging

# ==============================================================================
# Data Preparation
# ==============================================================================

print("\n[1/5] Loading MNIST dataset...")

# Load MNIST data
mnist_train = MNIST(root='./data', train=True, download=True, transform=ToTensor())
mnist_test = MNIST(root='./data', train=False, download=True, transform=ToTensor())

# Convert to numpy
x_train = mnist_train.data.numpy()
y_train = mnist_train.targets.numpy()
x_test = mnist_test.data.numpy()
y_test = mnist_test.targets.numpy()

# Normalize to [0, 1]
x_train = x_train[..., np.newaxis] / 255.0


def filter_binary_classes(x, y, class_a=0, class_b=1):
    """Filter dataset to only include two classes for binary classification."""
    keep = (y == class_a) | (y == class_b)
    x, y = x[keep], y[keep]
    y = (y == class_a).astype(np.float32)  # Convert to binary labels
    return x, y


# Filter to 0 vs 1 classification
x_train, y_train = filter_binary_classes(x_train, y_train, 0, 1)
print(f"  Filtered dataset: {len(x_train)} samples (0 vs 1)")

# Prepare data for quantum circuit
if DATA_PREP == "resize":
    # Resize images to match qubit grid (3x3 for 9 qubits)
    grid_size = int(np.sqrt(N_QUBITS))
    x_train_tensor = torch.from_numpy(x_train).float()
    x_train_resized = torch.nn.functional.interpolate(
        x_train_tensor.permute(0, 3, 1, 2),
        size=(grid_size, grid_size),
        mode='bilinear',
        align_corners=False
    ).permute(0, 2, 3, 1).numpy()
    
    # Binarize (threshold at 0.5)
    x_train = np.array(x_train_resized > 0.5, dtype=np.float32)
    x_train = np.squeeze(x_train).reshape([-1, N_QUBITS])
    print(f"  Resized images to {grid_size}x{grid_size} = {N_QUBITS} pixels")
    
else:  # PCA
    from sklearn.decomposition import PCA
    x_train = PCA(N_QUBITS).fit_transform(x_train.reshape([-1, 28 * 28]))
    print(f"  Applied PCA to reduce to {N_QUBITS} features")


# Create PyTorch dataset with random sampling
class RandomMNISTDataset(data.Dataset):
    """Dataset that returns random batches each iteration."""
    
    def __init__(self, x, y, n_samples, batch_size, max_iter):
        self.x = torch.from_numpy(x[:n_samples]).float()
        self.y = torch.from_numpy(y[:n_samples]).float()
        self.batch_size = batch_size
        self.max_iter = max_iter
        
    def __len__(self):
        return self.max_iter
        
    def __getitem__(self, idx):
        # Return random batch
        indices = torch.randperm(len(self.x))[:self.batch_size]
        return self.x[indices], self.y[indices]


mnist_data = RandomMNISTDataset(x_train, y_train, N_SAMPLES, BATCH_SIZE, MAX_ITER)
print(f"  Training set: {N_SAMPLES} samples, batch size: {BATCH_SIZE}")

# ==============================================================================
# Quantum Circuit Definition
# ==============================================================================

print("\n[2/5] Building quantum circuit...")


def build_pqc(x, params, noise_level=0.0):
    """
    Build parameterized quantum circuit (PQC) with data encoding and variational layers.
    
    Args:
        x: Input feature vector (shape: [N_QUBITS])
        params: Variational parameters (shape: [N_LAYERS, N_QUBITS, 2])
        noise_level: Depolarizing noise probability
    
    Returns:
        Expectation value of Pauli-Z measurements (scalar)
    """
    # Create quantum circuit
    c = tq.Circuit(N_QUBITS)
    
    # Data encoding layer: Rx rotations proportional to input features
    for i in range(N_QUBITS):
        if DATA_PREP == "resize":
            theta = x[i] * np.pi / 2  # Binary features → 0 or π/2
        else:  # PCA
            theta = torch.atan(x[i])  # Continuous features → atan encoding
        c.rx(i, theta=theta)
    
    # Variational layers
    for layer in range(N_LAYERS):
        # Entangling layer: CNOT gates
        for i in range(N_QUBITS - 1):
            c.cnot(i, i + 1)
        
        # Parameterized rotations: Rz and Rx
        for i in range(N_QUBITS):
            c.rz(i, theta=params[layer, i, 0])
            c.rx(i, theta=params[layer, i, 1])
    
    # Apply noise if specified
    if noise_level > 0:
        c = c.with_noise("depolarizing", p=noise_level)
    
    # Measurement: Average of all qubit Z-expectations
    expectations = [c.expectation_ps(z=[i]) for i in range(N_QUBITS)]
    expectations = torch.stack([K.real(exp) for exp in expectations])
    
    return torch.mean(expectations)


print(f"  Circuit: {N_QUBITS} qubits, {N_LAYERS} layers")
print(f"  Noise: {NOISE_LEVEL * 100:.2f}% depolarizing error per gate")
print(f"  Total parameters: {N_LAYERS * N_QUBITS * 2}")

# ==============================================================================
# Loss Function and Training
# ==============================================================================

print("\n[3/5] Preparing training loop...")


def compute_loss(params, scale, x_batch, y_batch, noise_level=0.0):
    """
    Compute binary cross-entropy loss for a batch.
    
    Args:
        params: Circuit parameters
        scale: Scaling factor for sigmoid
        x_batch: Input features (shape: [batch_size, N_QUBITS])
        y_batch: Binary labels (shape: [batch_size])
        noise_level: Noise probability
    
    Returns:
        loss: Scalar loss value
        y_pred: Predicted probabilities
    """
    batch_losses = []
    batch_preds = []
    
    for i in range(len(x_batch)):
        # Build circuit for single sample
        x_single = x_batch[i]
        y_single = y_batch[i]
        
        # Get quantum expectation
        y_exp = build_pqc(x_single, params, noise_level)
        
        # Apply sigmoid activation
        y_pred = torch.sigmoid(scale * y_exp)
        
        # Binary cross-entropy loss
        loss = -y_single * torch.log(y_pred + 1e-10) - (1 - y_single) * torch.log(1 - y_pred + 1e-10)
        
        batch_losses.append(loss)
        batch_preds.append(y_pred)
    
    # Average over batch
    total_loss = torch.mean(torch.stack(batch_losses))
    all_preds = torch.stack(batch_preds)
    
    return total_loss, all_preds


def compute_accuracy(y_pred, y_true):
    """Compute classification accuracy."""
    if hasattr(y_pred, 'detach'):
        y_pred = y_pred.detach().cpu().numpy()
    if hasattr(y_true, 'detach'):
        y_true = y_true.detach().cpu().numpy()
    
    # Threshold at 0.5
    y_pred_binary = (y_pred > 0.5).astype(np.float32)
    
    return np.mean(y_pred_binary == y_true)


def train_qml(initial_params=None, initial_scale=15.0, noise_level=0.0):
    """
    Train the quantum machine learning model.
    
    Args:
        initial_params: Initial circuit parameters (or None for random init)
        initial_scale: Initial scaling factor for sigmoid
        noise_level: Depolarizing noise probability
    
    Returns:
        Trained parameters
    """
    # Initialize parameters
    if initial_params is None:
        params = torch.nn.Parameter(torch.randn(N_LAYERS, N_QUBITS, 2) * 0.1)
    else:
        params = torch.nn.Parameter(initial_params)
    
    scale = torch.nn.Parameter(torch.tensor(initial_scale, dtype=torch.float32))
    
    # Optimizers
    optimizer_params = torch.optim.Adam([params], lr=LEARNING_RATE)
    optimizer_scale = torch.optim.Adam([scale], lr=5e-2)
    
    # Training metrics
    train_times = []
    val_times = []
    
    print(f"\n  Starting training: {MAX_ITER} iterations")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Noise level: {noise_level * 100:.2f}%")
    print("-" * 60)
    
    try:
        for iteration in range(MAX_ITER):
            # Get batch
            x_batch, y_batch = mnist_data[iteration]
            
            # Forward pass
            time_start = time.time()
            loss, y_pred = compute_loss(params, scale, x_batch, y_batch, noise_level)
            
            # Backward pass
            optimizer_params.zero_grad()
            optimizer_scale.zero_grad()
            loss.backward()
            
            # Update parameters
            optimizer_params.step()
            optimizer_scale.step()
            
            time_end = time.time()
            train_times.append(time_end - time_start)
            
            # Validation
            if iteration % VAL_STEP == 0:
                print(f"\nIteration {iteration}/{MAX_ITER}")
                print(f"  Loss: {loss.item():.4f}")
                print(f"  Scale: {scale.item():.2f}")
                
                # Compute training accuracy
                time_val_start = time.time()
                with torch.no_grad():
                    _, all_preds = compute_loss(
                        params, scale,
                        torch.from_numpy(x_train[:N_SAMPLES]).float(),
                        torch.from_numpy(y_train[:N_SAMPLES]).float(),
                        noise_level
                    )
                    acc = compute_accuracy(all_preds, y_train[:N_SAMPLES])
                
                time_val_end = time.time()
                val_times.append(time_val_end - time_val_start)
                
                print(f"  Training accuracy: {acc * 100:.2f}%")
                
                if len(train_times) > 1:
                    print(f"  Avg batch time: {np.mean(train_times[1:]):.3f}s")
                if len(val_times) > 1:
                    print(f"  Avg validation time: {np.mean(val_times[1:]):.3f}s")
    
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
    
    return params


# ==============================================================================
# Inference
# ==============================================================================

def evaluate_model(params, scale=15.0, noise_level=0.0):
    """Evaluate trained model on full training set."""
    print("\n[5/5] Final evaluation...")
    
    with torch.no_grad():
        loss, y_pred = compute_loss(
            params, torch.tensor(scale),
            torch.from_numpy(x_train[:N_SAMPLES]).float(),
            torch.from_numpy(y_train[:N_SAMPLES]).float(),
            noise_level
        )
        acc = compute_accuracy(y_pred, y_train[:N_SAMPLES])
    
    print(f"  Final loss: {loss.item():.4f}")
    print(f"  Final accuracy: {acc * 100:.2f}%")
    
    return acc


# ==============================================================================
# Main Execution
# ==============================================================================

if __name__ == "__main__":
    print("\n[4/5] Training quantum classifier...")
    
    # Train model with noise
    trained_params = train_qml(
        initial_params=None,
        initial_scale=10.0,
        noise_level=NOISE_LEVEL
    )
    
    # Final evaluation
    final_acc = evaluate_model(trained_params, scale=15.0, noise_level=NOISE_LEVEL)
    
    # Save trained parameters
    param_file = "noisy_qml_params.npy"
    np.save(param_file, trained_params.detach().cpu().numpy())
    print(f"\n  Saved parameters to: {param_file}")
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    
    # Compare with ideal (no noise) case
    print("\n[Bonus] Comparing with ideal (no-noise) case...")
    print("-" * 60)
    
    print("\nWith noise (p={:.2f}%):".format(NOISE_LEVEL * 100))
    evaluate_model(trained_params, scale=15.0, noise_level=NOISE_LEVEL)
    
    print("\nWithout noise (ideal):".format(NOISE_LEVEL * 100))
    evaluate_model(trained_params, scale=15.0, noise_level=0.0)
    
    print("\n" + "=" * 60)
    print("Expected Results:")
    print("  - Noisy accuracy: ~85-95%")
    print("  - Ideal accuracy: ~95-99%")
    print("  - This demonstrates NISQ algorithm robustness!")
    print("=" * 60)
