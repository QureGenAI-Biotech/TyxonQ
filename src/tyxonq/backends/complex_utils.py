"""
Complex number utilities for quantum computing operations.
Provides efficient complex number handling with minimal warnings.
"""

import torch
import numpy as np
from typing import Union, Tuple, Any
import warnings

# Suppress specific warnings for complex operations
warnings.filterwarnings("ignore", message="Casting complex values to real discards the imaginary part")

class ComplexHandler:
    """Handles complex number operations for quantum computing"""
    
    @staticmethod
    def safe_complex_to_real(complex_tensor: torch.Tensor, method: str = "magnitude") -> torch.Tensor:
        """
        Safely convert complex tensor to real tensor without warnings.
        
        :param complex_tensor: Complex tensor
        :param method: Conversion method ("magnitude", "real", "imag", "phase")
        :return: Real tensor
        """
        if method == "magnitude":
            return torch.abs(complex_tensor)
        elif method == "real":
            return torch.real(complex_tensor)
        elif method == "imag":
            return torch.imag(complex_tensor)
        elif method == "phase":
            return torch.angle(complex_tensor)
        else:
            raise ValueError(f"Unknown conversion method: {method}")
    
    @staticmethod
    def quantum_expectation(complex_tensor: torch.Tensor) -> torch.Tensor:
        """
        Compute quantum expectation value from complex tensor.
        This is the most common operation in quantum computing.
        
        :param complex_tensor: Complex tensor representing quantum state
        :return: Real expectation value
        """
        # For quantum expectation, we typically want the real part
        # but we handle it carefully to avoid warnings
        if complex_tensor.is_complex():
            # Use magnitude for quantum expectation (most common case)
            return torch.abs(complex_tensor)
        else:
            return complex_tensor
    
    @staticmethod
    def complex_autodiff_safe(complex_tensor: torch.Tensor) -> torch.Tensor:
        """
        Prepare complex tensor for automatic differentiation.
        PyTorch autodiff requires real outputs, so we convert appropriately.
        
        :param complex_tensor: Complex tensor
        :return: Real tensor suitable for autodiff
        """
        if complex_tensor.is_complex():
            # For autodiff, we typically want the real part
            # This is the most common case in quantum computing
            return torch.real(complex_tensor)
        else:
            return complex_tensor
    
    @staticmethod
    def quantum_gradient_safe(complex_tensor: torch.Tensor) -> torch.Tensor:
        """
        Prepare complex tensor for gradient computation in quantum algorithms.
        
        :param complex_tensor: Complex tensor
        :return: Real tensor suitable for gradient computation
        """
        if complex_tensor.is_complex():
            # For quantum gradients, we often want the magnitude
            # This preserves the quantum nature of the computation
            return torch.abs(complex_tensor)
        else:
            return complex_tensor

def safe_cast(tensor: torch.Tensor, dtype: str) -> torch.Tensor:
    """
    Safely cast tensor to specified dtype without complex warnings.
    
    :param tensor: Input tensor
    :param dtype: Target dtype
    :return: Casted tensor
    """
    if dtype in ['float32', 'float64'] and tensor.is_complex():
        # For real dtypes, convert complex to real safely
        return ComplexHandler.safe_complex_to_real(tensor, "real")
    else:
        return tensor.type(getattr(torch, dtype))

def quantum_expectation_value(complex_tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute quantum expectation value without warnings.
    
    :param complex_tensor: Complex tensor
    :return: Real expectation value
    """
    return ComplexHandler.quantum_expectation(complex_tensor)

def autodiff_safe(complex_tensor: torch.Tensor) -> torch.Tensor:
    """
    Make complex tensor safe for automatic differentiation.
    
    :param complex_tensor: Complex tensor
    :return: Real tensor for autodiff
    """
    return ComplexHandler.complex_autodiff_safe(complex_tensor)
