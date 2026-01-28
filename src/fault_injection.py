"""
Fault Injection Framework for LLM Training

Injects transient faults (INF, NaN, near-INF) into tensor computations
to simulate soft errors in hardware.
"""

import torch
import numpy as np
from typing import Tuple, List, Dict, Optional
from enum import Enum


class ErrorType(Enum):
    """Types of errors to inject"""
    INF = "INF"
    NaN = "NaN"
    NEAR_INF = "near-INF"


class InjectionLocation(Enum):
    """Locations in FFN where faults can be injected"""
    FIRST_GEMM = "first_gemm"  # After X @ W1^T
    AFTER_ACTIVATION = "after_activation"  # After GELU(X @ W1^T + b1)
    SECOND_GEMM = "second_gemm"  # After H @ W2^T
    FINAL_OUTPUT = "final_output"  # After H @ W2^T + b2


class FaultInjector:
    """
    Inject transient faults into tensor computations.
    
    Usage:
        injector = FaultInjector(error_types=['INF', 'NaN'])
        tensor = injector.inject(tensor, location=(0, 5, 10), error_type='INF')
    """
    
    def __init__(
        self, 
        error_types: Optional[List[str]] = None,
        seed: int = 42
    ):
        """
        Initialize fault injector.
        
        Args:
            error_types: List of error types to inject. 
                        If None, uses ['INF', 'NaN', 'near-INF']
            seed: Random seed for reproducibility
        """
        if error_types is None:
            error_types = ['INF', 'NaN', 'near-INF']
        
        self.error_types = error_types
        self.fault_log = []
        self.rng = np.random.RandomState(seed)
        
    def inject(
        self, 
        tensor: torch.Tensor, 
        location: Tuple[int, ...], 
        error_type: str
    ) -> torch.Tensor:
        """
        Inject a single fault into tensor at specified location.
        
        Args:
            tensor: PyTorch tensor to corrupt
            location: Tuple of indices (e.g., (batch, seq, dim))
            error_type: One of 'INF', 'NaN', 'near-INF'
            
        Returns:
            Corrupted tensor (modified in-place)
        """
        # Record original value
        original_value = tensor[location].item()
        
        # Inject appropriate error type
        if error_type == 'INF':
            # Random sign
            sign = self.rng.choice([-1, 1])
            tensor[location] = sign * float('inf')
            
        elif error_type == 'NaN':
            tensor[location] = float('nan')
            
        elif error_type == 'near-INF':
            # Flip MSB of exponent to create near-overflow value
            tensor[location] = self._flip_exponent_msb(original_value)
            
        else:
            raise ValueError(f"Unknown error type: {error_type}")
        
        # Log fault for analysis
        self.fault_log.append({
            'location': location,
            'error_type': error_type,
            'original_value': original_value,
            'corrupted_value': tensor[location].item(),
        })
        
        return tensor
    
    def inject_random(
        self,
        tensor: torch.Tensor,
        error_type: Optional[str] = None
    ) -> Tuple[torch.Tensor, Tuple[int, ...]]:
        """
        Inject fault at random location in tensor.
        
        Args:
            tensor: Tensor to corrupt
            error_type: Error type to inject. If None, chooses randomly.
            
        Returns:
            (corrupted_tensor, location)
        """
        # Choose random location
        location = tuple(self.rng.randint(0, s) for s in tensor.shape)
        
        # Choose random error type if not specified
        if error_type is None:
            error_type = self.rng.choice(self.error_types)
        
        # Inject fault
        tensor = self.inject(tensor, location, error_type)
        
        return tensor, location
    
    def inject_multiple(
        self,
        tensor: torch.Tensor,
        num_faults: int,
        error_type: Optional[str] = None
    ) -> List[Tuple[int, ...]]:
        """
        Inject multiple faults into tensor.
        
        Args:
            tensor: Tensor to corrupt
            num_faults: Number of faults to inject
            error_type: Error type to inject. If None, random for each fault.
            
        Returns:
            List of fault locations
        """
        locations = []
        
        for _ in range(num_faults):
            tensor, location = self.inject_random(tensor, error_type)
            locations.append(location)
        
        return locations
    
    def _flip_exponent_msb(self, value: float) -> float:
        """
        Flip most significant bit of exponent to create near-INF value.
        
        For float32:
        - Bit 30 is MSB of exponent
        - Flipping it typically creates value near 10^30 or larger
        
        Args:
            value: Original float value
            
        Returns:
            Value with flipped exponent MSB
        """
        # Convert to 32-bit representation
        value_np = np.float32(value)
        bits = value_np.view(np.uint32)
        
        # Flip bit 30 (MSB of exponent for float32)
        # Exponent bits are 30-23 (8 bits)
        bits ^= (1 << 30)
        
        # Convert back to float
        result = bits.view(np.float32)
        
        return float(result)
    
    def get_fault_log(self) -> List[Dict]:
        """Return log of all injected faults"""
        return self.fault_log
    
    def clear_log(self):
        """Clear fault log"""
        self.fault_log = []
    
    def get_statistics(self) -> Dict:
        """
        Get statistics about injected faults.
        
        Returns:
            Dictionary with fault statistics
        """
        if not self.fault_log:
            return {}
        
        total = len(self.fault_log)
        by_type = {}
        
        for fault in self.fault_log:
            error_type = fault['error_type']
            by_type[error_type] = by_type.get(error_type, 0) + 1
        
        return {
            'total_faults': total,
            'by_type': by_type,
            'percentages': {k: v/total*100 for k, v in by_type.items()}
        }


def inject_into_ffn_forward(
    injector: FaultInjector,
    location: InjectionLocation,
    tensor: torch.Tensor,
    error_type: str
) -> torch.Tensor:
    """
    Helper function to inject fault during FFN forward pass.
    
    Args:
        injector: FaultInjector instance
        location: Where in FFN to inject (first_gemm, etc.)
        tensor: Tensor at that location
        error_type: Type of error to inject
        
    Returns:
        Corrupted tensor
    """
    # Inject at random position
    tensor, pos = injector.inject_random(tensor, error_type)
    
    return tensor


def is_training_failed(loss: torch.Tensor) -> bool:
    """
    Check if training has failed due to extreme values.
    
    Args:
        loss: Training loss tensor
        
    Returns:
        True if loss is NaN or INF (non-trainable state)
    """
    return torch.isnan(loss).any() or torch.isinf(loss).any()


# Example usage
if __name__ == "__main__":
    # Create injector
    injector = FaultInjector()
    
    # Create sample tensor
    tensor = torch.randn(2, 10, 768)  # (batch, seq, hidden_dim)
    
    print("Original tensor stats:")
    print(f"  Mean: {tensor.mean():.4f}")
    print(f"  Std: {tensor.std():.4f}")
    print(f"  Min: {tensor.min():.4f}")
    print(f"  Max: {tensor.max():.4f}")
    
    # Inject INF error
    print("\nInjecting INF error...")
    tensor, location = injector.inject_random(tensor, 'INF')
    print(f"  Injected at: {location}")
    print(f"  Has INF: {torch.isinf(tensor).any()}")
    
    # Get statistics
    stats = injector.get_statistics()
    print(f"\nFault statistics: {stats}")
