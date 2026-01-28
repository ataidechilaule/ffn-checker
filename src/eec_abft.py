"""
Extreme Error Correcting ABFT (EEC-ABFT)

Implements Algorithm-Based Fault Tolerance that can handle
INF, NaN, and near-INF errors using specialized detection
and correction strategies.
"""

import torch
import numpy as np
from typing import Tuple, Optional, List


class EEC_ABFT:
    """
    Extreme Error Correcting Algorithm-Based Fault Tolerance.
    
    Handles INF, NaN, and near-INF errors that corrupt traditional
    checksum-based error correction.
    """
    
    def __init__(
        self,
        threshold_near_inf: float = 1e10,
        threshold_correct: float = 1e5,
        epsilon: float = 1e-3
    ):
        """
        Initialize EEC-ABFT.
        
        Args:
            threshold_near_inf: Values above this considered near-INF
            threshold_correct: Threshold for correction strategy selection
            epsilon: Tolerance for error detection
        """
        self.threshold_near_inf = threshold_near_inf
        self.threshold_correct = threshold_correct
        self.epsilon = epsilon
        
    def compute_checksums(
        self,
        tensor: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute weighted and unweighted column checksums.
        
        Args:
            tensor: Input tensor [batch, seq, dim]
            
        Returns:
            (unweighted_checksum, weighted_checksum)
            Each has shape [batch, seq, 2] where dim 2 contains the two checksums
        """
        batch_size, seq_len, hidden_dim = tensor.shape
        
        # Unweighted checksum: v1 = [1, 1, 1, ..., 1]
        unweighted = tensor.sum(dim=2)  # [batch, seq]
        
        # Weighted checksum: v2 = [1, 2, 3, ..., n]
        weights = torch.arange(1, hidden_dim + 1, dtype=tensor.dtype, device=tensor.device)
        weighted = (tensor * weights.view(1, 1, -1)).sum(dim=2)  # [batch, seq]
        
        # Stack checksums
        checksums = torch.stack([unweighted, weighted], dim=2)  # [batch, seq, 2]
        
        return checksums
    
    def detect_errors(
        self,
        tensor: torch.Tensor,
        checksums: torch.Tensor
    ) -> torch.Tensor:
        """
        Detect errors by recomputing checksums and comparing.
        
        Args:
            tensor: Data tensor [batch, seq, dim]
            checksums: Stored checksums [batch, seq, 2]
            
        Returns:
            Boolean mask [batch, seq] indicating errors
        """
        # Recompute checksums
        recomputed = self.compute_checksums(tensor)
        
        # Compute difference (delta)
        delta = checksums - recomputed
        
        # Error detected if |delta_1| > epsilon
        errors = torch.abs(delta[:, :, 0]) > self.epsilon
        
        return errors, delta
    
    def correct_row(
        self,
        row: torch.Tensor,
        delta: torch.Tensor
    ) -> torch.Tensor:
        """
        Correct errors in a single row using EEC-ABFT.
        
        Args:
            row: Data row [hidden_dim]
            delta: Checksum difference [2] (unweighted, weighted)
            
        Returns:
            Corrected row
        """
        delta_1 = delta[0]  # Unweighted
        delta_2 = delta[1]  # Weighted
        
        # Case 1: delta_1 < INF (finite error)
        if torch.isfinite(delta_1):
            # Count near-INF values
            near_inf_mask = torch.abs(row) > self.threshold_near_inf
            num_near_inf = near_inf_mask.sum().item()
            
            if num_near_inf == 1:
                # Single error (0D pattern)
                
                if torch.isfinite(delta_2):
                    # Normal localization
                    error_idx = int(torch.round(delta_2 / delta_1).item())
                    error_idx = max(0, min(error_idx, len(row) - 1))
                else:
                    # delta_2 overflowed, search for largest value
                    error_idx = torch.argmax(torch.abs(row)).item()
                
                # Correction strategy based on magnitude
                if torch.abs(delta_1) > self.threshold_correct:
                    # Large error: reconstruct
                    row[error_idx] = self._reconstruct_value(row, delta_1, error_idx)
                else:
                    # Small error: direct correction
                    row[error_idx] = row[error_idx] - delta_1
                    
            elif num_near_inf > 1:
                # Multiple errors (propagated)
                # Correct each near-INF value
                for idx in near_inf_mask.nonzero(as_tuple=True)[0]:
                    row[idx] = self._reconstruct_value(row, delta_1, idx)
        
        # Case 2: delta_1 == INF
        elif torch.isinf(delta_1):
            # Find INF or near-INF value
            inf_mask = torch.isinf(row)
            near_inf_mask = torch.abs(row) > self.threshold_near_inf
            
            error_mask = inf_mask | near_inf_mask
            
            if error_mask.any():
                error_idx = error_mask.nonzero(as_tuple=True)[0][0].item()
                row[error_idx] = self._reconstruct_value(row, delta_1, error_idx)
        
        # Case 3: delta_1 == NaN
        elif torch.isnan(delta_1):
            # Find NaN or extreme values
            nan_mask = torch.isnan(row)
            inf_mask = torch.isinf(row)
            near_inf_mask = torch.abs(row) > self.threshold_near_inf
            
            error_mask = nan_mask | inf_mask | near_inf_mask
            
            if error_mask.any():
                error_idx = error_mask.nonzero(as_tuple=True)[0][0].item()
                row[error_idx] = self._reconstruct_value(row, delta_1, error_idx)
        
        return row
    
    def _reconstruct_value(
        self,
        row: torch.Tensor,
        delta_1: torch.Tensor,
        error_idx: int
    ) -> torch.Tensor:
        """
        Reconstruct corrupted value using surrounding values and checksum.
        
        Strategy: Use median of surrounding values as estimate
        (more robust than mean for extreme values)
        
        Args:
            row: Data row
            delta_1: Unweighted checksum difference
            error_idx: Index of corrupted value
            
        Returns:
            Reconstructed value
        """
        # Get non-corrupted values
        mask = torch.ones_like(row, dtype=torch.bool)
        mask[error_idx] = False
        
        # Filter out other extreme values
        valid_mask = mask & torch.isfinite(row) & (torch.abs(row) < self.threshold_near_inf)
        
        if valid_mask.any():
            # Use median of valid values
            valid_values = row[valid_mask]
            reconstructed = torch.median(valid_values)
        else:
            # Fallback: use 0
            reconstructed = torch.tensor(0.0, dtype=row.dtype, device=row.device)
        
        return reconstructed
    
    def correct_tensor(
        self,
        tensor: torch.Tensor,
        checksums: torch.Tensor
    ) -> Tuple[torch.Tensor, int]:
        """
        Detect and correct errors in entire tensor.
        
        Args:
            tensor: Data tensor [batch, seq, dim]
            checksums: Checksums [batch, seq, 2]
            
        Returns:
            (corrected_tensor, num_errors_corrected)
        """
        errors, delta = self.detect_errors(tensor, checksums)
        
        num_errors = 0
        
        # Correct each row with error
        for batch_idx in range(tensor.shape[0]):
            for seq_idx in range(tensor.shape[1]):
                if errors[batch_idx, seq_idx]:
                    tensor[batch_idx, seq_idx] = self.correct_row(
                        tensor[batch_idx, seq_idx],
                        delta[batch_idx, seq_idx]
                    )
                    num_errors += 1
        
        return tensor, num_errors


# Example usage
if __name__ == "__main__":
    # Create EEC-ABFT instance
    abft = EEC_ABFT()
    
    # Create sample data
    batch_size, seq_len, hidden_dim = 2, 4, 768
    data = torch.randn(batch_size, seq_len, hidden_dim)
    
    print("Original data:")
    print(f"  Shape: {data.shape}")
    print(f"  Mean: {data.mean():.4f}")
    print(f"  Std: {data.std():.4f}")
    
    # Compute checksums
    checksums = abft.compute_checksums(data)
    print(f"\nChecksums shape: {checksums.shape}")
    
    # Inject error
    data[0, 0, 10] = float('inf')
    print(f"\nInjected INF at [0, 0, 10]")
    print(f"Has INF: {torch.isinf(data).any()}")
    
    # Detect errors
    errors, delta = abft.detect_errors(data, checksums)
    print(f"\nErrors detected: {errors.sum().item()}")
    print(f"Error at [0, 0]: {errors[0, 0]}")
    
    # Correct errors
    data, num_corrected = abft.correct_tensor(data, checksums)
    print(f"\nCorrected {num_corrected} errors")
    print(f"Has INF after correction: {torch.isinf(data).any()}")
    print(f"Value at [0, 0, 10]: {data[0, 0, 10]:.4f}")
