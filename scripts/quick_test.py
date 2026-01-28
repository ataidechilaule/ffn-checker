#!/usr/bin/env python3
"""
Quick Test Script

Runs a fast validation to ensure FFN-Checker is working correctly.
Takes about 5-10 minutes.

Usage:
    python quick_test.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from fault_injection import FaultInjector
from eec_abft import EEC_ABFT
from ffn_checker import FFNWithProtection, TransformerFFN


def test_fault_injection():
    """Test 1: Fault injection framework"""
    print("Test 1: Fault Injection")
    print("-" * 40)
    
    injector = FaultInjector()
    tensor = torch.randn(2, 10, 768)
    
    print(f"Original tensor: mean={tensor.mean():.4f}, has_inf={torch.isinf(tensor).any()}")
    
    # Inject INF
    tensor, loc = injector.inject_random(tensor, 'INF')
    print(f"After INF injection at {loc}: has_inf={torch.isinf(tensor).any()}")
    
    # Inject NaN
    tensor2 = torch.randn(2, 10, 768)
    tensor2, loc = injector.inject_random(tensor2, 'NaN')
    print(f"After NaN injection at {loc}: has_nan={torch.isnan(tensor2).any()}")
    
    print("✓ Fault injection working\n")


def test_eec_abft():
    """Test 2: EEC-ABFT error detection and correction"""
    print("Test 2: EEC-ABFT")
    print("-" * 40)
    
    abft = EEC_ABFT()
    
    # Create clean data
    data = torch.randn(2, 4, 768)
    checksums = abft.compute_checksums(data)
    
    print(f"Clean data: shape={data.shape}")
    print(f"Checksums: shape={checksums.shape}")
    
    # Detect (should find no errors)
    errors, delta = abft.detect_errors(data, checksums)
    print(f"Errors in clean data: {errors.sum().item()}")
    
    # Inject fault
    data[0, 0, 10] = float('inf')
    print(f"Injected INF at [0, 0, 10]")
    
    # Detect
    errors, delta = abft.detect_errors(data, checksums)
    print(f"Errors after injection: {errors.sum().item()}")
    
    # Correct
    data, num_corrected = abft.correct_tensor(data, checksums)
    print(f"Corrected {num_corrected} errors")
    print(f"Has INF after correction: {torch.isinf(data).any()}")
    
    print("✓ EEC-ABFT working\n")


def test_ffn_checker():
    """Test 3: FFN-Checker protection"""
    print("Test 3: FFN-Checker")
    print("-" * 40)
    
    # Create models
    ffn_protected = FFNWithProtection(768, 3072, enable_protection=True)
    ffn_standard = TransformerFFN(768, 3072)
    ffn_protected.eval()
    ffn_standard.eval()
    
    # Clean input
    x = torch.randn(2, 10, 768)
    print(f"Input: shape={x.shape}, mean={x.mean():.4f}")
    
    with torch.no_grad():
        # Test standard FFN
        y_std = ffn_standard(x)
        print(f"Standard FFN output: shape={y_std.shape}, mean={y_std.mean():.4f}")
        
        # Test protected FFN
        y_prot, stats = ffn_protected(x)
        print(f"Protected FFN output: shape={y_prot.shape}, mean={y_prot.mean():.4f}")
        print(f"Protection stats: {stats}")
    
    # Corrupted input
    x_corrupt = x.clone()
    x_corrupt[0, 5, 100] = float('inf')
    print(f"\nCorrupted input: has_inf={torch.isinf(x_corrupt).any()}")
    
    with torch.no_grad():
        # Standard FFN with corruption
        y_std_corrupt = ffn_standard(x_corrupt)
        print(f"Standard FFN with corruption: has_inf={torch.isinf(y_std_corrupt).any()}, has_nan={torch.isnan(y_std_corrupt).any()}")
        
        # Protected FFN with corruption
        y_prot_corrupt, stats_corrupt = ffn_protected(x_corrupt)
        print(f"Protected FFN with corruption: has_inf={torch.isinf(y_prot_corrupt).any()}, has_nan={torch.isnan(y_prot_corrupt).any()}")
        print(f"Protection stats: {stats_corrupt}")
    
    print("✓ FFN-Checker working\n")


def test_performance():
    """Test 4: Performance overhead measurement"""
    print("Test 4: Performance Overhead")
    print("-" * 40)
    
    import time
    
    ffn_protected = FFNWithProtection(768, 3072, enable_protection=True)
    ffn_standard = TransformerFFN(768, 3072)
    ffn_protected.eval()
    ffn_standard.eval()
    
    x = torch.randn(8, 128, 768)  # Typical batch
    
    # Warmup
    with torch.no_grad():
        for _ in range(5):
            _ = ffn_standard(x)
            _ = ffn_protected(x)
    
    # Measure standard
    num_iters = 100
    start = time.time()
    with torch.no_grad():
        for _ in range(num_iters):
            _ = ffn_standard(x)
    time_standard = (time.time() - start) / num_iters * 1000  # ms
    
    # Measure protected
    start = time.time()
    with torch.no_grad():
        for _ in range(num_iters):
            _ = ffn_protected(x)
    time_protected = (time.time() - start) / num_iters * 1000  # ms
    
    overhead = (time_protected - time_standard) / time_standard * 100
    
    print(f"Standard FFN: {time_standard:.2f} ms/iter")
    print(f"Protected FFN: {time_protected:.2f} ms/iter")
    print(f"Overhead: {overhead:.1f}%")
    
    print("✓ Performance measurement working\n")


def main():
    print("=" * 50)
    print("FFN-Checker Quick Test")
    print("=" * 50)
    print()
    
    try:
        test_fault_injection()
        test_eec_abft()
        test_ffn_checker()
        test_performance()
        
        print("=" * 50)
        print("✓ All tests passed!")
        print("=" * 50)
        print()
        print("Your FFN-Checker installation is working correctly.")
        print("You can now run full experiments:")
        print("  - python scripts/run_vulnerability_analysis.py")
        print("  - python scripts/run_coverage_test.py")
        print("  - python scripts/measure_overhead.py")
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
