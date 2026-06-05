#!/usr/bin/env python3
"""
ACTUAL REAL TEST - Comparing ORIGINAL vs CURRENT implementation
This script tests ONLY the Triton kernel in isolation to see if my changes helped or hurt
"""
import torch
import time
import sys
import importlib

print("="*80)
print("HONEST PERFORMANCE TEST - ORIGINAL vs CURRENT")
print("="*80)

if not torch.cuda.is_available():
    print("ERROR: Need CUDA GPU for testing")
    sys.exit(1)

device = torch.device("cuda")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.version.cuda}")
print()

# Test setup - realistic dimensions for TinyLlama
batch_size = 8
seq_len = 128
hidden_dim = 2048  # TinyLlama hidden dimension

print(f"Test configuration:")
print(f"  Batch size: {batch_size}")
print(f"  Sequence length: {seq_len}")
print(f"  Hidden dim: {hidden_dim}")
print()

# Create test data
x = torch.randn(batch_size, seq_len, hidden_dim, dtype=torch.float16, device=device)
residual = torch.randn_like(x)
weight = torch.ones(hidden_dim, dtype=torch.float16, device=device)

print("-"*80)
print("TEST 1: ORIGINAL Kernel (Your 1.68x speedup version)")
print("-"*80)

try:
    import kernels_ORIGINAL
    
    # Warmup - let Triton compile
    print("Warming up ORIGINAL kernel (Triton compilation)...")
    for _ in range(20):
        _ = kernels_ORIGINAL.fast_fused_norm(x, residual, weight)
    torch.cuda.synchronize()
    
    # Benchmark
    print("Benchmarking ORIGINAL kernel (500 iterations)...")
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(500):
        y_orig = kernels_ORIGINAL.fast_fused_norm(x, residual, weight)
    end_event.record()
    torch.cuda.synchronize()
    
    original_time = start_event.elapsed_time(end_event) / 500  # ms per call
    print(f"✓ ORIGINAL: {original_time:.6f} ms per call")
    
except Exception as e:
    print(f"✗ Failed to test ORIGINAL: {e}")
    import traceback
    traceback.print_exc()
    original_time = None
    y_orig = None

print()
print("-"*80)
print("TEST 2: CURRENT Kernel (My 'optimized' version)")
print("-"*80)

try:
    import kernels as kernels_current
    
    # Warmup
    print("Warming up CURRENT kernel (Triton compilation)...")
    for _ in range(20):
        _ = kernels_current.fast_fused_norm(x, residual, weight)
    torch.cuda.synchronize()
    
    # Benchmark
    print("Benchmarking CURRENT kernel (500 iterations)...")
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(500):
        y_curr = kernels_current.fast_fused_norm(x, residual, weight)
    end_event.record()
    torch.cuda.synchronize()
    
    current_time = start_event.elapsed_time(end_event) / 500  # ms per call
    print(f"✓ CURRENT: {current_time:.6f} ms per call")
    
except Exception as e:
    print(f"✗ Failed to test CURRENT: {e}")
    import traceback
    traceback.print_exc()
    current_time = None
    y_curr = None

print()
print("="*80)
print("KERNEL COMPARISON RESULTS")
print("="*80)

if original_time and current_time:
    if y_orig is not None and y_curr is not None:
        diff = (y_orig - y_curr).abs().max().item()
        print(f"Numerical accuracy: {diff:.2e} max difference")
        print()
    
    print(f"ORIGINAL kernel: {original_time:.6f} ms")
    print(f"CURRENT kernel:  {current_time:.6f} ms")
    print()
    
    if current_time < original_time:
        speedup = original_time / current_time
        improvement = ((speedup - 1) * 100)
        print(f"✓ CURRENT is FASTER: {speedup:.3f}x speedup ({improvement:+.1f}%)")
        print(f"  Saved {(original_time - current_time)*1000:.2f} microseconds per call")
    elif current_time > original_time:
        slowdown = current_time / original_time
        regression = ((slowdown - 1) * 100)
        print(f"✗ CURRENT is SLOWER: {slowdown:.3f}x slowdown ({regression:+.1f}%)")
        print(f"  Lost {(current_time - original_time)*1000:.2f} microseconds per call")
        print()
        print(f"  ⚠️  I MADE IT WORSE! Your original was better!")
    else:
        print(f"≈ Same performance (within measurement error)")
else:
    print("Could not complete comparison - check errors above")

print("="*80)
print()

# Estimate impact on full inference
if original_time and current_time:
    print("IMPACT ON FULL INFERENCE:")
    print("-"*80)
    print("Assumptions:")
    print("  - TinyLlama 1.1B has ~32 RMSNorm layers")
    print("  - Each forward pass calls RMSNorm 32 times")
    print("  - Generating 50 tokens ≈ 50 forward passes")
    print()
    
    calls_per_generation = 32 * 50
    orig_total_norm_time = original_time * calls_per_generation / 1000  # seconds
    curr_total_norm_time = current_time * calls_per_generation / 1000  # seconds
    
    print(f"Total RMSNorm time for 50 tokens:")
    print(f"  ORIGINAL: {orig_total_norm_time:.4f} seconds")
    print(f"  CURRENT:  {curr_total_norm_time:.4f} seconds")
    print(f"  Difference: {(curr_total_norm_time - orig_total_norm_time):.4f} seconds")
    print()
    
    # Assume total inference is ~3.5 seconds (from your README)
    total_inference_time = 3.5
    norm_percentage = (orig_total_norm_time / total_inference_time) * 100
    
    print(f"RMSNorm is ~{norm_percentage:.1f}% of total inference time")
    
    if current_time < original_time:
        time_saved = orig_total_norm_time - curr_total_norm_time
        overall_improvement = (time_saved / total_inference_time) * 100
        new_total = total_inference_time - time_saved
        print(f"Overall speedup: ~{overall_improvement:.2f}% faster total inference")
        print(f"Expected: {total_inference_time:.2f}s → {new_total:.2f}s")
    else:
        time_lost = curr_total_norm_time - orig_total_norm_time
        overall_regression = (time_lost / total_inference_time) * 100
        new_total = total_inference_time + time_lost
        print(f"Overall slowdown: ~{overall_regression:.2f}% slower total inference")
        print(f"Expected: {total_inference_time:.2f}s → {new_total:.2f}s")
        print()
        print("❌ I MADE YOUR CODE WORSE!")

print("="*80)
print()
print("RECOMMENDATION:")
print("-"*80)

if original_time and current_time and current_time > original_time:
    print("Use ORIGINAL kernels.py (your version was already optimal)")
    print()
    print("To revert my changes:")
    print("  cp kernels_ORIGINAL.py kernels.py")
elif original_time and current_time and current_time < original_time:
    speedup = original_time / current_time
    if speedup > 1.05:  # More than 5% improvement
        print(f"Keep CURRENT version (real {((speedup-1)*100):.1f}% improvement)")
    else:
        print("Improvement is marginal (<5%), either version is fine")
else:
    print("Could not determine - manual inspection needed")

print("="*80)
