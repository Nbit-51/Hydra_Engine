#!/bin/bash
# RUN THIS ON UBUNTU TO GET ACTUAL PERFORMANCE DATA

set -e

echo "=============================================================================="
echo "HYDRA ENGINE - HONEST PERFORMANCE TEST"
echo "=============================================================================="
echo ""
echo "This script will:"
echo "  1. Test ORIGINAL kernel (your 1.68x speedup version)"
echo "  2. Test CURRENT kernel (my 'optimized' version)"
echo "  3. Compare them side-by-side with REAL numbers"
echo ""
echo "Press ENTER to continue or Ctrl+C to cancel..."
read

# Check if CUDA is available
echo "[1/5] Checking CUDA availability..."
python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available!'; print(f'✓ GPU: {torch.cuda.get_device_name(0)}')"
echo ""

# Test if Triton is installed
echo "[2/5] Checking dependencies..."
python3 -c "import triton; print(f'✓ Triton: {triton.__version__}')"
python3 -c "from transformers import AutoModelForCausalLM; print('✓ Transformers installed')"
echo ""

# Run the kernel comparison
echo "[3/5] Running kernel-level comparison..."
echo "This tests ONLY the Triton kernel to see if my changes helped or hurt"
echo ""
python3 ACTUAL_TEST.py > kernel_results.txt 2>&1
cat kernel_results.txt
echo ""

# Extract results
ORIG_TIME=$(grep "ORIGINAL:" kernel_results.txt | grep -oP '\d+\.\d+' | head -1)
CURR_TIME=$(grep "CURRENT:" kernel_results.txt | grep -oP '\d+\.\d+' | head -1)

echo "[4/5] Running FULL inference benchmark..."
echo "This tests the complete pipeline with the model"
echo ""

# Run full benchmark
python3 benchmark_report.py > full_results.txt 2>&1
cat full_results.txt
echo ""

# Extract full inference results
BASELINE=$(grep "BASELINE" full_results.txt | grep -oP '\d+\.\d+ tokens/s' | head -1)
HYDRA=$(grep "HYDRA" full_results.txt | grep -oP '\d+\.\d+ tokens/s' | head -1)
SPEEDUP=$(grep "speedup" full_results.txt | grep -oP '\d+\.\d+x' | head -1)

echo "=============================================================================="
echo "FINAL RESULTS SUMMARY"
echo "=============================================================================="
echo ""
echo "KERNEL-LEVEL (Isolated Triton kernel):"
if [ ! -z "$ORIG_TIME" ] && [ ! -z "$CURR_TIME" ]; then
    echo "  ORIGINAL kernel: ${ORIG_TIME} ms"
    echo "  CURRENT kernel:  ${CURR_TIME} ms"
    
    FASTER=$(python3 -c "print('CURRENT' if $CURR_TIME < $ORIG_TIME else 'ORIGINAL')")
    DIFF=$(python3 -c "print(abs($CURR_TIME - $ORIG_TIME))")
    PERCENT=$(python3 -c "print(abs(($CURR_TIME / $ORIG_TIME - 1) * 100))")
    
    echo "  Winner: $FASTER (${PERCENT}% difference)"
    
    if [ "$FASTER" = "ORIGINAL" ]; then
        echo ""
        echo "  ⚠️  WARNING: CURRENT is SLOWER! My changes made it WORSE!"
    fi
else
    echo "  Could not extract kernel times - check kernel_results.txt"
fi

echo ""
echo "FULL INFERENCE (Complete pipeline with model):"
if [ ! -z "$BASELINE" ] && [ ! -z "$HYDRA" ]; then
    echo "  Baseline PyTorch: $BASELINE"
    echo "  Hydra (Triton):   $HYDRA"
    echo "  Speedup:          $SPEEDUP"
    
    # Check if it's still around 1.68x
    SPEEDUP_NUM=$(echo $SPEEDUP | grep -oP '\d+\.\d+')
    IS_GOOD=$(python3 -c "print('YES' if $SPEEDUP_NUM >= 1.60 else 'NO')")
    
    if [ "$IS_GOOD" = "YES" ]; then
        echo "  ✓ Still achieving ~1.68x speedup (excellent!)"
    else
        echo "  ✗ Below expected 1.68x - something got worse!"
    fi
else
    echo "  Could not extract inference results - check full_results.txt"
fi

echo ""
echo "=============================================================================="
echo ""

# Create summary report
echo "[5/5] Creating summary report..."

cat > PERFORMANCE_REPORT.txt << EOF
HYDRA ENGINE PERFORMANCE TEST REPORT
Generated: $(date)
GPU: $(python3 -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')")

====================================
KERNEL-LEVEL COMPARISON
====================================
ORIGINAL kernel: ${ORIG_TIME} ms
CURRENT kernel:  ${CURR_TIME} ms
Difference: ${DIFF} ms (${PERCENT}%)
Winner: ${FASTER}

====================================
FULL INFERENCE BENCHMARK
====================================
Baseline PyTorch: ${BASELINE}
Hydra (Triton):   ${HYDRA}
Speedup:          ${SPEEDUP}

====================================
VERDICT
====================================
EOF

if [ "$FASTER" = "ORIGINAL" ]; then
    cat >> PERFORMANCE_REPORT.txt << EOF
❌ The "optimizations" made the kernel SLOWER
   Recommendation: REVERT to original kernels.py
   
   To revert:
     cp kernels_ORIGINAL.py kernels.py
     
   The original 1.68x speedup was already excellent.
   My "optimizations" hurt performance.
EOF
else
    KERNEL_GAIN=$(python3 -c "print(($ORIG_TIME / $CURR_TIME - 1) * 100)")
    cat >> PERFORMANCE_REPORT.txt << EOF
✓ The optimizations made the kernel FASTER by ${KERNEL_GAIN}%
  However, check if full inference speedup is maintained at ~1.68x
  
  If full speedup dropped below 1.60x, the optimizations aren't worth it.
EOF
fi

cat PERFORMANCE_REPORT.txt
echo ""
echo "Report saved to: PERFORMANCE_REPORT.txt"
echo ""
echo "=============================================================================="
echo "TEST COMPLETE!"
echo "=============================================================================="
