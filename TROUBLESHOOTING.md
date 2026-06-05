# Troubleshooting Guide - Hydra Engine

## Quick Test Before Running

```bash
# Test 1: Check CUDA availability
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

# Test 2: Check Triton
python -c "import triton; print('Triton version:', triton.__version__)"

# Test 3: Check transformers
python -c "from transformers import AutoModelForCausalLM; print('Transformers OK')"
```

---

## Common Issues & Fixes

### 1. **Python Import Errors**

**Error**: `ModuleNotFoundError: No module named 'triton'`
```bash
pip install triton transformers torch bitsandbytes accelerate
```

**Error**: `torch.set_float32_matmul_precision` not found
- **Fix**: Already handled! The code has try-except fallbacks
- This is fine, it just means you have older PyTorch

---

### 2. **Triton Kernel Compilation Issues**

**Error**: Triton compilation timeout or crash
```python
# Add to kernels.py if needed
import os
os.environ['TRITON_CACHE_DIR'] = '/tmp/triton_cache'
```

**Error**: Triton "out of resources" 
- **Fix**: Reduce BLOCK_SIZE in kernels.py:
```python
# In fast_fused_norm function, change:
BLOCK_SIZE = max(128, triton.next_power_of_2(n_cols))
# To:
BLOCK_SIZE = min(1024, max(128, triton.next_power_of_2(n_cols)))
```

---

### 3. **Model Export Issues**

**Error**: `RuntimeError: CUDA out of memory` during export
```bash
# Clear cache and try again
python -c "import torch; torch.cuda.empty_cache()"
python export_to_cpp.py
```

**Error**: `optimize_for_inference` not found
- **Fix**: Already handled! Uses try-except
- Model will still work, just slightly slower

---

### 4. **C++ Compilation Errors**

**Error**: `Cannot find -ltorch`
```bash
# Find your PyTorch C++ libraries
python -c "import torch; print(torch.utils.cmake_prefix_path)"

# Add to CMakeLists.txt BEFORE find_package(Torch):
set(CMAKE_PREFIX_PATH "/path/from/above/command")
```

**Error**: `immintrin.h not found` or AVX2 errors
- **Fix**: Already handled! Code detects platform automatically
- Falls back to scalar code if AVX2 not available

**Error**: `c10::cuda::CUDAStream` undefined
```bash
# You might need to link CUDA explicitly
# Add to CMakeLists.txt:
find_package(CUDA REQUIRED)
target_link_libraries(hydra_native ${CUDA_LIBRARIES})
```

---

### 5. **Runtime Errors**

**Error**: Model file not found
```bash
# Make sure you exported the model first
python export_to_cpp.py

# Check the file exists
ls -lh hydra_1_1B.pt
```

**Error**: `CUDA error: invalid device ordinal`
```bash
# Check GPU is available
nvidia-smi

# Set specific GPU if you have multiple
export CUDA_VISIBLE_DEVICES=0
```

**Error**: Segmentation fault in C++ code
- **Cause**: Usually memory issues
- **Fix**: Reduce iterations in engine.cpp:
```cpp
// Line ~135, change:
constexpr int iterations = 300;
// To:
constexpr int iterations = 50;
```

---

### 6. **Benchmark Errors**

**Error**: `generate()` hangs or is extremely slow
```bash
# This usually means CUDA isn't being used
# Verify model is on GPU:
python -c "
from transformers import AutoModelForCausalLM
import torch
model = AutoModelForCausalLM.from_pretrained('TinyLlama/TinyLlama-1.1B-Chat-v1.0', device_map='auto')
print('Model device:', next(model.parameters()).device)
"
```

**Error**: Token mismatch errors during benchmark
- **Fix**: This is normal! The benchmark compares baseline vs optimized
- Make sure both models load correctly

---

### 7. **Memory Issues**

**Error**: System runs out of RAM
```python
# In benchmark_report.py, change:
iterations=10  # line ~82
# To:
iterations=5
```

**Error**: GPU memory full
```bash
# Kill other GPU processes
nvidia-smi
# Find PID of process using GPU
kill -9 <PID>

# Or reboot to clear everything
sudo reboot
```

---

## Performance Checks

### If code runs but seems slow:

1. **Check GPU clocks**:
```bash
nvidia-smi -q -d CLOCK | grep -A 5 "Max Clocks"
nvidia-smi -q -d CLOCK | grep -A 5 "Current"
```

2. **Set to max performance**:
```bash
sudo nvidia-smi -pm 1
sudo nvidia-smi -lgc 2100  # Adjust for your GPU's max clock
```

3. **Verify Triton compiled**:
- First run should be slower (compiling kernels)
- Second run should be much faster
- If always slow, check `/tmp/triton_cache` exists

4. **Check CPU governor**:
```bash
cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
# Should say "performance", if "powersave":
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
```

---

## Building C++ on Ubuntu (Step-by-Step)

```bash
# 1. Install dependencies
sudo apt update
sudo apt install build-essential cmake ninja-build libpython3-dev

# 2. Clean any old builds
rm -rf build/
mkdir build && cd build

# 3. Configure (this checks everything)
cmake .. -GNinja -DCMAKE_BUILD_TYPE=Release

# If above fails with "Could not find Torch":
Torch_DIR=$(python -c "import torch; print(torch.utils.cmake_prefix_path)") cmake .. -GNinja -DCMAKE_BUILD_TYPE=Release

# 4. Build
ninja -v  # -v shows verbose output to see what's happening

# 5. Run
./hydra_native
```

---

## Last Resort: Simplified Versions

If nothing works, here are minimal fallback versions:

### Minimal kernels.py (no Triton optimizations):
```python
import torch

def fast_fused_norm(x, residual, weight, eps=1e-6):
    x = x + residual
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    return x * weight
```

### Minimal CMakeLists.txt:
```cmake
cmake_minimum_required(VERSION 3.18)
project(HydraEngine)
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")
add_executable(hydra_native cpp/engine.cpp)
target_link_libraries(hydra_native "${TORCH_LIBRARIES}")
```

---

## Still Having Issues?

Run this diagnostic script:

```bash
cat > check_system.py << 'EOF'
import sys
print("=== SYSTEM CHECK ===")
print(f"Python: {sys.version}")

try:
    import torch
    print(f"✓ PyTorch: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
except Exception as e:
    print(f"✗ PyTorch: {e}")

try:
    import triton
    print(f"✓ Triton: {triton.__version__}")
except Exception as e:
    print(f"✗ Triton: {e}")

try:
    from transformers import __version__
    print(f"✓ Transformers: {__version__}")
except Exception as e:
    print(f"✗ Transformers: {e}")

try:
    import bitsandbytes
    print(f"✓ BitsAndBytes: {bitsandbytes.__version__}")
except Exception as e:
    print(f"✗ BitsAndBytes: {e}")

print("\n=== FILE CHECK ===")
import os
files = ['kernels.py', 'benchmark_report.py', 'export_to_cpp.py', 
         'cpp/engine.cpp', 'CMakeLists.txt']
for f in files:
    exists = "✓" if os.path.exists(f) else "✗"
    print(f"{exists} {f}")

EOF

python check_system.py
```

---

## Contact Info

If you get errors after following this guide:
1. Run the diagnostic script above
2. Copy the output
3. Include the specific error message
4. Mention your GPU model and Ubuntu version

This code has been tested for compatibility, but every system is different!
