# Hydra Engine: High-Performance LLM Inference Infrastructure

Hydra Engine is an optimized inference framework designed for low-latency LLM serving on NVIDIA Ada Lovelace architectures. The system implements hardware-software co-design principles to maximize throughput on resource-constrained hardware.

## Performance Analysis
- **Kernel Execution:** 1.68x speedup in steady-state execution compared to standard PyTorch 2.x implementations.
- **Memory Bandwidth:** 68% reduction in global VRAM round-trips via operator fusion.
- **VRAM Optimization:** Validated 1.1B+ parameter model execution on 6GB VRAM (RTX 4050) using 4-bit NF4 quantization.

## Technical Specification
- **GPU Kernels:** OpenAI Triton implementation of Fused RMSNorm and Residual Addition.
- **Systems Integration:** C++17 sampling module integrated via PyBind11.
- **Algorithm:** Speculative Decoding architecture for accelerated token generation.

## Architectural Optimizations

### 1. Fused Triton Kernels
The engine replaces standard modular layers with custom fused kernels. By combining Residual Addition and Root Mean Square Normalization into a single GPU kernel, the system maintains intermediate activation data within the GPU SRAM (L1/Shared Memory), bypassing the bandwidth bottleneck of the global memory (HBM/VRAM).

### 2. C++ Sampling Bridge
To eliminate the overhead of the Python Global Interpreter Lock (GIL) and the latency of high-level tensor operations, the Top-K sampling logic is implemented as a compiled C++ extension.

## Build and Run
1. Compile the C++ sampling extension:
   python setup.py build_ext --inplace

2. Execute the comparative performance benchmark:
   python benchmark_report.py

---
Navaneeth Singh (Nbit-51)

## Native C++ LibTorch Engine (Zero Python Overhead)

To eliminate Python's Global Interpreter Lock (GIL) and CPU-bound launch latency during speculative decoding, the generation loop was rewritten in pure C++ using **LibTorch**. 

The Hugging Face `TinyLlama-1.1B` model is traced into a static `.pt` execution graph. Memory allocation, inference, and token verification happen entirely within the C++ runtime, utilizing direct pointer memory access (`.data_ptr<int64_t>()`) and AVX2/O3 compiler optimizations.

**Performance on RTX 4050 (Mobile):**
* **Latency:** `16.6 ms / step` (End-to-End Speculative Step)
* **Throughput:** ~60 iterations/sec
* **Peak Generation:** Up to 240 tokens/sec (Processing 4 speculative draft tokens per step)

### Build Instructions for C++ Engine
```bash
# 1. Export model to static TorchScript graph
python3 export_to_cpp.py

# 2. Build the engine with CMake
mkdir build && cd build
cmake -DCMAKE_PREFIX_PATH=$(python3 -c 'import torch; print(torch.utils.cmake_prefix_path)') ..
make

# 3. Execute zero-overhead runtime
./hydra_native
\```
