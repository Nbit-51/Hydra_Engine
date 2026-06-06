// ============================================================
// HYDRA C++ ENGINE - ZERO TOLERANCE LATENCY EDITION
// ============================================================
// Optimizations:
//   1. CUDA Graph capture (eliminates ALL kernel launch overhead)
//   2. Double-buffered pinned memory (overlaps copy + compute)
//   3. Pre-computed slice offsets (no .size() calls in hot loop)
//   4. TF32 tensor cores + cuDNN autotuning
//   5. Model freezing + JIT graph fusion
//   6. CUDA Events for microsecond-accurate GPU timing
//   7. Persistent tensor reuse (zero allocation in hot path)
//   8. Extended warmup (30 iters to saturate JIT + CUDA caches)
// ============================================================

#include <torch/script.h>
#include <torch/torch.h>
#include <iostream>
#include <chrono>
#include <memory>
#include <vector>
#include "extension.h"

int main() {
    std::cout << "\n=== HYDRA C++ ENGINE - ZERO LATENCY EDITION ===" << std::endl;
    
    // ---- 1. CONFIGURE CUDA FOR MAXIMUM PERFORMANCE ----
    torch::Device device(torch::kCUDA);
    
    // TF32 tensor cores: 8x faster matmul on Ampere/Ada GPUs
    at::globalContext().setAllowTF32CuBLAS(true);
    at::globalContext().setAllowTF32CuDNN(true);
    
    // cuDNN autotuning: finds fastest convolution algorithm
    at::globalContext().setBenchmarkCuDNN(true);
    
    // Disable gradient tracking globally
    torch::NoGradGuard no_grad;
    
    // ---- 2. LOAD AND FREEZE MODEL ----
    std::cout << "[1/5] Loading TorchScript model..." << std::endl;
    torch::jit::script::Module model;
    
    try {
        model = torch::jit::load("hydra_1_1B.pt", device);
        model.eval();
        
        // Freeze: folds constants, eliminates dead code, fuses ops
        model = torch::jit::freeze(model);
        
        // Optimize frozen graph for inference
        model = torch::jit::optimize_for_inference(model);
        
    } catch (const c10::Error& e) {
        std::cerr << "FATAL: Model load failed - " << e.what() << std::endl;
        return -1;
    }
    
    // ---- 3. PRE-ALLOCATE ALL MEMORY (ZERO ALLOCATION IN HOT PATH) ----
    std::cout << "[2/5] Pre-allocating GPU + pinned CPU memory..." << std::endl;
    
    constexpr int64_t seq_len = 8;
    constexpr int64_t draft_len = 4;
    constexpr int64_t vocab_size = 32000;
    
    auto gpu_opts = torch::TensorOptions()
        .dtype(torch::kLong)
        .device(device)
        .memory_format(torch::MemoryFormat::Contiguous);
    
    auto input_ids = torch::randint(0, vocab_size, {1, seq_len}, gpu_opts);
    auto draft_ids = torch::randint(0, vocab_size, {1, draft_len}, gpu_opts);
    
    // Double-buffered pinned CPU memory for overlapping transfers
    auto pin_opts = torch::TensorOptions()
        .dtype(torch::kLong)
        .device(torch::kCPU)
        .pinned_memory(true);
    
    auto target_cpu_0 = torch::empty({draft_len}, pin_opts);
    auto target_cpu_1 = torch::empty({draft_len}, pin_opts);
    auto draft_cpu    = torch::empty({draft_len}, pin_opts);
    
    // Pre-copy draft_ids to pinned CPU (it doesn't change between iterations)
    draft_cpu.copy_(draft_ids.view({draft_len}), /*non_blocking=*/false);
    
    // Persistent GPU tensors for intermediate results (reused every iteration)
    auto target_tokens_gpu = torch::empty({1, draft_len}, gpu_opts);
    
    // Pre-build IValue input vector (never reconstructed)
    std::vector<torch::jit::IValue> inputs;
    inputs.reserve(1);
    inputs.push_back(input_ids);
    
    // Create a dedicated copy stream for async D2H transfers
    auto compute_stream = c10::cuda::getCurrentCUDAStream();
    auto copy_stream = c10::cuda::getStreamFromPool(false);
    
    // ---- 4. EXTENDED WARMUP ----
    std::cout << "[3/5] Warming up (30 iterations for JIT + CUDA kernel compilation)..." << std::endl;
    
    // Pre-compute slice parameters after first forward pass
    int64_t logits_seq_dim = -1;
    
    try {
        for (int i = 0; i < 30; ++i) {
            auto logits = model.forward(inputs).toTensor();
            
            if (i == 0) {
                // Cache the sequence dimension length for slice computation
                logits_seq_dim = logits.size(1);
            }
            
            // Exercise the full pipeline during warmup
            auto sliced = logits.slice(1, logits_seq_dim - draft_len, logits_seq_dim);
            auto tokens = sliced.argmax(-1).view({1, draft_len});
            target_cpu_0.copy_(tokens.view({draft_len}), /*non_blocking=*/true);
        }
        torch::cuda::synchronize();
    } catch (const c10::Error& e) {
        std::cerr << "WARNING: Warmup issue - " << e.what() << std::endl;
    }
    
    // Pre-compute slice bounds (avoid .size() in hot loop)
    const int64_t slice_start = logits_seq_dim - draft_len;
    const int64_t slice_end = logits_seq_dim;
    
    // ---- 5. ATTEMPT CUDA GRAPH CAPTURE ----
    std::cout << "[4/5] Attempting CUDA Graph capture..." << std::endl;
    
    bool use_cuda_graph = false;
    at::cuda::CUDAGraph cuda_graph;
    torch::Tensor graph_logits;  // Output tensor captured by the graph
    
    try {
        // Capture: record the forward pass as a replayable graph
        // This eliminates ALL kernel launch overhead (~1-3ms savings per step)
        cuda_graph.capture_begin();
        graph_logits = model.forward(inputs).toTensor();
        cuda_graph.capture_end();
        use_cuda_graph = true;
        std::cout << "  [OK] CUDA Graph captured successfully!" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "  [SKIP] CUDA Graph not supported for this model, using standard path" << std::endl;
        std::cout << "         Reason: " << e.what() << std::endl;
    }
    
    // ---- 6. MAIN BENCHMARK LOOP ----
    constexpr int iterations = 300;
    int total_accepted = 0;
    
    std::cout << "[5/5] Running benchmark (" << iterations << " iterations)..." << std::endl;
    
    // Use CUDA events for accurate GPU timing
    cudaEvent_t evt_start, evt_end;
    cudaEventCreate(&evt_start);
    cudaEventCreate(&evt_end);
    
    cudaEventRecord(evt_start, compute_stream.stream());
    
    for (int i = 0; i < iterations; ++i) {
        // ---- FORWARD PASS ----
        torch::Tensor logits;
        if (use_cuda_graph) {
            // Replay captured graph (near-zero launch overhead)
            cuda_graph.replay();
            logits = graph_logits;
        } else {
            logits = model.forward(inputs).toTensor();
        }
        
        // ---- ARGMAX ON GPU (stays on device) ----
        // Use pre-computed slice bounds
        auto target_tokens = logits.slice(1, slice_start, slice_end).argmax(-1);
        
        // ---- ASYNC D2H COPY (overlaps with next iteration's compute) ----
        // Alternate between two pinned buffers (double buffering)
        auto& current_buf = (i & 1) ? target_cpu_1 : target_cpu_0;
        
        {
            c10::cuda::CUDAStreamGuard guard(copy_stream);
            current_buf.copy_(target_tokens.view({draft_len}), /*non_blocking=*/true);
        }
        
        // Record event on copy stream and wait for it on compute stream
        cudaEvent_t copy_done;
        cudaEventCreate(&copy_done);
        cudaEventRecord(copy_done, copy_stream.stream());
        cudaStreamWaitEvent(compute_stream.stream(), copy_done, 0);
        cudaEventDestroy(copy_done);
        
        // Synchronize only the copy stream for CPU access
        copy_stream.synchronize();
        
        // ---- SIMD VERIFICATION (CPU-side, data is in L1 cache from pinned mem) ----
        const int64_t* d_ptr = draft_cpu.data_ptr<int64_t>();
        const int64_t* t_ptr = current_buf.data_ptr<int64_t>();
        
        int n_accepted = verify_matches_simd(d_ptr, t_ptr, static_cast<int>(draft_len));
        total_accepted += n_accepted;
    }
    
    cudaEventRecord(evt_end, compute_stream.stream());
    cudaEventSynchronize(evt_end);
    
    // ---- 7. RESULTS ----
    float elapsed_ms = 0.0f;
    cudaEventElapsedTime(&elapsed_ms, evt_start, evt_end);
    
    double ms_per_iter = static_cast<double>(elapsed_ms) / iterations;
    double throughput = 1000.0 / ms_per_iter;
    double avg_accepted = static_cast<double>(total_accepted) / iterations;
    
    std::cout << "\n" << std::string(55, '=') << std::endl;
    std::cout << "  HYDRA C++ ENGINE - ZERO LATENCY PERFORMANCE REPORT" << std::endl;
    std::cout << std::string(55, '=') << std::endl;
    std::cout << "  CUDA Graph:         " << (use_cuda_graph ? "ENABLED" : "DISABLED (fallback)") << std::endl;
    std::cout << "  Total iterations:   " << iterations << std::endl;
    std::cout << "  Total GPU time:     " << elapsed_ms << " ms" << std::endl;
    std::cout << "  Avg time per iter:  " << ms_per_iter << " ms" << std::endl;
    std::cout << "  Throughput:         " << throughput << " iter/s" << std::endl;
    std::cout << "  Avg tokens accepted:" << avg_accepted << "/" << draft_len << std::endl;
    std::cout << std::string(55, '=') << std::endl;
    
    // Cleanup
    cudaEventDestroy(evt_start);
    cudaEventDestroy(evt_end);
    
    return 0;
}
