#include <torch/script.h>
#include <torch/torch.h>
#include <iostream>
#include <chrono>
#include <memory>
#include <vector>
#include "extension.h"

// CUDA stream management for async execution
class CUDAStreamPool {
private:
    std::vector<c10::cuda::CUDAStream> streams;
    size_t current = 0;

public:
    explicit CUDAStreamPool(size_t count = 2) {
        for (size_t i = 0; i < count; ++i) {
            streams.push_back(c10::cuda::getStreamFromPool(false));
        }
    }

    c10::cuda::CUDAStream& getStream() {
        auto& stream = streams[current];
        current = (current + 1) % streams.size();
        return stream;
    }
};

int main() {
    std::cout << "=== HYDRA C++ ENGINE - ULTRA OPTIMIZED ===" << std::endl;
    
    // 1. Configure CUDA for maximum performance
    torch::Device device(torch::kCUDA);
    
    // Enable TF32 for faster matmul on Ampere+ GPUs
    at::globalContext().setAllowTF32CuBLAS(true);
    at::globalContext().setAllowTF32CuDNN(true);
    
    // Disable gradient computation globally
    torch::NoGradGuard no_grad;
    
    std::cout << "[1/4] Loading optimized TorchScript model..." << std::endl;
    torch::jit::script::Module model;
    
    try {
        // Load with memory mapping for faster loading
        model = torch::jit::load("hydra_1_1B.pt", device);
        model.eval();
        
        // Freeze model for inference optimizations
        torch::jit::freeze(model);
        
    } catch (const c10::Error& e) {
        std::cerr << "ERROR: Failed to load model - " << e.what() << std::endl;
        return -1;
    }

    // 2. Pre-allocate persistent GPU tensors (avoid allocation overhead)
    std::cout << "[2/4] Pre-allocating GPU memory..." << std::endl;
    
    constexpr int64_t seq_len = 8;
    constexpr int64_t draft_len = 4;
    constexpr int64_t vocab_size = 32000;
    
    auto options = torch::TensorOptions()
        .dtype(torch::kLong)
        .device(device)
        .memory_format(torch::MemoryFormat::Contiguous);
    
    auto input_ids = torch::randint(0, vocab_size, {1, seq_len}, options);
    auto draft_ids = torch::randint(0, vocab_size, {1, draft_len}, options);
    
    // Pre-allocate CPU pinned memory for faster transfers
    auto cpu_options = torch::TensorOptions()
        .dtype(torch::kLong)
        .device(torch::kCPU)
        .pinned_memory(true);
    
    auto draft_cpu = torch::empty({1, draft_len}, cpu_options);
    auto target_cpu = torch::empty({1, draft_len}, cpu_options);
    
    std::vector<torch::jit::IValue> inputs;
    inputs.reserve(1);
    inputs.push_back(input_ids);

    // 3. Create CUDA streams for overlapping computation
    CUDAStreamPool stream_pool(2);
    auto main_stream = c10::cuda::getCurrentCUDAStream();
    
    // 4. Extended warmup with graph optimization
    std::cout << "[3/4] Warming up (compiling CUDA kernels)..." << std::endl;
    
    try {
        for (int i = 0; i < 20; i++) {
            auto logits = model.forward(inputs).toTensor();
            
            // Touch all memory paths during warmup
            if (i % 5 == 0) {
                auto tokens = logits.argmax(-1);
                torch::cuda::synchronize();
            }
        }
        torch::cuda::synchronize();
    } catch (const c10::Error& e) {
        std::cerr << "WARNING: Warmup failed - " << e.what() << std::endl;
        std::cerr << "Continuing anyway..." << std::endl;
    }

    // 5. Main benchmark loop with optimizations
    std::cout << "[4/4] Running benchmark (300 iterations)..." << std::endl;
    
    constexpr int iterations = 300;
    int total_accepted = 0;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < iterations; i++) {
        // Forward pass (GPU computation)
        auto logits = model.forward(inputs).toTensor();
        
        // Extract target predictions using slicing
        auto last_logits = logits.slice(1, -draft_len, logits.size(1));
        auto target_tokens = last_logits.argmax(-1);
        
        // Async copy to pinned CPU memory
        target_cpu.copy_(target_tokens, /*non_blocking=*/true);
        draft_cpu.copy_(draft_ids, /*non_blocking=*/true);
        
        // Sync before CPU comparison
        torch::cuda::synchronize();
        
        const int64_t* d_ptr = draft_cpu.data_ptr<int64_t>();
        const int64_t* t_ptr = target_cpu.data_ptr<int64_t>();
        
        // SIMD verification (with AVX2 support and scalar fallback)
        int n_accepted = verify_matches_simd(d_ptr, t_ptr, draft_len);
        total_accepted += n_accepted;
    }
    
    torch::cuda::synchronize();
    auto end = std::chrono::high_resolution_clock::now();
    
    // Calculate and display results
    std::chrono::duration<double, std::milli> elapsed = end - start;
    double ms_per_iter = elapsed.count() / iterations;
    double throughput = 1000.0 / ms_per_iter;  // iterations per second
    double avg_accepted = static_cast<double>(total_accepted) / iterations;
    
    std::cout << "\n" << std::string(50, '=') << std::endl;
    std::cout << "HYDRA C++ ENGINE - PERFORMANCE REPORT" << std::endl;
    std::cout << std::string(50, '=') << std::endl;
    std::cout << "Total iterations:     " << iterations << std::endl;
    std::cout << "Total time:           " << elapsed.count() << " ms" << std::endl;
    std::cout << "Avg time per iter:    " << ms_per_iter << " ms" << std::endl;
    std::cout << "Throughput:           " << throughput << " iter/s" << std::endl;
    std::cout << "Avg tokens accepted:  " << avg_accepted << "/" << draft_len << std::endl;
    std::cout << std::string(50, '=') << std::endl;

    return 0;
}
