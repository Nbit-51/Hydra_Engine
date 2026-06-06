#include <torch/script.h>
#include <torch/torch.h>
#include <iostream>
#include <vector>
#include "extension.h"
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDAGraph.h>

int main() {
    std::cout << "\n=== HYDRA C++ ENGINE - CUDA GRAPH EDITION ===" << std::endl;

    torch::Device device(torch::kCUDA);
    at::globalContext().setAllowTF32CuBLAS(true);
    at::globalContext().setAllowTF32CuDNN(true);
    at::globalContext().setBenchmarkCuDNN(true);
    torch::NoGradGuard no_grad;

    std::cout << "[1/5] Loading model..." << std::endl;
    torch::jit::script::Module model;
    try {
        model = torch::jit::load("hydra_1_1B.pt", device);
        model.eval();
    } catch (const c10::Error& e) {
        std::cerr << "FATAL: " << e.what() << std::endl;
        return -1;
    }

    std::cout << "[2/5] Allocating memory..." << std::endl;

    constexpr int64_t seq_len    = 8;
    constexpr int64_t draft_len  = 4;
    constexpr int64_t vocab_size = 32000;
    constexpr int64_t slice_start = seq_len - draft_len;  // 4
    constexpr int64_t slice_end   = seq_len;              // 8

    auto gpu_long = torch::TensorOptions().dtype(torch::kLong).device(device);
    auto pin_long = torch::TensorOptions().dtype(torch::kLong).device(torch::kCPU).pinned_memory(true);

    // Static input tensor — MUST be the same tensor for CUDA graph replay
    auto input_ids    = torch::randint(0, vocab_size, {1, seq_len}, gpu_long);
    auto target_cpu_0 = torch::empty({draft_len}, pin_long);
    auto target_cpu_1 = torch::empty({draft_len}, pin_long);
    auto draft_cpu    = torch::empty({draft_len}, pin_long);

    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(input_ids);

    auto compute_stream = c10::cuda::getCurrentCUDAStream();
    auto copy_stream    = c10::cuda::getStreamFromPool(false);

    // 3. WARMUP on default stream
    std::cout << "[3/5] Warming up (30 iters)..." << std::endl;
    for (int i = 0; i < 30; ++i) {
        auto logits = model.forward(inputs).toTensor();
        if (i == 29) {
            auto tokens = logits.slice(1, slice_start, slice_end).argmax(-1).view({draft_len});
            draft_cpu.copy_(tokens, false);
            std::cout << "  Draft seeded from real predictions" << std::endl;
        }
    }
    torch::cuda::synchronize();

    // 4. CUDA GRAPH CAPTURE
    std::cout << "[4/5] Capturing CUDA Graph..." << std::endl;

    bool use_cuda_graph = false;
    at::cuda::CUDAGraph cuda_graph;
    torch::Tensor graph_logits;

    try {
        auto capture_stream = c10::cuda::getStreamFromPool(false);

        // Warmup on capture stream (mandatory)
        {
            c10::cuda::CUDAStreamGuard g(capture_stream);
            for (int i = 0; i < 5; ++i)
                model.forward(inputs).toTensor();
        }
        torch::cuda::synchronize();

        // Capture on same stream
        {
            c10::cuda::CUDAStreamGuard g(capture_stream);
            cuda_graph.capture_begin();
            graph_logits = model.forward(inputs).toTensor();
            cuda_graph.capture_end();
        }
        torch::cuda::synchronize();

        // Verify replay
        cuda_graph.replay();
        torch::cuda::synchronize();

        use_cuda_graph = true;
        std::cout << "  [OK] CUDA Graph captured! Output: [1,"
                  << graph_logits.size(1) << "," << graph_logits.size(2) << "]" << std::endl;

    } catch (const std::exception& e) {
        use_cuda_graph = false;
        std::cout << "  [FAIL] " << e.what() << std::endl;
    }

    // 5. BENCHMARK
    constexpr int iterations = 300;
    int total_accepted = 0;

    std::cout << "[5/5] Benchmarking " << iterations << " iters ["
              << (use_cuda_graph ? "CUDA GRAPH" : "Standard") << "]..." << std::endl;

    cudaEvent_t evt_start, evt_end;
    cudaEventCreate(&evt_start);
    cudaEventCreate(&evt_end);
    cudaEventRecord(evt_start, compute_stream.stream());

    for (int i = 0; i < iterations; ++i) {
        torch::Tensor logits;
        if (use_cuda_graph) {
            cuda_graph.replay();
            logits = graph_logits;
        } else {
            logits = model.forward(inputs).toTensor();
        }

        auto target_tokens = logits.slice(1, slice_start, slice_end).argmax(-1);

        auto& buf = (i & 1) ? target_cpu_1 : target_cpu_0;
        {
            c10::cuda::CUDAStreamGuard g(copy_stream);
            buf.copy_(target_tokens.view({draft_len}), true);
        }
        copy_stream.synchronize();

        total_accepted += verify_matches_simd(
            draft_cpu.data_ptr<int64_t>(),
            buf.data_ptr<int64_t>(),
            (int)draft_len
        );

        if ((i & 15) == 0)
            draft_cpu.copy_(target_tokens.view({draft_len}), false);
    }

    cudaEventRecord(evt_end, compute_stream.stream());
    cudaEventSynchronize(evt_end);

    float ms = 0;
    cudaEventElapsedTime(&ms, evt_start, evt_end);
    double ms_per_iter  = ms / iterations;
    double avg_accepted = (double)total_accepted / iterations;

    std::cout << "\n" << std::string(55, '=') << std::endl;
    std::cout << "  HYDRA C++ ENGINE - PERFORMANCE REPORT" << std::endl;
    std::cout << std::string(55, '=') << std::endl;
    std::cout << "  CUDA Graph:          " << (use_cuda_graph ? "ENABLED" : "DISABLED") << std::endl;
    std::cout << "  Avg time/iter:       " << ms_per_iter << " ms" << std::endl;
    std::cout << "  Throughput:          " << 1000.0/ms_per_iter << " iter/s" << std::endl;
    std::cout << "  Avg tokens accepted: " << avg_accepted << "/" << draft_len << std::endl;
    std::cout << std::string(55, '=') << std::endl;

    cudaEventDestroy(evt_start);
    cudaEventDestroy(evt_end);
    return 0;
}
