#include <torch/script.h>
#include <torch/torch.h>
#include <iostream>
#include <chrono>

int main() {
    // 1. Force C++ to use the GPU
    torch::Device device(torch::kCUDA);
    torch::jit::script::Module model;

    std::cout << "Loading Native C++ Model to GPU..." << std::endl;
    try {
        model = torch::jit::load("hydra_1_1B.pt", device);
        model.eval();
    } catch (const c10::Error& e) {
        std::cerr << "Error loading model. Check file path.\n";
        return -1;
    }

    // 2. Setup Memory directly in C++
    auto input_ids = torch::randint(0, 30000, {1, 8}, torch::TensorOptions().dtype(torch::kLong).device(device));
    auto draft_ids = torch::randint(0, 30000, {1, 4}, torch::TensorOptions().dtype(torch::kLong).device(device));
    
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(input_ids);

    std::cout << "Warming up C++ GPU Backend..." << std::endl;
    for (int i = 0; i < 10; i++) {
        model.forward(inputs);
    }
    torch::cuda::synchronize();

    std::cout << "Running Pure C++ Benchmark (200 steps)..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < 200; i++) {
        // 3. Ultra-fast native execution (Returns pure tensor)
        auto logits = model.forward(inputs).toTensor();

        // 4. Extract target tokens directly in GPU memory
        auto target_tokens = logits.slice(1, -4, torch::indexing::None).argmax(-1);

        // 5. Raw Memory Speculative Verification
        auto t_cpu = target_tokens.to(torch::kCPU).contiguous();
        auto d_cpu = draft_ids.to(torch::kCPU).contiguous();
        
        const int64_t* d_ptr = d_cpu.data_ptr<int64_t>();
        const int64_t* t_ptr = t_cpu.data_ptr<int64_t>();
        
        int n_accepted = 0;
        for(int j = 0; j < 4; j++) {
            if(d_ptr[j] == t_ptr[j]) n_accepted++;
            else break;
        }
    }
    
    torch::cuda::synchronize();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> ms = end - start;

    std::cout << "\n========================================" << std::endl;
    std::cout << "HYDRA PURE C++ ENGINE REPORT" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Latency per step: " << (ms.count() / 200.0) << " ms" << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}
