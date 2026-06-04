#include <torch/extension.h>

// Optimized C++ verification using direct pointer access
int verify_matches(torch::Tensor draft_ids, torch::Tensor target_preds) {
    // 1. Safely move to CPU and ensure memory is contiguous
    auto d = draft_ids.to(torch::kCPU).contiguous();
    auto t = target_preds.to(torch::kCPU).contiguous();
    
    // 2. Extract raw C++ pointers (Bypasses all PyTorch overhead)
    const int64_t* d_ptr = d.data_ptr<int64_t>();
    const int64_t* t_ptr = t.data_ptr<int64_t>();
    
    int n_accepted = 0;
    int size = d.numel();
    
    // 3. Raw loop (Compiler will auto-vectorize this using AVX/SSE)
    for (int i = 0; i < size; ++i) {
        if (d_ptr[i] == t_ptr[i]) {
            n_accepted++;
        } else {
            break;
        }
    }
    return n_accepted;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("verify_matches", &verify_matches, "High-performance direct pointer verification");
}
