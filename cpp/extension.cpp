#include <torch/extension.h>
#include <pybind11/pybind11.h>

// Only include SIMD on x86 platforms
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)
#include <immintrin.h>
#define HAS_AVX2
#endif

#include <cstring>

namespace py = pybind11;

// Ultra-optimized verification with SIMD and early exit
int verify_matches_simd(const int64_t* draft, 
                        const int64_t* target, 
                        int size) {
    int n_accepted = 0;
    
    #if defined(HAS_AVX2) && defined(__AVX2__)
    // AVX2 path: process 4 int64_t at once (256 bits)
    constexpr int simd_width = 4;
    int simd_iters = size / simd_width;
    
    for (int i = 0; i < simd_iters; ++i) {
        __m256i d = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(draft + i * simd_width));
        __m256i t = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(target + i * simd_width));
        __m256i cmp = _mm256_cmpeq_epi64(d, t);
        
        int mask = _mm256_movemask_epi8(cmp);
        
        // Check all 32 bytes match (0xFFFFFFFF)
        if (static_cast<uint32_t>(mask) != 0xFFFFFFFF) {
            // Find first mismatch
            for (int j = 0; j < simd_width; ++j) {
                if (draft[i * simd_width + j] != target[i * simd_width + j]) {
                    return n_accepted + j;
                }
            }
        }
        n_accepted += simd_width;
    }
    
    // Handle remainder
    for (int i = simd_iters * simd_width; i < size; ++i) {
        if (draft[i] == target[i]) n_accepted++;
        else return n_accepted;
    }
    
    #else
    // Scalar path with manual unrolling for better performance
    int unroll_iters = size / 4;
    for (int i = 0; i < unroll_iters; ++i) {
        int base = i * 4;
        if (draft[base] != target[base]) return n_accepted;
        n_accepted++;
        if (draft[base+1] != target[base+1]) return n_accepted;
        n_accepted++;
        if (draft[base+2] != target[base+2]) return n_accepted;
        n_accepted++;
        if (draft[base+3] != target[base+3]) return n_accepted;
        n_accepted++;
    }
    
    // Handle remainder
    for (int i = unroll_iters * 4; i < size; ++i) {
        if (draft[i] == target[i]) n_accepted++;
        else return n_accepted;
    }
    #endif
    
    return n_accepted;
}

// Main function with optimized memory handling
int verify_matches(torch::Tensor draft_ids, torch::Tensor target_preds) {
    // Validate inputs
    TORCH_CHECK(draft_ids.dim() == target_preds.dim(), "Dimension mismatch");
    TORCH_CHECK(draft_ids.numel() == target_preds.numel(), "Size mismatch");
    
    int size = draft_ids.numel();
    
    // Fast path: already on CPU and contiguous
    if (draft_ids.device().is_cpu() && target_preds.device().is_cpu() &&
        draft_ids.is_contiguous() && target_preds.is_contiguous()) {
        
        const int64_t* d_ptr = draft_ids.data_ptr<int64_t>();
        const int64_t* t_ptr = target_preds.data_ptr<int64_t>();
        return verify_matches_simd(d_ptr, t_ptr, size);
    }
    
    // Slow path: need to move to CPU with pinned memory for speed
    auto options = torch::TensorOptions()
        .dtype(torch::kLong)
        .device(torch::kCPU)
        .pinned_memory(draft_ids.is_cuda() || target_preds.is_cuda());
    
    auto d_cpu = draft_ids.to(options, /*non_blocking=*/false).contiguous();
    auto t_cpu = target_preds.to(options, /*non_blocking=*/false).contiguous();
    
    const int64_t* d_ptr = d_cpu.data_ptr<int64_t>();
    const int64_t* t_ptr = t_cpu.data_ptr<int64_t>();
    
    return verify_matches_simd(d_ptr, t_ptr, size);
}
