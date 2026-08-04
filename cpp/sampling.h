#pragma once

#include <vector>

// Memory-efficient top-K filtering using manual binary heap (O(n log k))
// Uses inline heap ops with branchless sift-down and cache-aligned storage
std::vector<int> top_k_filter(const std::vector<float>& logits, int k);

// Fast top-K using nth_element partitioning (O(n) average)
// Optimized lambda to avoid std::function overhead
std::vector<int> top_k_filter_fast(const std::vector<float>& logits, int k);

// SIMD-accelerated top-K: uses AVX2 to scan for max in 8-float chunks
// Falls back to manual heap for extraction
std::vector<int> top_k_filter_simd(const std::vector<float>& logits, int k);
