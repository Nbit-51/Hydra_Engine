#pragma once

#include <vector>

// Memory-efficient top-K filtering using heap (O(n log k))
std::vector<int> top_k_filter(const std::vector<float>& logits, int k);

// Fast approximate top-K using partitioning (O(n) average)
std::vector<int> top_k_filter_fast(const std::vector<float>& logits, int k);
