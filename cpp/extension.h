#pragma once

#include <torch/torch.h>

// Ultra-optimized verification with SIMD and early exit
int verify_matches_simd(const int64_t* draft, const int64_t* target, int size);

// Main function with optimized memory handling
int verify_matches(torch::Tensor draft_ids, torch::Tensor target_preds);
