#pragma once

#include <torch/torch.h>
#include <cstdint>

// Core SIMD verification (defined in extension.cpp)
int verify_matches_simd(const int64_t* __restrict__ draft, 
                        const int64_t* __restrict__ target, 
                        int size);

// Python-facing wrapper (handles GPU->CPU transfer)
int verify_matches(torch::Tensor draft_ids, torch::Tensor target_preds);
