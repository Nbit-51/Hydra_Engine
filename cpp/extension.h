#pragma once

#include <torch/torch.h>
#include "simd_verify.h"

// Python-facing wrapper (handles GPU->CPU transfer)
int verify_matches(torch::Tensor draft_ids, torch::Tensor target_preds);
