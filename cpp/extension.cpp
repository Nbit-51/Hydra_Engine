#include "extension.h"

int verify_matches(torch::Tensor draft_ids, torch::Tensor target_preds) {
    TORCH_CHECK(
        draft_ids.numel() == target_preds.numel(),
        "verify_matches: size mismatch (",
        draft_ids.numel(),
        " vs ",
        target_preds.numel(),
        ")"
    );

    const int size = static_cast<int>(draft_ids.numel());
    if (size == 0) return 0;

    if (
        draft_ids.device().is_cpu() &&
        target_preds.device().is_cpu() &&
        draft_ids.is_contiguous() &&
        target_preds.is_contiguous()
    ) {
        return verify_matches_simd(
            draft_ids.data_ptr<int64_t>(),
            target_preds.data_ptr<int64_t>(),
            size
        );
    }

    const bool from_gpu = draft_ids.is_cuda() || target_preds.is_cuda();
    auto cpu_options = torch::TensorOptions()
        .dtype(torch::kLong)
        .device(torch::kCPU)
        .pinned_memory(from_gpu);

    auto draft_cpu = draft_ids.to(cpu_options, false).contiguous();
    auto target_cpu = target_preds.to(cpu_options, false).contiguous();
    return verify_matches_simd(
        draft_cpu.data_ptr<int64_t>(),
        target_cpu.data_ptr<int64_t>(),
        size
    );
}
