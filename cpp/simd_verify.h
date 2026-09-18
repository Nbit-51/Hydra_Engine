#pragma once

#include <cstdint>

#if defined(_MSC_VER)
#define HYDRA_RESTRICT __restrict
#else
#define HYDRA_RESTRICT __restrict__
#endif

int verify_matches_simd(
    const int64_t* HYDRA_RESTRICT draft,
    const int64_t* HYDRA_RESTRICT target,
    int size
);

const char* verify_matches_backend();
