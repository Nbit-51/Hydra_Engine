#include "simd_verify.h"

#include <cstdint>

#if defined(__AVX2__) || defined(_M_AVX2)
#include <immintrin.h>
#define HYDRA_HAS_AVX2 1
#endif

#if defined(_MSC_VER)
#include <intrin.h>
static __forceinline int hydra_ctz(uint32_t value) {
    unsigned long index;
    _BitScanForward(&index, value);
    return static_cast<int>(index);
}
#else
static __attribute__((always_inline)) inline int hydra_ctz(uint32_t value) {
    return __builtin_ctz(value);
}
#endif

int verify_matches_simd(
    const int64_t* HYDRA_RESTRICT draft,
    const int64_t* HYDRA_RESTRICT target,
    int size
) {
    if (size <= 0) return 0;

    int accepted = 0;

#if defined(HYDRA_HAS_AVX2)
    constexpr int width = 4;
    const int vectorized = size / width;

    for (int i = 0; i < vectorized; ++i) {
        const int offset = i * width;
        if (i + 1 < vectorized) {
            _mm_prefetch(reinterpret_cast<const char*>(draft + offset + width), _MM_HINT_T0);
            _mm_prefetch(reinterpret_cast<const char*>(target + offset + width), _MM_HINT_T0);
        }

        const __m256i draft_values =
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(draft + offset));
        const __m256i target_values =
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(target + offset));
        const __m256i matches = _mm256_cmpeq_epi64(draft_values, target_values);
        const uint32_t mask = static_cast<uint32_t>(_mm256_movemask_epi8(matches));

        if (mask == 0xFFFFFFFFu) {
            accepted += width;
            continue;
        }

        const int first_mismatch_byte = hydra_ctz(~mask);
        return accepted + first_mismatch_byte / static_cast<int>(sizeof(int64_t));
    }

    for (int i = vectorized * width; i < size; ++i) {
        if (draft[i] != target[i]) return accepted;
        ++accepted;
    }
#else
    for (int i = 0; i < size; ++i) {
        if (draft[i] != target[i]) return accepted;
        ++accepted;
    }
#endif

    return accepted;
}

const char* verify_matches_backend() {
#if defined(HYDRA_HAS_AVX2)
    return "avx2";
#else
    return "scalar";
#endif
}

