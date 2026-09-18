#include "simd_verify.h"

#include <cstdint>
#include <iostream>
#include <vector>

static bool expect_prefix(
    const std::vector<int64_t>& draft,
    const std::vector<int64_t>& target,
    int expected
) {
    const int actual = verify_matches_simd(
        draft.data(),
        target.data(),
        static_cast<int>(draft.size())
    );
    if (actual == expected) return true;

    std::cerr << "expected " << expected << " accepted token(s), got " << actual
              << std::endl;
    return false;
}

int main() {
    bool ok = true;
    ok &= expect_prefix({}, {}, 0);
    ok &= expect_prefix({1}, {1}, 1);
    ok &= expect_prefix({1}, {2}, 0);
    ok &= expect_prefix({1, 2, 3, 4}, {1, 2, 3, 4}, 4);
    ok &= expect_prefix({1, 2, 3, 4}, {9, 2, 3, 4}, 0);
    ok &= expect_prefix({1, 2, 3, 4}, {1, 2, 9, 4}, 2);
    ok &= expect_prefix({1, 2, 3, 4}, {1, 2, 3, 9}, 3);
    ok &= expect_prefix({1, 2, 3, 4, 5}, {1, 2, 3, 4, 5}, 5);
    ok &= expect_prefix({1, 2, 3, 4, 5}, {1, 2, 3, 4, 9}, 4);
    ok &= expect_prefix(
        {1, 2, 3, 4, 5, 6, 7, 8, 9},
        {1, 2, 3, 4, 5, 6, 9, 8, 9},
        6
    );

    if (!ok) return 1;
    std::cout << "SIMD verification tests passed (backend: "
              << verify_matches_backend() << ")" << std::endl;
    return 0;
}
