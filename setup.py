import sys
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

# ============================================================
# AGGRESSIVE COMPILER FLAGS FOR ZERO-LATENCY C++ EXTENSION
# ============================================================

extra_compile_args = []
extra_link_args = []

if sys.platform == "win32":
    extra_compile_args = [
        "/O2", "/Ob2", "/Oi", "/Ot",   # Maximum optimization + intrinsics
        "/arch:AVX2",                     # AVX2 SIMD
        "/fp:fast",                       # Fast floating point
        "/GS-",                           # Disable security checks (speed)
        "/GL",                            # Whole program optimization
    ]
    extra_link_args = ["/LTCG"]           # Link-time code generation
else:
    extra_compile_args = [
        "-O3",                            # Maximum optimization
        "-march=native",                  # CPU-specific tuning
        "-mavx2", "-mfma",               # AVX2 + FMA instructions
        "-mbmi", "-mbmi2",               # Bit manipulation (for CTZ)
        "-ffast-math",                    # Fast math (IEEE non-strict)
        "-funroll-loops",                 # Loop unrolling
        "-fprefetch-loop-arrays",         # Auto-prefetch in loops
        "-fomit-frame-pointer",           # Free up a register
        "-flto",                          # Link-time optimization
    ]
    extra_link_args = ["-flto"]

setup(
    name="hydra_cpp",
    ext_modules=[
        CppExtension(
            name="hydra_cpp",
            sources=[
                "cpp/bindings.cpp",
                "cpp/sampling.cpp",
                "cpp/extension.cpp"
            ],
            extra_compile_args={"cxx": extra_compile_args},
            extra_link_args=extra_link_args,
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
