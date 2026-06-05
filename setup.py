import sys
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

extra_compile_args = []
if sys.platform == "win32":
    extra_compile_args = ["/O2", "/Ob2", "/arch:AVX2", "/fp:fast"]
else:
    extra_compile_args = ["-O3", "-march=native", "-mavx2", "-mfma", "-ffast-math"]

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
            extra_compile_args={"cxx": extra_compile_args}
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
