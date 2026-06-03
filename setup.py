from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext
setup(
    name="hydra_cpp",
    ext_modules=[Pybind11Extension("hydra_cpp", ["cpp/bindings.cpp"])],
    cmdclass={"build_ext": build_ext},
)
