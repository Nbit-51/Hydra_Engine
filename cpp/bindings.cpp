#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "sampling.cpp"

namespace py = pybind11;
PYBIND11_MODULE(hydra_cpp, m) {
    m.def("top_k_filter", &top_k_filter, "Top-K filtering in C++");
}
