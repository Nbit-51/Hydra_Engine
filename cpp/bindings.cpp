#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>
#include "sampling.h"
#include "extension.h"

namespace py = pybind11;

PYBIND11_MODULE(hydra_cpp, m) {
    m.doc() = "Hydra C++ optimized operations (sampling and verification)";
    
    m.def("top_k_filter", &top_k_filter, 
          "Memory-efficient top-K filtering using heap (O(n log k))",
          py::arg("logits"), 
          py::arg("k"));
    
    m.def("top_k_filter_fast", &top_k_filter_fast,
          "Fast approximate top-K using partitioning (O(n) average)",
          py::arg("logits"),
          py::arg("k"));

    m.def("verify_matches", &verify_matches,
          "Ultra-optimized verification with SIMD and early exit",
          py::arg("draft_ids"),
          py::arg("target_preds"));
}
