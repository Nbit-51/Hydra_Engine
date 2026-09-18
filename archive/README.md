# Historical Triton Path

This directory contains the earlier TinyLlama-specific Python benchmark and Triton RMSNorm experiment. It is retained for reference and isolated benchmarking only.

It is not imported by `hydra_model.py`, serialized into `hydra_model.pt`, or called by `cpp/engine.cpp`. Its results must not be attributed to the current LibTorch runtime.

The Triton kernel can become an active optimization only after it is integrated through a native/AOT custom operator that remains callable from the exported TorchScript model and the C++ process. Until then, changes here should be evaluated independently.

To install the historical benchmark dependencies:

```bash
python3 -m pip install -r requirements.txt -r archive/requirements.txt
```

Run the archived benchmark from this directory so its local kernel import resolves:

```bash
cd archive
python3 benchmark_report.py
```

The scripts are historical and may require adaptation for current Transformers versions.
