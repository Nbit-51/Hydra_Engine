# Hydra Engine — Session Handover

**Date:** 2026-08-05
**Branch:** feature/general-purpose-engine (not yet pushed to origin — origin only has `main`, last updated 2 months ago)
**Last commit before this doc:** b48dd67 — "perf: eliminate per-decode-step CPU-GPU sync from cur_len computation"

## TL;DR

Started this session at ~15 tok/s on Qwen2.5-0.5B (a regression from TinyLlama's known-good ~90 tok/s on `main`). Found and fixed two real, confirmed bugs. Currently at **58.3 tok/s on Qwen2.5-0.5B** (3.9x improvement from session start).

**Real baseline comparison (measured this session, `profile_baseline.py`):** pure HuggingFace eager PyTorch on the same GPU/model = **39.86 tok/s**. Hydra Engine = **58.3 tok/s = 1.46x over stock PyTorch**, honest and current — use this over the old stale "1.68x" TinyLlama-only claim in the README until TinyLlama is re-verified with a fresh baseline too.

**This branch has never been pushed.** Everything below only exists locally in this WSL checkout until pushed.

## Environment

- RTX 4050 Laptop GPU (6GB VRAM, Ada Lovelace, compute capability 8.9), WSL2 Ubuntu, project at `/mnt/c/Users/navaneeth/Hydra_Engine` (Windows-mounted 9P bridge — slow for large I/O)
- venv: `source venv/bin/activate` — confirm `(venv) root@LAPTOP-977VC2OP:/mnt/c/Users/navaneeth/Hydra_Engine#` prefix before running anything
- torch 2.13.0+cu130, nvcc 12.8, cuDNN 9.20. `libcudart.so.13 vs .12` linker warning is harmless.
- Build: CMake + LibTorch → `~/build/hydra_native`

## Bugs found and fixed this session

### Fix 1: KV cache not sliced to actual length during decode

`hydra_model.py`, `HydraAttention.forward()` decode branch. Attention ran over the *entire* `k_cache`/`v_cache` (sized to `max_position_embeddings`) with a mask, instead of being sliced — `scaled_dot_product_attention` with `attn_mask=` still does dense compute over the full length passed in. Cost scaled with max context window, not actual conversation length.

Why Qwen and not TinyLlama: TinyLlama `max_position_embeddings=2048` vs Qwen2.5-0.5B `32768` — 16x larger, so Qwen did ~16x more attention compute per decode step despite being the smaller model.

Fix: slice to `[:cur_len]`, drop the manual mask, `is_causal=False`.

**Result: 14.8 → 48.7 tok/s (3.3x).**

Note: old code's "CUDA Graph compatibility" comment was aspirational — no CUDA Graph capture exists anywhere in the codebase (confirmed via grep + `torch.jit.load(path).graph`). If CUDA Graphs get built later, reconcile via bucketed fixed-length capture, not a revert to always-max-len.

### Fix 2: Unnecessary per-decode-step CPU-GPU sync

`hydra_model.py` (all forward signatures), `cpp/engine.cpp`, `export_to_cpp.py`. Fix 1's first pass computed `cur_len` via `.item()` inside the model — forces a device sync every step. `engine.cpp` already tracks `pos` as a host-side int; threaded `cur_len: int` through all 4 forward signatures instead, sourced from the host loop.

**Result: 48.7 → 58.3 tok/s (+20%). Combined: 14.8 → 58.3 tok/s, 3.9x total.**

Remaining necessary sync: `sample()`'s `.item<int64_t>()` — structurally required, don't remove without a bigger restructure.

## Dead / unwired code — confirmed, don't re-investigate without new evidence

- **`kernels.py`'s Triton fused RMSNorm kernels are NOT in the execution path** — confirmed via `torch.jit.load(path).graph`. Either wire in for real (`torch.library.custom_op` + TorchScript registration) or remove/quarantine — currently misleading.
- **No CUDA Graph capture exists anywhere in `engine.cpp`**, despite comments referencing it.
- **Lesson from this session:** always check `torch.jit.load(path).graph` before debugging kernel/op-level behavior — lost real time investigating Triton/CUDA-Graph interactions that weren't reachable at all.

## Working-directory cleanup done this session

- Archived (git-detected as renames): `RUN_THIS_TEST.sh`, `benchmark_report.py`, `check_env.py`, `download_model.py` → `archive/`
- Deleted: `setup.py`, `cpp/bindings.cpp`, `cpp/extension.cpp`, `cpp/extension.h`, `cpp/sampling.cpp`, `cpp/sampling.h` — this was the PyBind11 extension (min-heap top-k sampling, AVX2 speculative-decode verification). **CONFIRM this was intentional before final commit** — if the plan is native-C++-only decode going forward this is reasonable, but it's real prior work being dropped, worth a deliberate yes/no rather than an accidental commit.
- New: `test_models.py` (multi-model harness — TinyLlama currently commented out in `TEST_CASES`, add back once re-verified), `profile_baseline.py` (pure-PyTorch baseline benchmark — now run, see TL;DR for the number)
- `.bak`/`.bak2` files from this session's patches — safe to delete, already in git history via commits `a43b3b6`/`b48dd67`

## Known infra issue

`/mnt/c` WSL↔Windows 9P bridge is slow for large I/O (git, cmake, big writes). Not the cause of any bug found this session (ruled out for the earlier 15-min hang, which was a slow HF download), but worth moving to native WSL filesystem (`~/hydra_work`) if build times keep dragging.

## Untested / not yet re-verified

- **TinyLlama not re-run since Fix 1 or Fix 2.** Command:
```bash
  python3 export_to_cpp.py --model_id "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
  cd ~/build && make -j$(nproc) && cd /mnt/c/Users/navaneeth/Hydra_Engine
  ~/build/hydra_native hydra_model.pt vocab.bin "1,450,7483,310,3444,338" 64
```
  Also worth getting TinyLlama's `profile_baseline.py`-equivalent number (edit the model_id in that script) for an honest updated speedup claim.
- **`max abs logit diff` fluctuates 0.20–0.47 across runs** — suspected fp16/SDPA nondeterminism, not correctness bug so far (output stayed coherent). Watch if it jumps notably higher.
- **No `nsys` profiling done yet.** Both fixes this session came from log-reading and config diffing, not a profiler trace — worked because the bugs were large-effect. Further gains should be profiler-driven.

## Prioritized next steps

1. Confirm the pybind11-extension deletion is intentional, then commit working directory cleanly.
2. Update README with the honest 1.46x-over-PyTorch-eager number (Qwen2.5-0.5B) + methodology, replacing the stale unexplained 1.68x claim.
3. Re-verify TinyLlama (command above), get its fresh baseline too.
4. Install and run `nsys` — profile the remaining ~17ms/token before choosing between CUDA Graph capture, wiring in the Triton kernel, or something else.
5. Decide on `kernels.py`: wire in or remove.
6. Push `feature/general-purpose-engine` to origin once clean.
7. Broader model-family testing (Mistral, Llama 3.x, Gemma) — expect new architecture surprises, use the same discipline (config diff + traced-graph inspection before assuming a fix).

## Process notes for next session

- Confirm shell/venv context before running commands.
- Check `torch.jit.load(path).graph` before debugging any kernel/op-level theory.
- Profile before further optimization guesses.
- Use scripted Python patches with `assert old in content` guards for multi-line edits, not heredoc pastes into `cat >`.
- Verify a file was actually written before assuming it happened.
