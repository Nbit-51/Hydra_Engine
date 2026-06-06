import torch
from transformers import AutoModelForCausalLM
import gc

print("=== HYDRA MODEL EXPORT - ZERO LATENCY EDITION ===\n")

# ---- 1. LOAD MODEL WITH MINIMAL MEMORY ----
print("[1/4] Loading TinyLlama 1.1B...")

hf_model = AutoModelForCausalLM.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    torch_dtype=torch.float16,
    device_map="cuda",
    local_files_only=True,
    low_cpu_mem_usage=True,
    use_safetensors=True,
)
hf_model.eval()
torch.set_grad_enabled(False)

# ---- 2. WRAP FOR CLEAN TORCHSCRIPT EXPORT ----
print("[2/4] Creating optimized wrapper...")

class HydraExportWrapper(torch.nn.Module):
    """Zero-overhead wrapper that returns only logits tensor."""
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        out = self.model(input_ids, use_cache=False, return_dict=False)
        return out[0] if isinstance(out, tuple) else out.logits

wrapper = HydraExportWrapper(hf_model)
wrapper.eval()

# ---- 3. TRACE WITH OPTIMIZATIONS ----
print("[3/4] Tracing and optimizing graph...")

# Use a small input (reduces trace overhead, model handles dynamic shapes)
dummy = torch.randint(0, 32000, (1, 4), device="cuda", dtype=torch.long)

with torch.inference_mode(), torch.jit.optimized_execution(True):
    # Warmup: compile all CUDA kernels before tracing
    for _ in range(5):
        _ = wrapper(dummy)
    torch.cuda.synchronize()
    
    # Trace the model
    traced = torch.jit.trace(wrapper, dummy, strict=False, check_trace=False)
    
    # Freeze: fold batch norms, eliminate dead code, constant propagation
    traced = torch.jit.freeze(traced)
    
    # Optimize for inference: fuse operations, eliminate overhead
    traced = torch.jit.optimize_for_inference(traced)

# ---- 4. SAVE ----
print("[4/4] Saving optimized model...")
traced.save("hydra_1_1B.pt", _use_new_zipfile_serialization=True)

# Cleanup
del hf_model, wrapper, dummy, traced
gc.collect()
torch.cuda.empty_cache()

print("\n✓ Export complete: hydra_1_1B.pt")
print("  → Graph frozen + optimized for inference")
print("  → Ready for C++ engine (zero Python overhead)")
