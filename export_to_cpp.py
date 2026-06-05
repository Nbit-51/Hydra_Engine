import torch
from transformers import AutoModelForCausalLM
import gc

print("Loading 1.1B Model with memory optimization...")

# Load with aggressive memory management
hf_model = AutoModelForCausalLM.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    torch_dtype=torch.float16,
    device_map="cuda",
    local_files_only=True,
    low_cpu_mem_usage=True,  # Reduces CPU memory during load
    use_safetensors=True  # Faster and safer loading
)
hf_model.eval()

# Disable gradient computation globally
torch.set_grad_enabled(False)

# Optimize model for inference
if hasattr(torch.cuda, 'empty_cache'):
    torch.cuda.empty_cache()

class OptimizedModelWrapper(torch.nn.Module):
    """Zero-overhead wrapper for TorchScript export"""
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    @torch.jit.export
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # Direct logits extraction with minimal overhead
        # use_cache=False saves memory, return_dict=False is faster
        out = self.model(input_ids, use_cache=False, return_dict=False)
        # Extract only logits (first element of tuple)
        return out[0] if isinstance(out, tuple) else out.logits

clean_model = OptimizedModelWrapper(hf_model)
clean_model.eval()

# Use smaller dummy input to reduce trace size
dummy_input = torch.randint(0, 32000, (1, 4), device="cuda", dtype=torch.long)

print("Compiling to optimized TorchScript with graph optimization...")
with torch.inference_mode(), torch.jit.optimized_execution(True):
    # Warmup to compile CUDA kernels
    for _ in range(3):
        _ = clean_model(dummy_input)
    torch.cuda.synchronize()
    
    # Trace with strict=False for flexibility
    traced_model = torch.jit.trace(
        clean_model, 
        dummy_input, 
        strict=False,
        check_trace=False  # Skip validation for speed
    )
    
    # Apply graph optimizations
    traced_model = torch.jit.optimize_for_inference(traced_model)

# Save with compression
print("Saving optimized model...")
traced_model.save("hydra_1_1B.pt", _use_new_zipfile_serialization=True)

# Aggressive cleanup
del hf_model, clean_model, dummy_input
gc.collect()
torch.cuda.empty_cache()

print("✓ Export complete! Saved as hydra_1_1B.pt (optimized)")
print("  - Memory footprint minimized")
print("  - Graph optimizations applied")
print("  - Ready for ultra-fast C++ inference")
