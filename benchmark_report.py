import torch, time, os
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from kernels import fast_fused_norm
import hydra_cpp

# 1. SETUP - Scaling up to 1.1B to show the Hydra Advantage
MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True, 
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4"
)

def patch_model(model):
    from transformers.models.llama.modeling_llama import LlamaRMSNorm
    def patched_forward(instance, x):
        # Fusing RMSNorm + Residual Zero-Init
        return fast_fused_norm(x, torch.zeros_like(x), instance.weight, instance.variance_epsilon)
    for m in model.modules():
        if isinstance(m, LlamaRMSNorm): 
            m.forward = patched_forward.__get__(m, LlamaRMSNorm)
    return model

print("\n--- LOADING MODELS (This may take a minute for 1.1B) ---")
model_base = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb_cfg, device_map="auto")
model_hydra = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb_cfg, device_map="auto")
model_hydra = patch_model(model_hydra)

@torch.inference_mode()
def run_bench(model, name):
    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    # 1. WARMUP (Crucial for Triton to compile)
    for _ in range(3):
        _ = model.generate(**inputs, max_new_tokens=5, pad_token_id=tokenizer.pad_token_id)
    torch.cuda.synchronize()
    
    # 2. ACTUAL MEASUREMENT
    start = time.time()
    tokens_to_gen = 50
    iterations = 5
    for _ in range(iterations):
        _ = model.generate(**inputs, max_new_tokens=tokens_to_gen, pad_token_id=tokenizer.pad_token_id)
    torch.cuda.synchronize()
    
    avg_time = (time.time() - start) / iterations
    tps = tokens_to_gen / avg_time
    print(f"[{name}] Avg Time: {avg_time:.4f}s | Throughput: {tps:.2f} tokens/s")
    return tps

tps_base = run_bench(model_base, "BASELINE PYTORCH")
tps_hydra = run_bench(model_hydra, "HYDRA (TRITON+C++)")

speedup = (tps_hydra / tps_base)
print(f"\n--- FINAL PERFORMANCE RESULT ---")
print(f"Hydra is {speedup:.2f}x faster than Baseline PyTorch on RTX 4050")
print(f"Total Gain: {((speedup-1)*100):.1f}% improvement")
