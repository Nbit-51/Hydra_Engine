import torch
import time
import os
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from kernels import fast_fused_norm

# 1. SETUP - Scaling up to 1.1B to show the Hydra Advantage
MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None: 
    tokenizer.pad_token = tokenizer.eos_token

bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True, 
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4"
)

def patch_model(model):
    from transformers.models.llama.modeling_llama import LlamaDecoderLayer
    
    def patched_forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        cache_position=None,
        **kwargs
    ):
        # 1. Input Layernorm (uses Triton RMSNorm without residual addition)
        residual = hidden_states
        hidden_states = fast_fused_norm(
            hidden_states, 
            torch.zeros_like(hidden_states), 
            self.input_layernorm.weight, 
            self.input_layernorm.variance_epsilon
        )
        
        # 2. Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs
        )
        
        # 3. Fused Post-Attention RMSNorm + Residual Addition
        # Updates 'residual' in-place to (residual + hidden_states) and returns normalized output
        normalized_output = fast_fused_norm(
            hidden_states,
            residual,
            self.post_attention_layernorm.weight,
            self.post_attention_layernorm.variance_epsilon
        )
        
        # 4. MLP
        hidden_states = self.mlp(normalized_output)
        
        # 5. Final residual addition of the MLP block (residual now contains the sum)
        hidden_states.add_(residual)
        
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)
            
        return outputs

    for m in model.modules():
        if isinstance(m, LlamaDecoderLayer): 
            m.forward = patched_forward.__get__(m, LlamaDecoderLayer)
            
    return model

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

print("\n--- LOADING BASELINE MODEL ---")
model_base = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb_cfg, device_map="auto")
tps_base = run_bench(model_base, "BASELINE PYTORCH")

# Aggressive cleanup of baseline model to free VRAM before loading Hydra
print("\nCleaning up baseline model from memory...")
del model_base
gc.collect()
torch.cuda.empty_cache()

print("\n--- LOADING HYDRA MODEL (with Triton optimizations) ---")
model_hydra = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb_cfg, device_map="auto")
model_hydra = patch_model(model_hydra)
tps_hydra = run_bench(model_hydra, "HYDRA (TRITON+C++)")

speedup = (tps_hydra / tps_base)
print(f"\n--- FINAL PERFORMANCE RESULT ---")
print(f"Hydra is {speedup:.2f}x faster than Baseline PyTorch on RTX 4050")
print(f"Total Gain: {((speedup-1)*100):.1f}% improvement")
