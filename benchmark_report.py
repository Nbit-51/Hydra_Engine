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
        # ============================================================
        # SAFETY: In newer transformers (>=4.45), the framework's
        # output_capturing / modeling_layers wrapper may pass the
        # previous layer's full tuple output as `hidden_states`.
        # We must unwrap it to get the raw tensor.
        # ============================================================
        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[0]
        
        # 1. Input Layernorm (uses Triton RMSNorm; zero residual = pure norm)
        residual = hidden_states
        hidden_states = fast_fused_norm(
            hidden_states, 
            torch.zeros_like(hidden_states), 
            self.input_layernorm.weight, 
            self.input_layernorm.variance_epsilon
        )
        
        # 2. Self Attention
        # Capture ALL outputs dynamically — the number of returned values
        # varies across transformers versions:
        #   Old: (hidden_states, attn_weights, present_key_value)
        #   New: (hidden_states, present_key_value)
        attn_outputs = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs
        )
        
        # The first element is always the hidden states tensor
        hidden_states = attn_outputs[0]
        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[0]
        
        # 3. Fused Post-Attention RMSNorm + Residual Addition
        # This kernel computes: residual = hidden_states + residual (in-place)
        # and returns: RMSNorm(hidden_states + residual)
        normalized_output = fast_fused_norm(
            hidden_states,
            residual,
            self.post_attention_layernorm.weight,
            self.post_attention_layernorm.variance_epsilon
        )
        
        # 4. MLP
        hidden_states = self.mlp(normalized_output)
        
        # 5. Final residual addition (residual was updated in-place by the kernel)
        hidden_states = hidden_states + residual
        
        # ============================================================
        # BUILD OUTPUT TUPLE
        # The LlamaModel.forward() loop expects:
        #   layer_outputs[0] = hidden_states (always)
        #   layer_outputs[1] = attn_weights  (if output_attentions)  
        #   layer_outputs[-1] = present_key_value (if use_cache)
        #
        # After the loop, LlamaModel does:
        #   hidden_states = self.norm(hidden_states)
        # where hidden_states = layer_outputs[0]
        # So outputs[0] MUST be a raw tensor, never a tuple.
        # ============================================================
        outputs = (hidden_states,)
        
        if output_attentions:
            # Pull attention weights from the original attn output if available
            self_attn_weights = attn_outputs[1] if len(attn_outputs) > 1 else None
            outputs += (self_attn_weights,)
        
        if use_cache:
            # In newer transformers: present_key_value is attn_outputs[1]
            # In older transformers: present_key_value is attn_outputs[2]
            if output_attentions and len(attn_outputs) > 2:
                present_kv = attn_outputs[2]
            elif not output_attentions and len(attn_outputs) > 1:
                present_kv = attn_outputs[1]
            else:
                present_kv = None
            outputs += (present_kv,)
            
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
print(f"\n{'='*50}")
print(f"HYDRA ENGINE - PERFORMANCE REPORT")
print(f"{'='*50}")
print(f"Baseline PyTorch:  {tps_base:.2f} tokens/s")
print(f"Hydra (Triton+C++): {tps_hydra:.2f} tokens/s")
print(f"Speedup:           {speedup:.2f}x Faster")
print(f"Total Gain:        {((speedup-1)*100):.1f}% improvement")
print(f"{'='*50}")
