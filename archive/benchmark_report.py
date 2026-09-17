import torch
import os
import gc
import shutil

system_ptxas = shutil.which("ptxas")
if system_ptxas:
    os.environ["TRITON_PTXAS_PATH"] = system_ptxas
    print(f"[INIT] Using ptxas: {system_ptxas}")

from transformers import AutoModelForCausalLM, AutoTokenizer
from kernels import fast_fused_norm, fast_rms_norm

MODEL_ID = "./TinyLlama-1.1B-Chat-v1.0-git"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# TinyLlama constants
BATCH         = 1
NUM_HEADS     = 32
NUM_KV_HEADS  = 4
NUM_KV_GROUPS = 8
HEAD_DIM      = 64
HALF_HEAD     = 32

def patch_model(model):
    from transformers.models.llama.modeling_llama import LlamaDecoderLayer
    import torch.nn.functional as F

    # Pre-compute static RoPE cos/sin for seq_len=1 (decode step)
    # generate() calls the model one token at a time after prefill
    # We patch at decoder layer level and handle variable seq_len
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
        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[0]

        residual = hidden_states

        # 1. Input layernorm
        hidden_states = fast_rms_norm(
            hidden_states,
            self.input_layernorm.weight,
            self.input_layernorm.variance_epsilon
        )

        # 2. Self attention — pass everything through, let it handle KV cache
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

        attn_hidden = attn_outputs[0]
        if isinstance(attn_hidden, tuple):
            attn_hidden = attn_hidden[0]

        # 3. Fused post-attention norm + residual
        normalized_output = fast_fused_norm(
            attn_hidden,
            residual,
            self.post_attention_layernorm.weight,
            self.post_attention_layernorm.variance_epsilon
        )

        # 4. MLP
        hidden_states = self.mlp(normalized_output)

        # 5. Final residual
        hidden_states = hidden_states + residual

        # Return exactly what transformers expects
        # attn_outputs[1:] contains (attn_weights?, present_kv?)
        outputs = (hidden_states,) + attn_outputs[1:]
        return outputs

    for m in model.modules():
        if isinstance(m, LlamaDecoderLayer):
            m.forward = patched_forward.__get__(m, LlamaDecoderLayer)

    # CRITICAL FIX: patch the LlamaModel.forward to disable legacy cache
    # 'to_legacy_cache' is called when use_cache=True returns a DynamicCache
    # We force the model to use the new cache format by patching _update_causal_mask
    original_forward = model.model.forward

    def patched_model_forward(self_model, *args, **kwargs):
        # Force use_cache=False to avoid DynamicCache.to_legacy_cache() crash
        kwargs['use_cache'] = False
        result = original_forward(*args, **kwargs)
        return result

    import types
    model.model.forward = types.MethodType(patched_model_forward, model.model)

    return model


@torch.inference_mode()
def run_bench(model, name):
    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    print(f"  [{name}] Warming up...")
    for _ in range(5):
        _ = model.generate(**inputs, max_new_tokens=5, pad_token_id=tokenizer.pad_token_id)
    torch.cuda.synchronize()
    print(f"  [{name}] Running benchmark...")

    tokens_to_gen = 50
    iterations = 8

    start_event = torch.cuda.Event(enable_timing=True)
    end_event   = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(iterations):
        _ = model.generate(**inputs, max_new_tokens=tokens_to_gen, pad_token_id=tokenizer.pad_token_id)
    end_event.record()
    torch.cuda.synchronize()

    total_ms = start_event.elapsed_time(end_event)
    avg_time = total_ms / (iterations * 1000)
    tps = tokens_to_gen / avg_time

    print(f"  [{name}] {avg_time:.4f}s | {tps:.2f} tok/s | {total_ms/iterations:.2f}ms/iter")
    return tps


print("\n--- LOADING BASELINE ---")
model_base = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16, device_map="auto")
tps_base = run_bench(model_base, "BASELINE PYTORCH")

print("\nCleaning up...")
del model_base
gc.collect()
torch.cuda.empty_cache()

print("\n--- LOADING HYDRA ---")
model_hydra = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16, device_map="auto")
model_hydra = patch_model(model_hydra)
tps_hydra = run_bench(model_hydra, "HYDRA (TRITON+C++)")

speedup = tps_hydra / tps_base
print(f"\n{'='*55}")
print(f"  Baseline:  {tps_base:.2f} tok/s")
print(f"  Hydra:     {tps_hydra:.2f} tok/s")
print(f"  Speedup:   {speedup:.2f}x  ({((speedup-1)*100):.1f}% faster)")
print(f"{'='*55}")
