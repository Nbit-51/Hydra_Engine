import torch
import argparse
from transformers import AutoModelForCausalLM
from hydra_config import HydraConfig
from hydra_model import HydraModelForCausalLM

def export_model(model_id, output_path):
    cfg = HydraConfig.from_pretrained(model_id)
    print(f"Config: {cfg.num_hidden_layers} layers, {cfg.num_attention_heads} Q-heads, {cfg.num_key_value_heads} KV-heads")

    dtype = torch.float16
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16

    print("Loading HF weights (loader only, forward is ours)...")
    hf = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
    sd = hf.state_dict()
    del hf

    cfg.qkv_bias = "model.layers.0.self_attn.q_proj.bias" in sd
    cfg.o_bias = "model.layers.0.self_attn.o_proj.bias" in sd
    print(f"Detected biases: QKV={cfg.qkv_bias}, O={cfg.o_bias}")

    if "lm_head.weight" not in sd:
        print("Tied embeddings detected, copying embed weight to lm_head")
        sd["lm_head.weight"] = sd["model.embed_tokens.weight"].clone()

    model = HydraModelForCausalLM(cfg).to(dtype).eval()
    missing, unexpected = model.load_state_dict(sd, strict=False)
    
    missing = [k for k in missing if not k.endswith("k_cache") and not k.endswith("v_cache")]
    print(f"missing keys (excluding cache): {missing}")
    print(f"unexpected keys (ignored): {unexpected}")
    assert not missing, "Weight names do not match"

    print("Verifying logits against HF reference on CUDA...")
    device = torch.device("cuda")
    model = model.to(device)
    
    ids = torch.randint(0, cfg.vocab_size, (1, 7), device=device)
    pos = torch.arange(7, device=device).view(1, 7)
    
    with torch.no_grad():
        ref = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, device_map="cuda")
        ref_logits = ref(input_ids=ids).logits[:, -1, :]
        del ref
        model.reset_cache()
        our_logits = model(ids, pos)[:, -1, :]
        diff = (ref_logits - our_logits).abs().max().item()
        print(f"max abs logit diff: {diff}")

    print("Tracing model with torch.jit.trace (Preserves FlashAttention)...")
    dummy_ids = torch.ones((1, 1), dtype=torch.long, device=device)
    dummy_pos = torch.tensor([[5]], dtype=torch.long, device=device)
    
    with torch.no_grad():
        # Warmup to populate cache buffers before tracing
        model(dummy_ids, dummy_pos)
        sm = torch.jit.trace(model, (dummy_ids, dummy_pos), strict=False)

    sm.save(output_path)
    print(f"Saved {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--output", type=str, default="hydra_model.pt")
    args = parser.parse_args()
    export_model(args.model_id, args.output)
