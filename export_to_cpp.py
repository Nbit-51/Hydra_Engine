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

    hf = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
    sd = hf.state_dict()
    del hf

    cfg.qkv_bias = "model.layers.0.self_attn.q_proj.bias" in sd
    cfg.o_bias = "model.layers.0.self_attn.o_proj.bias" in sd
    if "lm_head.weight" not in sd:
        sd["lm_head.weight"] = sd["model.embed_tokens.weight"].clone()

    model = HydraModelForCausalLM(cfg).to(dtype).eval()
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [k for k in missing if not k.endswith("k_cache") and not k.endswith("v_cache")]
    assert not missing, f"missing: {missing}"

    device = torch.device("cuda")
    model = model.to(device)
    ids = torch.randint(0, cfg.vocab_size, (1, 7), device=device)
    pos = torch.arange(7, device=device).view(1, 7)

    print("Verifying on CUDA...")
    with torch.no_grad():
        ref = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, device_map="cuda")
        ref_logits = ref(input_ids=ids).logits[:, -1, :]
        del ref
        model.reset_cache()
        our_logits = model(ids, pos, ids.shape[1])[:, -1, :]
        print(f"max abs logit diff: {(ref_logits - our_logits).abs().max().item()}")

    print("Compiling with torch.jit.script...")
    with torch.no_grad():
        try:
            sm = torch.jit.script(model)
        except Exception as e:
            print(f"script failed ({e}), tracing instead")
            sm = torch.jit.trace(model, (ids, pos, ids.shape[1]), strict=False)
    sm.save(output_path)
    print(f"Saved {output_path}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_id", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--output", default="hydra_model.pt")
    a = p.parse_args()
    export_model(a.model_id, a.output)
