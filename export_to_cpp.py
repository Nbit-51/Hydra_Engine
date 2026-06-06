import torch
import os
from transformers import AutoModelForCausalLM

MODEL_ID  = "./TinyLlama-1.1B-Chat-v1.0-git"
SEQ_LEN   = 8
# TinyLlama-1.1B constants — hardcoded to eliminate ALL dynamic size() calls
BATCH         = 1
NUM_HEADS     = 32
NUM_KV_HEADS  = 4
NUM_KV_GROUPS = NUM_HEADS // NUM_KV_HEADS   # 8
HEAD_DIM      = 64
HALF_HEAD     = HEAD_DIM // 2               # 32

print("=== HYDRA EXPORT - CUDA GRAPH EDITION ===\n")

print("[1/4] Loading model...")
full_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.float16, device_map="cuda"
)
full_model.eval()
device = torch.device("cuda")

print("[2/4] Pre-computing static tensors...")
input_ids = torch.randint(0, 32000, (1, SEQ_LEN), device=device)
pos_ids   = torch.arange(SEQ_LEN, device=device).unsqueeze(0)

rotary = full_model.model.layers[0].self_attn.rotary_emb
with torch.no_grad():
    cos, sin = rotary(full_model.model.embed_tokens(input_ids), pos_ids)
print(f"  cos: {cos.shape}  sin: {sin.shape}")

attn_mask = torch.zeros((1, 1, SEQ_LEN, SEQ_LEN), device=device, dtype=torch.float16)
causal    = torch.triu(torch.full((SEQ_LEN, SEQ_LEN), float('-inf'), device=device, dtype=torch.float16), diagonal=1)
attn_mask = attn_mask + causal

print("[3/4] Patching LlamaAttention.forward with fully static version...")

import torch.nn.functional as F

def make_static_attn_forward(layer_idx):
    def static_forward(self, hidden_states, attention_mask=None,
                       position_ids=None, past_key_value=None,
                       output_attentions=False, use_cache=False,
                       cache_position=None, position_embeddings=None, **kwargs):

        # Project Q K V — shapes known statically
        q = self.q_proj(hidden_states)  # [1, seq, num_heads*head_dim]
        k = self.k_proj(hidden_states)  # [1, seq, num_kv_heads*head_dim]
        v = self.v_proj(hidden_states)  # [1, seq, num_kv_heads*head_dim]

        # Reshape with HARDCODED constants — no .size() calls
        q = q.view(BATCH, SEQ_LEN, NUM_HEADS,    HEAD_DIM).transpose(1, 2)  # [1,32,8,64]
        k = k.view(BATCH, SEQ_LEN, NUM_KV_HEADS, HEAD_DIM).transpose(1, 2)  # [1,4,8,64]
        v = v.view(BATCH, SEQ_LEN, NUM_KV_HEADS, HEAD_DIM).transpose(1, 2)  # [1,4,8,64]

        # Apply RoPE using pre-computed cos/sin (passed as position_embeddings)
        cos, sin = position_embeddings  # both [1, seq, head_dim]
        cos = cos.unsqueeze(1)  # [1,1,seq,head_dim]
        sin = sin.unsqueeze(1)  # [1,1,seq,head_dim]

        # rotate_half with HARDCODED slice — no dynamic shape
        def rotate(x):
            x1 = x[..., :HALF_HEAD]
            x2 = x[..., HALF_HEAD:]
            return torch.cat((-x2, x1), dim=-1)

        q = (q * cos) + (rotate(q) * sin)
        k = (k * cos) + (rotate(k) * sin)

        # GQA: repeat k,v with HARDCODED n_rep — no dynamic reshape
        # [1, 4, seq, 64] -> [1, 32, seq, 64]
        k = k.unsqueeze(2).expand(BATCH, NUM_KV_HEADS, NUM_KV_GROUPS, SEQ_LEN, HEAD_DIM)
        k = k.reshape(BATCH, NUM_HEADS, SEQ_LEN, HEAD_DIM)
        v = v.unsqueeze(2).expand(BATCH, NUM_KV_HEADS, NUM_KV_GROUPS, SEQ_LEN, HEAD_DIM)
        v = v.reshape(BATCH, NUM_HEADS, SEQ_LEN, HEAD_DIM)

        # Scaled dot-product attention
        scale  = HEAD_DIM ** -0.5
        scores = torch.matmul(q, k.transpose(2, 3)) * scale  # [1,32,8,8]
        if attention_mask is not None:
            scores = scores + attention_mask
        scores = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        out    = torch.matmul(scores, v)  # [1,32,8,64]

        # Merge heads — HARDCODED output shape
        out = out.transpose(1, 2).reshape(BATCH, SEQ_LEN, NUM_HEADS * HEAD_DIM)
        out = self.o_proj(out)

        return (out, None, None)

    return static_forward

# Patch every attention layer
from transformers.models.llama.modeling_llama import LlamaAttention
for i, layer in enumerate(full_model.model.layers):
    layer.self_attn.forward = make_static_attn_forward(i).__get__(
        layer.self_attn, type(layer.self_attn)
    )

print(f"  Patched {len(full_model.model.layers)} attention layers")

print("[4/4] Building wrapper + tracing...")

class FullyStaticWrapper(torch.nn.Module):
    def __init__(self, full_model, cos, sin, attn_mask):
        super().__init__()
        self.embed_tokens = full_model.model.embed_tokens
        self.layers       = full_model.model.layers
        self.norm         = full_model.model.norm
        self.lm_head      = full_model.lm_head
        self.register_buffer("cos",       cos)
        self.register_buffer("sin",       sin)
        self.register_buffer("attn_mask", attn_mask)

    def forward(self, input_ids):
        hidden = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden = layer(
                hidden,
                attention_mask=self.attn_mask,
                position_ids=None,
                position_embeddings=(self.cos, self.sin),
                past_key_value=None,
                output_attentions=False,
                use_cache=False,
            )[0]
        hidden = self.norm(hidden)
        return self.lm_head(hidden)

wrapper = FullyStaticWrapper(full_model, cos, sin, attn_mask)
wrapper.eval()

with torch.no_grad():
    for _ in range(3):
        wrapper(input_ids)
    torch.cuda.synchronize()

    traced = torch.jit.trace(wrapper, input_ids, strict=False)
    traced = torch.jit.freeze(traced)

    out = traced(input_ids)
    print(f"  Output: {out.shape}  dtype: {out.dtype}")
    assert out.shape == (1, SEQ_LEN, 32000)

graph_str = str(traced.graph)
bad_ops   = ["prim::NumToTensor", "aten::Int(", "floor_divide"]
found     = [op for op in bad_ops if op in graph_str]
if found:
    print(f"  [WARN] Still present: {found}")
else:
    print("  [OK] ZERO dynamic ops — CUDA graph WILL work!")

traced.save("hydra_1_1B.pt")
size_gb = os.path.getsize("hydra_1_1B.pt") / 1e9
print(f"  Saved: hydra_1_1B.pt ({size_gb:.2f} GB)")
