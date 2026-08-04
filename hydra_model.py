import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (self.weight * x).to(input_dtype)

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

class HydraAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_key_value_heads
        self.n_rep = self.num_heads // self.num_kv_heads
        self.head_dim = cfg.head_dim
        self.max_len = 4096

        self.q_proj = nn.Linear(cfg.hidden_size, self.num_heads * self.head_dim, bias=cfg.qkv_bias)
        self.k_proj = nn.Linear(cfg.hidden_size, self.num_kv_heads * self.head_dim, bias=cfg.qkv_bias)
        self.v_proj = nn.Linear(cfg.hidden_size, self.num_kv_heads * self.head_dim, bias=cfg.qkv_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, cfg.hidden_size, bias=cfg.o_bias)

        self.register_buffer("k_cache", torch.zeros(1, self.num_kv_heads, self.max_len, self.head_dim))
        self.register_buffer("v_cache", torch.zeros(1, self.num_kv_heads, self.max_len, self.head_dim))

        inv_freq = 1.0 / (cfg.rope_theta ** (torch.arange(0, self.head_dim, 2, dtype=torch.int64).float() / self.head_dim))
        t = torch.arange(self.max_len).float()
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cache", emb.cos(), persistent=False)
        self.register_buffer("sin_cache", emb.sin(), persistent=False)

    def reset_cache(self):
        self.k_cache.zero_()
        self.v_cache.zero_()

    def forward(self, hidden: torch.Tensor, position_ids: torch.Tensor, seq_len: int) -> torch.Tensor:
        B = hidden.shape[0]
        S = hidden.shape[1]

        q = self.q_proj(hidden).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)

        cos = self.cos_cache[position_ids].unsqueeze(1)
        sin = self.sin_cache[position_ids].unsqueeze(1)
        q = (q * cos) + (rotate_half(q) * sin)
        k = (k * cos) + (rotate_half(k) * sin)

        pos_idx = position_ids.reshape(-1)
        self.k_cache.index_copy_(2, pos_idx, k)
        self.v_cache.index_copy_(2, pos_idx, v)

        # THE FIX: .contiguous() is CRITICAL!
        # Slicing the sequence dimension breaks memory contiguity.
        # If tensors are not contiguous, PyTorch silently disables FlashAttention
        # and falls back to the incredibly slow O(N^2) math backend.
        k_all = self.k_cache[:, :, :seq_len, :].contiguous()
        v_all = self.v_cache[:, :, :seq_len, :].contiguous()

        # Native GQA + FlashAttention
        if S > 1:
            out = F.scaled_dot_product_attention(q, k_all, v_all, is_causal=True)
        else:
            out = F.scaled_dot_product_attention(q, k_all, v_all, is_causal=False)

        out = out.transpose(1, 2).contiguous().view(B, S, -1)
        return self.o_proj(out)

class HydraMLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

class HydraLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.self_attn = HydraAttention(cfg)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.mlp = HydraMLP(cfg)

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor, seq_len: int) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x), position_ids, seq_len)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x

class HydraBackbone(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList([HydraLayer(cfg) for _ in range(cfg.num_hidden_layers)])
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

    def forward(self, input_ids: torch.Tensor, position_ids: torch.Tensor, seq_len: int) -> torch.Tensor:
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x, position_ids, seq_len)
        return self.norm(x)

class HydraModelForCausalLM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.model = HydraBackbone(cfg)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

    def reset_cache(self):
        for layer in self.model.layers:
            layer.self_attn.reset_cache()

    def forward(self, input_ids: torch.Tensor, position_ids: torch.Tensor, seq_len: int) -> torch.Tensor:
        return self.lm_head(self.model(input_ids, position_ids, seq_len))
