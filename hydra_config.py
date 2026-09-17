from dataclasses import dataclass
from transformers import AutoConfig

@dataclass
class HydraConfig:
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int
    vocab_size: int
    max_position_embeddings: int
    rms_norm_eps: float
    rope_theta: float
    num_hidden_layers: int
    tie_word_embeddings: bool
    qkv_bias: bool
    o_bias: bool

    @classmethod
    def from_pretrained(cls, model_name: str):
        c = AutoConfig.from_pretrained(model_name)
        return cls(
            hidden_size=c.hidden_size,
            num_attention_heads=c.num_attention_heads,
            num_key_value_heads=getattr(c, "num_key_value_heads", c.num_attention_heads),
            head_dim=getattr(c, "head_dim", None) or c.hidden_size // c.num_attention_heads,
            intermediate_size=getattr(c, "intermediate_size", c.hidden_size * 4),
            vocab_size=c.vocab_size,
            max_position_embeddings=getattr(c, "max_position_embeddings", 4096),
            rms_norm_eps=getattr(c, "rms_norm_eps", 1e-6),
            rope_theta=getattr(c, "rope_theta", 10000.0),
            num_hidden_layers=getattr(c, "num_hidden_layers", getattr(c, "num_layer", 32)),
            tie_word_embeddings=getattr(c, "tie_word_embeddings", True),
            qkv_bias=False,
            o_bias=False,
        )
