import unittest

import torch

from hydra_config import HydraConfig
from hydra_model import HydraModelForCausalLM


def tiny_config() -> HydraConfig:
    return HydraConfig(
        hidden_size=16,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        intermediate_size=32,
        vocab_size=32,
        max_position_embeddings=16,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        num_hidden_layers=2,
        tie_word_embeddings=False,
        qkv_bias=False,
        o_bias=False,
    )


class HydraModelSmokeTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.model = HydraModelForCausalLM(tiny_config()).eval()

    def test_prefill_and_decode(self) -> None:
        input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
        positions = torch.arange(3, dtype=torch.long).view(1, 3)

        with torch.no_grad():
            prefill_logits = self.model(input_ids, positions, 3)
            decode_logits = self.model(
                torch.tensor([[4]], dtype=torch.long),
                torch.tensor([[3]], dtype=torch.long),
                4,
            )

        self.assertEqual(tuple(prefill_logits.shape), (1, 3, 32))
        self.assertEqual(tuple(decode_logits.shape), (1, 1, 32))
        self.assertTrue(torch.isfinite(prefill_logits).all().item())
        self.assertTrue(torch.isfinite(decode_logits).all().item())

        first_cache = self.model.model.layers[0].self_attn.k_cache
        self.assertGreater(torch.count_nonzero(first_cache[:, :, :4, :]).item(), 0)

    def test_cache_reset(self) -> None:
        with torch.no_grad():
            self.model(
                torch.tensor([[1]], dtype=torch.long),
                torch.tensor([[0]], dtype=torch.long),
                1,
            )

        self.model.reset_cache()
        for layer in self.model.model.layers:
            self.assertEqual(torch.count_nonzero(layer.self_attn.k_cache).item(), 0)
            self.assertEqual(torch.count_nonzero(layer.self_attn.v_cache).item(), 0)

    def test_torchscript_forward(self) -> None:
        scripted = torch.jit.script(self.model)
        with torch.no_grad():
            logits = scripted(
                torch.tensor([[1, 2]], dtype=torch.long),
                torch.tensor([[0, 1]], dtype=torch.long),
                2,
            )

        self.assertEqual(tuple(logits.shape), (1, 2, 32))
        self.assertTrue(torch.isfinite(logits).all().item())


if __name__ == "__main__":
    unittest.main()

