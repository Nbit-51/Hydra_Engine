import argparse

import torch
from torch import nn


class CaptureFailureModel(nn.Module):
    """Minimal engine-compatible model with a deliberately uncapturable decode."""

    @torch.jit.export
    def reset_cache(self) -> None:
        return None

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        cache_len: int,
        mask_unused_cache: bool,
    ) -> torch.Tensor:
        del cache_len, mask_unused_cache
        host_position = int(position_ids[0, -1].item())
        logits = torch.zeros(
            (input_ids.size(0), input_ids.size(1), 64),
            dtype=torch.float32,
            device=input_ids.device,
        )
        logits[:, :, host_position % 64] = 1.0
        return logits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", nargs="?", default="capture_failure_model.pt")
    args = parser.parse_args()
    torch.jit.script(CaptureFailureModel()).save(args.output)


if __name__ == "__main__":
    main()
