import torch
import time
from transformers import AutoModelForCausalLM

print("Loading pure HuggingFace model...")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", torch_dtype=torch.float16, device_map="cuda")
model.eval()

input_ids = torch.ones((1, 1), dtype=torch.long, device="cuda")
past_key_values = None

print("Warming up...")
with torch.no_grad():
    for _ in range(10):
        out = model(input_ids, past_key_values=past_key_values, use_cache=True)
        past_key_values = out.past_key_values

print("Profiling pure PyTorch autoregressive decode (50 tokens)...")
torch.cuda.synchronize()
start = time.time()
with torch.no_grad():
    for _ in range(50):
        out = model(input_ids, past_key_values=past_key_values, use_cache=True)
        past_key_values = out.past_key_values
torch.cuda.synchronize()
end = time.time()

total_ms = (end - start) * 1000
print(f"Pure PyTorch decode: {total_ms/50:.2f} ms per token")
print(f"Pure PyTorch speed:  {50000/total_ms:.2f} tok/s")
