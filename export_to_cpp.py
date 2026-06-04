import torch
from transformers import AutoModelForCausalLM

print("Loading 1.1B Model...")
# Load the model normally
hf_model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.float16, device_map="cuda", local_files_only=True)
hf_model.eval()

# Create a clean wrapper that only outputs the raw tensor
class NativeModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids):
        # We let HF use its internal dicts to avoid crashing, 
        # but we ONLY return the raw logits tensor to C++
        out = self.model(input_ids, use_cache=False, return_dict=True)
        return out.logits

clean_model = NativeModelWrapper(hf_model)

dummy_input = torch.randint(0, 32000, (1, 8), device="cuda")

print("Compiling model to Native C++ format (TorchScript)...")
with torch.no_grad():
    traced_model = torch.jit.trace(clean_model, dummy_input, strict=False)

traced_model.save("hydra_1_1B.pt")
print("Export complete! Saved as hydra_1_1B.pt")
