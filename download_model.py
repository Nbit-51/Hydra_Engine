from huggingface_hub import snapshot_download
import os

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
local_dir = "./TinyLlama-1.1B-Chat-v1.0"

print(f"Downloading {model_id} to {local_dir}...")
snapshot_download(
    repo_id=model_id,
    local_dir=local_dir,
    local_dir_use_symlinks=False,  # Important for Windows-to-WSL compat
    ignore_patterns=["*.msgpack", "*.h5", "*.ot", "onnx/*"] # Only get what we need
)
print("Download complete!")
