import sys
print("Python version:", sys.version)

try:
    import torch
    print("Torch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
except ImportError:
    print("Torch not installed")

try:
    import triton
    print("Triton version:", triton.__version__)
except ImportError:
    print("Triton not installed")

try:
    import transformers
    print("Transformers version:", transformers.__version__)
except ImportError:
    print("Transformers not installed")
