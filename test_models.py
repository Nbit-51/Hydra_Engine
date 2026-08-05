#!/usr/bin/env python3
"""
Hydra Engine - Multi-Model Validation Harness
Tests the engine across different model families and sizes
"""
import subprocess
import sys

# Test cases: (model_id, prompt_text)
TEST_CASES = [
    # Base models (completion)
    ("Qwen/Qwen2.5-0.5B", "The capital of France is"),
    
    # Add more models here as you test:
    # ("meta-llama/Llama-3.2-1B", "Once upon a time"),
    # ("microsoft/Phi-3.5-mini-instruct", "Explain quantum computing"),
    # ("google/gemma-2b", "The future of AI is"),
    # ("mistralai/Mistral-7B-v0.1", "Write a haiku about"),
]

def run_cmd(cmd):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)
    return result.returncode == 0, result.stdout, result.stderr

def encode_prompt(model_id, text):
    code = f'''
from transformers import AutoTokenizer
t = AutoTokenizer.from_pretrained("{model_id}")
ids = t.encode("{text}", add_special_tokens=True)
print(",".join(map(str, ids)))
'''
    ok, out, err = run_cmd(f'python -c \'{code}\'')
    return out.strip() if ok else None

def test_model(model_id, prompt_text):
    print(f"\n{'='*70}")
    print(f"Testing: {model_id}")
    print(f"Prompt: '{prompt_text}'")
    print('='*70)
    
    # Encode
    print("[1/4] Encoding prompt...")
    prompt_ids = encode_prompt(model_id, prompt_text)
    if not prompt_ids:
        print("✗ Failed to encode")
        return False
    print(f"✓ IDs: {prompt_ids}")
    
    # Export
    print("\n[2/4] Exporting model...")
    ok, out, err = run_cmd(f'python export_to_cpp.py --model_id "{model_id}"')
    if not ok:
        print(f"✗ Export failed:\n{err}")
        return False
    print("✓ Export complete")
    for line in out.split('\n'):
        if 'max abs logit diff' in line:
            print(f"  {line.strip()}")
    
    # Build
    print("\n[3/4] Building engine...")
    ok, out, err = run_cmd('bash -c "cd ~/build && make -j$(nproc)"')
    if not ok:
        print(f"✗ Build failed:\n{err}")
        return False
    print("✓ Build complete")
    
    # Run
    print("\n[4/4] Running inference (64 tokens)...")
    ok, out, err = run_cmd(f'~/build/hydra_native hydra_model.pt vocab.bin "{prompt_ids}" 64')
    if not ok:
        print(f"✗ Inference failed:\n{err}")
        return False
    
    print("-"*70)
    for line in out.split('\n'):
        if '[decode]' in line:
            print(line)
    print("✓ Test complete")
    return True

print("="*70)
print("HYDRA ENGINE - MULTI-MODEL VALIDATION")
print("="*70)

results = []
for model_id, prompt in TEST_CASES:
    success = test_model(model_id, prompt)
    results.append((model_id, success))

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
for model_id, success in results:
    print(f"{'✓ PASS' if success else '✗ FAIL'}: {model_id}")

passed = sum(1 for _, s in results if s)
print(f"\n{passed}/{len(results)} models passed")
