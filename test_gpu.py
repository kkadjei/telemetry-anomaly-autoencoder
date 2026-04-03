# test_gpu.py
# Purpose: verify your GPU is ready for training
# Run this with: python test_gpu.py

import torch
import torch.nn as nn

# ── 1. Check CUDA availability ───────────────────────────────────────────────
# torch.cuda.is_available() returns True if PyTorch can see your RTX 3050
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ── 2. Check GPU name ────────────────────────────────────────────────────────
if device.type == "cuda":
    print(f"GPU name: {torch.cuda.get_device_name(0)}")

# ── 3. Run a small tensor operation on GPU ───────────────────────────────────
x = torch.randn(100, 100).to(device)
y = x @ x.T
print(f"Tensor operation successful. Output shape: {y.shape}")
print(f"Tensor is on: {y.device}")

# ── 4. Check available GPU memory ────────────────────────────────────────────
if device.type == "cuda":
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"Total GPU memory: {total:.1f} GB")

print("\nAll checks passed. Ready for Session 2.")