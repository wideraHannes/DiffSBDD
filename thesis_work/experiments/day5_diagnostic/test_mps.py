import torch

# Test MPS availability
if torch.backends.mps.is_available():
    device = torch.device("mps")
    x = torch.randn(100, 100, device=device)
    y = x @ x.T
    print(f"✓ MPS working! Tensor on {y.device}")
    print(f"  PyTorch version: {torch.__version__}")
else:
    print("✗ MPS not available")
    print(f"  PyTorch version: {torch.__version__}")
    if hasattr(torch.backends, 'mps'):
        print(f"  MPS backend exists but not available")
    else:
        print(f"  MPS backend not found (need PyTorch >=1.12 for M1)")
