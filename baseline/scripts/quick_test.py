"""
Quick test to verify everything is set up correctly.

Usage:
    python quick_test.py
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

print("="*80)
print("BASELINE EVALUATION - QUICK TEST")
print("="*80)

print("\n[1/5] Checking imports...")
try:
    import numpy as np
    import torch
    from rdkit import Chem
    from rdkit.Chem import Descriptors, QED
    print("  ✓ NumPy, PyTorch, RDKit imported successfully")
except ImportError as e:
    print(f"  ✗ Import error: {e}")
    sys.exit(1)

print("\n[2/5] Checking project modules...")
try:
    from lightning_modules import LigandPocketDDPM
    from analysis.molecule_builder import build_molecule
    from utils import write_sdf_file
    print("  ✓ Project modules imported successfully")
except ImportError as e:
    print(f"  ✗ Import error: {e}")
    sys.exit(1)

print("\n[3/5] Checking for checkpoint...")
checkpoint_path = Path("checkpoints/crossdocked_fullatom_cond.ckpt")
if checkpoint_path.exists():
    print(f"  ✓ Checkpoint found: {checkpoint_path}")
    print(f"    Size: {checkpoint_path.stat().st_size / 1024 / 1024:.1f} MB")
else:
    print(f"  ✗ Checkpoint not found: {checkpoint_path}")
    print("    Download from: https://zenodo.org/record/8183747")
    print("    Or check if you have it elsewhere")

print("\n[4/5] Checking for test data...")
test_paths = [
    "data/processed_crossdock_noH_full/test.npz",
    "data/processed_crossdock_noH_full_temp/test.npz",
]
test_found = False
for test_path in test_paths:
    if Path(test_path).exists():
        print(f"  ✓ Test data found: {test_path}")
        test_found = True
        break

if not test_found:
    print(f"  ✗ Test data not found in:")
    for p in test_paths:
        print(f"    - {p}")

print("\n[5/5] Checking GPU availability...")
if torch.cuda.is_available():
    print(f"  ✓ CUDA available")
    print(f"    Device: {torch.cuda.get_device_name(0)}")
    print(f"    Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print("  ⚠ CUDA not available - will use CPU (slower)")

print("\n" + "="*80)
print("TEST SUMMARY")
print("="*80)

if checkpoint_path.exists() and test_found:
    print("\n✓ ALL CHECKS PASSED!")
    print("\nYou can now run the baseline evaluation:")
    print("  cd baseline/scripts")
    print("  bash run_all.sh")
else:
    print("\n⚠ SOME CHECKS FAILED")
    print("\nPlease fix the issues above before running baseline evaluation.")
    if not checkpoint_path.exists():
        print("\n1. Download baseline checkpoint")
    if not test_found:
        print("\n2. Ensure test data is available")

print("\n" + "="*80)
