"""
Deep Analysis of Pocket Representation

Investigates the unexpected 11-dimensional pocket encoding.
"""

import numpy as np
import torch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from constants import dataset_params


def analyze_pocket_encoding():
    """Analyze pocket one-hot encoding in detail"""
    print("="*80)
    print("POCKET ENCODING DEEP DIVE")
    print("="*80)

    # Load data
    data_path = Path("data/processed_crossdock_noH_full_temp/train.npz")
    data = np.load(data_path, allow_pickle=True)

    pocket_one_hot = data['pocket_one_hot']
    pocket_coords = data['pocket_coords']
    pocket_mask = data['pocket_mask']

    print(f"\nData shapes:")
    print(f"  pocket_one_hot: {pocket_one_hot.shape}")
    print(f"  pocket_coords: {pocket_coords.shape}")
    print(f"  pocket_mask: {pocket_mask.shape}")

    # Check first sample
    sample_idx = 0
    sample_mask = (pocket_mask == sample_idx)
    sample_one_hot = pocket_one_hot[sample_mask]
    sample_coords = pocket_coords[sample_mask]

    print(f"\nFirst sample:")
    print(f"  Number of atoms/residues: {len(sample_one_hot)}")
    print(f"  One-hot shape: {sample_one_hot.shape}")

    # Analyze one-hot vectors
    print(f"\nFirst 10 one-hot vectors:")
    for i in range(min(10, len(sample_one_hot))):
        vec = sample_one_hot[i]
        active_idx = np.where(vec > 0)[0]
        print(f"  {i}: {vec} → active index: {active_idx}")

    # Check if it's truly one-hot
    sum_check = sample_one_hot.sum(axis=1)
    print(f"\nOne-hot verification (sum should be 1.0):")
    print(f"  Min sum: {sum_check.min()}")
    print(f"  Max sum: {sum_check.max()}")
    print(f"  Mean sum: {sum_check.mean()}")
    print(f"  All exactly 1.0: {np.allclose(sum_check, 1.0)}")

    # Count unique types
    unique_types = set()
    for vec in sample_one_hot:
        active_idx = tuple(np.where(vec > 0)[0])
        unique_types.add(active_idx)

    print(f"\nUnique atom/residue types in first sample: {len(unique_types)}")
    print(f"Active indices: {sorted(unique_types)}")

    # Check constants
    print(f"\n{'='*80}")
    print("CHECKING CONSTANTS.PY")
    print("="*80)

    crossdock_info = dataset_params.get('crossdock', {})

    print("\nAvailable encoders:")
    if 'atom_encoder' in crossdock_info:
        atom_encoder = crossdock_info['atom_encoder']
        print(f"  atom_encoder ({len(atom_encoder)} types): {atom_encoder}")

    if 'aa_encoder' in crossdock_info:
        aa_encoder = crossdock_info['aa_encoder']
        print(f"  aa_encoder ({len(aa_encoder)} types): {list(aa_encoder.keys())}")

    # Hypothesis: 11 = 10 atom types + 1 extra (maybe for unknown/other)
    print(f"\n{'='*80}")
    print("HYPOTHESIS")
    print("="*80)
    print("""
The pocket has 11 features, which is unexpected:
- Constants define 10 atom types: C, N, O, S, B, Br, Cl, P, I, F
- Constants define 20 amino acid types

Possibilities:
1. 11 = 10 atom types + 1 extra (e.g., 'X' for unknown)
2. Full-atom representation with modified encoding
3. Custom encoding specific to this processed dataset

Need to check process_crossdock.py to see how pocket_one_hot is created.
""")

    # Check ligand encoding for comparison
    print(f"\n{'='*80}")
    print("LIGAND ENCODING (for comparison)")
    print("="*80)

    lig_one_hot = data['lig_one_hot']
    print(f"lig_one_hot shape: {lig_one_hot.shape}")
    print(f"Number of ligand features: {lig_one_hot.shape[1]}")

    sample_lig = lig_one_hot[data['lig_mask'] == sample_idx]
    lig_sum_check = sample_lig.sum(axis=1)
    print(f"\nLigand one-hot verification:")
    print(f"  All exactly 1.0: {np.allclose(lig_sum_check, 1.0)}")

    unique_lig_types = set()
    for vec in sample_lig:
        active_idx = tuple(np.where(vec > 0)[0])
        unique_lig_types.add(active_idx)

    print(f"Unique ligand types in first sample: {len(unique_lig_types)}")
    print(f"Active indices: {sorted(unique_lig_types)}")

    # KEY INSIGHT
    print(f"\n{'='*80}")
    print("KEY INSIGHT")
    print("="*80)
    print(f"""
Both ligand and pocket have 11 features!
This suggests:
- Full-atom representation (not CA-only)
- Same encoding scheme for both ligand and pocket
- Likely: 10 standard atom types + 1 for 'other' or padding

For ESM-C integration:
- We still use per-atom representation (not per-residue)
- Need to map atoms → residues for ESM-C broadcasting
- Residue IDs must be inferred from PDB processing
- Each atom gets the embedding of its parent residue

NEXT STEP: Check process_crossdock.py to understand atom→residue mapping
""")


def check_process_crossdock():
    """Check if process_crossdock.py is available"""
    print(f"\n{'='*80}")
    print("CHECKING PROCESSING SCRIPT")
    print("="*80)

    process_script = Path("process_crossdock.py")
    if process_script.exists():
        print(f"✓ Found: {process_script}")
        print("\nRecommendation: Analyze this script to understand:")
        print("  1. How pocket atoms are extracted from PDB")
        print("  2. If residue information is stored anywhere")
        print("  3. How to reconstruct atom→residue mapping")
    else:
        print(f"✗ Not found: {process_script}")
        print("  May need to write custom residue extraction logic")


if __name__ == "__main__":
    analyze_pocket_encoding()
    check_process_crossdock()
