"""
Extract a 10-pocket subset from the test set for quick baseline evaluation.

Usage:
    python 1_extract_subset.py
"""

import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def extract_subset(
    test_set_path='data/processed_crossdock_noH_full_temp/test.npz',
    n_pockets=10,
    output_path='baseline/data/test_subset_10.npz',
    seed=42
):
    """Extract random subset of test pockets."""

    print("="*80)
    print("STEP 1: EXTRACT TEST SUBSET")
    print("="*80)

    # Load full test set
    print(f"\nLoading test set: {test_set_path}")
    test_set_path = Path(test_set_path)

    if not test_set_path.exists():
        print(f"ERROR: Test set not found at {test_set_path}")
        print("\nTry these paths:")
        print("  - data/processed_crossdock_noH_full/test.npz")
        print("  - data/processed_crossdock_noH_full_temp/test.npz")
        return False

    data = np.load(test_set_path, allow_pickle=True)

    total_pockets = len(data['names'])
    print(f"✓ Loaded test set with {total_pockets} pockets")

    # Random sample
    np.random.seed(seed)
    selected_indices = np.random.choice(total_pockets, size=n_pockets, replace=False)
    selected_indices = sorted(selected_indices)

    print(f"\nSelected {n_pockets} random pockets:")
    for i, idx in enumerate(selected_indices):
        print(f"  {i}: {data['names'][idx]}")

    # Extract subset
    subset = {}

    for key in data.keys():
        if key in ['names', 'receptors']:
            # String arrays - direct indexing
            subset[key] = data[key][selected_indices]
        elif 'mask' in key:
            # Masks need special handling
            if key.startswith('lig'):
                # Find atoms for selected ligands
                mask_values = []
                for new_idx, old_idx in enumerate(selected_indices):
                    mask_values.extend([new_idx] * np.sum(data['lig_mask'] == old_idx))
                subset[key] = np.array(mask_values)
            else:  # pocket
                mask_values = []
                for new_idx, old_idx in enumerate(selected_indices):
                    mask_values.extend([new_idx] * np.sum(data['pocket_mask'] == old_idx))
                subset[key] = np.array(mask_values)
        else:
            # Coordinates and one-hot - concatenate atoms from selected samples
            if key.startswith('lig'):
                selected_atoms = np.concatenate([
                    data[key][data['lig_mask'] == idx]
                    for idx in selected_indices
                ])
            else:  # pocket
                selected_atoms = np.concatenate([
                    data[key][data['pocket_mask'] == idx]
                    for idx in selected_indices
                ])
            subset[key] = selected_atoms

    # Verify subset
    print(f"\nSubset verification:")
    print(f"  Names: {len(subset['names'])} pockets")
    print(f"  Ligand atoms: {len(subset['lig_coords'])}")
    print(f"  Pocket atoms: {len(subset['pocket_coords'])}")
    print(f"  Ligand mask unique values: {len(np.unique(subset['lig_mask']))}")
    print(f"  Pocket mask unique values: {len(np.unique(subset['pocket_mask']))}")

    # Save subset
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving subset to: {output_path}")
    np.savez_compressed(output_path, **subset)

    print(f"\n✓ Successfully created {n_pockets}-pocket subset!")
    print(f"  Output: {output_path}")
    print(f"  Size: {output_path.stat().st_size / 1024:.1f} KB")

    return True


if __name__ == '__main__':
    success = extract_subset()

    if success:
        print("\n" + "="*80)
        print("STEP 1 COMPLETE")
        print("="*80)
        print("\nNext step: Run 2_generate_molecules.py")
    else:
        print("\n" + "="*80)
        print("STEP 1 FAILED")
        print("="*80)
        print("\nPlease check the error messages above and fix the test set path.")
        sys.exit(1)
