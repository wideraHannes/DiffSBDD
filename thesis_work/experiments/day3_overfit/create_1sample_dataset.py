#!/usr/bin/env python3
"""
Create a 1-sample dataset for overfit test.
The crossdock data uses concatenated arrays with masks identifying each sample.
"""

import numpy as np
from pathlib import Path

# Paths
source_npz = Path("data/processed_crossdock_noH_full_temp/test.npz")
esmc_npz = Path("esmc_integration/embeddings_cache/test_esmc_embeddings.npz")
output_dir = Path("thesis_work/experiments/day3_overfit/data_1sample")
output_dir.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("Creating 1-Sample Dataset for Overfit Test")
print("=" * 60)

# Load source data
print("\n1. Loading source data...")
data = np.load(source_npz, allow_pickle=True)
print(f"   Keys: {list(data.keys())}")

# Extract sample 0 using masks
sample_idx = 0
lig_mask = data['lig_mask'] == sample_idx
pocket_mask = data['pocket_mask'] == sample_idx

print(f"\n2. Extracting sample {sample_idx}...")
print(f"   Name: {data['names'][sample_idx]}")
print(f"   Ligand atoms: {lig_mask.sum()}")
print(f"   Pocket atoms: {pocket_mask.sum()}")

# Create single-sample dataset (reset masks to 0)
single_sample = {
    'names': np.array([data['names'][sample_idx]]),
    'lig_coords': data['lig_coords'][lig_mask],
    'lig_one_hot': data['lig_one_hot'][lig_mask],
    'lig_mask': np.zeros(lig_mask.sum()),  # All atoms belong to sample 0
    'pocket_coords': data['pocket_coords'][pocket_mask],
    'pocket_one_hot': data['pocket_one_hot'][pocket_mask],
    'pocket_mask': np.zeros(pocket_mask.sum()),  # All atoms belong to sample 0
}

print(f"\n3. Data shapes:")
for key, val in single_sample.items():
    print(f"   {key}: {val.shape}")

# Atom type analysis
atom_types = np.argmax(single_sample['lig_one_hot'], axis=1)
atom_names = ['C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I', 'B', 'other']
print(f"\n4. Ligand atom composition:")
for i, name in enumerate(atom_names):
    count = (atom_types == i).sum()
    if count > 0:
        print(f"   {name}: {count}")

# Save train/val/test (all same for overfit test)
for split in ['train', 'val', 'test']:
    path = output_dir / f"{split}.npz"
    np.savez_compressed(path, **single_sample)
    print(f"\n5. Saved {split} set to {path}")

# Load and save ESM-C embeddings
print("\n6. Loading ESM-C embeddings...")
esmc_data = np.load(esmc_npz, allow_pickle=True)
esmc_single = {
    'embeddings': esmc_data['embeddings'][sample_idx:sample_idx+1],
    'sequences': esmc_data['sequences'][sample_idx:sample_idx+1],
    'names': esmc_data['names'][sample_idx:sample_idx+1],
}
print(f"   Embedding shape: {esmc_single['embeddings'][0].shape}")
print(f"   Sequence length: {len(esmc_single['sequences'][0])}")

esmc_output_dir = output_dir / "esmc_embeddings"
esmc_output_dir.mkdir(exist_ok=True)

for split in ['train', 'val', 'test']:
    path = esmc_output_dir / f"{split}_esmc_embeddings.npz"
    np.savez_compressed(path, **esmc_single)
    print(f"   Saved {split} embeddings to {path}")

# Create size distribution (for sampling during generation)
# Format: 2D array (n_ligand_sizes, n_pocket_sizes) joint distribution
print("\n7. Creating size distribution...")
lig_size = int(lig_mask.sum())
pocket_size = int(pocket_mask.sum())
size_dist = np.zeros((max(48, lig_size+1), max(646, pocket_size+1)))
size_dist[lig_size, pocket_size] = 1.0  # All probability at this single point
np.save(output_dir / "size_distribution.npy", size_dist)
print(f"   Ligand size: {lig_size}, Pocket size: {pocket_size}")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Created 1-sample dataset at: {output_dir}")
print(f"Sample: {data['names'][sample_idx]}")
print(f"Ligand: {lig_mask.sum()} atoms")
print(f"Pocket: {pocket_mask.sum()} atoms")
print("\nReady for overfit test!")
