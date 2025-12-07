#!/usr/bin/env python3
"""
Create a configurable overfit dataset from the crossdock data.
Supports extracting N training samples and M validation samples.
"""

import argparse
import numpy as np
from pathlib import Path


def extract_samples(data, esmc_data, sample_indices):
    """Extract multiple samples from the dataset."""
    all_lig_coords = []
    all_lig_one_hot = []
    all_lig_mask = []
    all_pocket_coords = []
    all_pocket_one_hot = []
    all_pocket_mask = []
    names = []

    all_embeddings = []
    all_sequences = []
    all_esmc_names = []

    for new_idx, orig_idx in enumerate(sample_indices):
        # Extract molecular data using masks
        lig_mask = data['lig_mask'] == orig_idx
        pocket_mask = data['pocket_mask'] == orig_idx

        all_lig_coords.append(data['lig_coords'][lig_mask])
        all_lig_one_hot.append(data['lig_one_hot'][lig_mask])
        all_lig_mask.append(np.full(lig_mask.sum(), new_idx, dtype=np.float64))

        all_pocket_coords.append(data['pocket_coords'][pocket_mask])
        all_pocket_one_hot.append(data['pocket_one_hot'][pocket_mask])
        all_pocket_mask.append(np.full(pocket_mask.sum(), new_idx, dtype=np.float64))

        names.append(data['names'][orig_idx])

        # Extract ESMC embeddings if available
        if esmc_data is not None:
            all_embeddings.append(esmc_data['embeddings'][orig_idx])
            all_sequences.append(esmc_data['sequences'][orig_idx])
            all_esmc_names.append(esmc_data['names'][orig_idx])

    result = {
        'names': np.array(names),
        'lig_coords': np.concatenate(all_lig_coords, axis=0),
        'lig_one_hot': np.concatenate(all_lig_one_hot, axis=0),
        'lig_mask': np.concatenate(all_lig_mask, axis=0),
        'pocket_coords': np.concatenate(all_pocket_coords, axis=0),
        'pocket_one_hot': np.concatenate(all_pocket_one_hot, axis=0),
        'pocket_mask': np.concatenate(all_pocket_mask, axis=0),
    }

    esmc_result = None
    if esmc_data is not None:
        esmc_result = {
            'embeddings': np.array(all_embeddings, dtype=object),
            'sequences': np.array(all_sequences, dtype=object),
            'names': np.array(all_esmc_names),
        }

    return result, esmc_result


def main():
    parser = argparse.ArgumentParser(description='Create overfit dataset')
    parser.add_argument('--n_train', type=int, default=1, help='Number of training samples')
    parser.add_argument('--n_val', type=int, default=1, help='Number of validation samples')
    parser.add_argument('--train_start', type=int, default=0, help='Starting index for training samples')
    parser.add_argument('--val_start', type=int, default=None, help='Starting index for val (default: after train)')
    parser.add_argument('--source', type=str, default='test', choices=['train', 'val', 'test'],
                        help='Source split to extract from')
    parser.add_argument('--output_dir', type=str, default='thesis_work/experiments/day3_overfit/data_overfit',
                        help='Output directory')
    parser.add_argument('--source_dir', type=str, default='data/processed_crossdock_noH_full_temp',
                        help='Source data directory')
    parser.add_argument('--esmc_dir', type=str, default='esmc_integration/embeddings_cache',
                        help='ESMC embeddings directory (set to "none" to skip)')
    args = parser.parse_args()

    # Set default val_start to after training samples
    if args.val_start is None:
        args.val_start = args.train_start + args.n_train

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Creating Overfit Dataset")
    print("=" * 60)
    print(f"Training samples: {args.n_train} (indices {args.train_start}-{args.train_start + args.n_train - 1})")
    print(f"Validation samples: {args.n_val} (indices {args.val_start}-{args.val_start + args.n_val - 1})")
    print(f"Source: {args.source}.npz")
    print(f"Output: {output_dir}")

    # Load source data
    print("\n1. Loading source data...")
    source_npz = Path(args.source_dir) / f"{args.source}.npz"
    data = np.load(source_npz, allow_pickle=True)
    print(f"   Keys: {list(data.keys())}")

    # Load ESMC embeddings if available
    esmc_data = None
    if args.esmc_dir.lower() != 'none':
        esmc_npz = Path(args.esmc_dir) / f"{args.source}_esmc_embeddings.npz"
        if esmc_npz.exists():
            print(f"   Loading ESMC embeddings from {esmc_npz}")
            esmc_data = np.load(esmc_npz, allow_pickle=True)
        else:
            print(f"   ESMC embeddings not found at {esmc_npz}, skipping")

    # Extract training samples
    train_indices = list(range(args.train_start, args.train_start + args.n_train))
    val_indices = list(range(args.val_start, args.val_start + args.n_val))

    print(f"\n2. Extracting training samples: {train_indices}")
    train_data, train_esmc = extract_samples(data, esmc_data, train_indices)

    print(f"   Extracting validation samples: {val_indices}")
    val_data, val_esmc = extract_samples(data, esmc_data, val_indices)

    # Print summary
    print(f"\n3. Dataset shapes:")
    print(f"   Train - Ligand atoms: {len(train_data['lig_coords'])}, Pocket atoms: {len(train_data['pocket_coords'])}")
    print(f"   Val - Ligand atoms: {len(val_data['lig_coords'])}, Pocket atoms: {len(val_data['pocket_coords'])}")

    # Save datasets
    print(f"\n4. Saving datasets...")
    np.savez_compressed(output_dir / "train.npz", **train_data)
    np.savez_compressed(output_dir / "val.npz", **val_data)
    np.savez_compressed(output_dir / "test.npz", **val_data)  # Use val for test too
    print(f"   Saved train.npz, val.npz, test.npz")

    # Save ESMC embeddings if available
    if train_esmc is not None:
        esmc_output_dir = output_dir / "esmc_embeddings"
        esmc_output_dir.mkdir(exist_ok=True)
        np.savez_compressed(esmc_output_dir / "train_esmc_embeddings.npz", **train_esmc)
        np.savez_compressed(esmc_output_dir / "val_esmc_embeddings.npz", **val_esmc)
        np.savez_compressed(esmc_output_dir / "test_esmc_embeddings.npz", **val_esmc)
        print(f"   Saved ESMC embeddings")

    # Create size distribution
    print(f"\n5. Creating size distribution...")
    max_lig = max(48, max((train_data['lig_mask'] == i).sum() for i in range(args.n_train)) + 1)
    max_pocket = max(646, max((train_data['pocket_mask'] == i).sum() for i in range(args.n_train)) + 1)
    size_dist = np.zeros((max_lig, max_pocket))
    for i in range(args.n_train):
        lig_size = int((train_data['lig_mask'] == i).sum())
        pocket_size = int((train_data['pocket_mask'] == i).sum())
        size_dist[lig_size, pocket_size] += 1.0
    size_dist /= size_dist.sum()  # Normalize
    np.save(output_dir / "size_distribution.npy", size_dist)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print(f"Training samples: {args.n_train}")
    print(f"Validation samples: {args.n_val}")
    for i, idx in enumerate(train_indices):
        print(f"  Train[{i}]: {data['names'][idx]}")
    for i, idx in enumerate(val_indices):
        print(f"  Val[{i}]: {data['names'][idx]}")
    print("\nReady for overfit test!")


if __name__ == "__main__":
    main()
