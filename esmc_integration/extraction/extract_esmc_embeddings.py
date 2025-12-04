#!/usr/bin/env python3
"""
Extract ESM-C global embeddings for DiffSBDD pockets.

This script:
1. Loads pocket data from NPZ files
2. Extracts amino acid sequences from PDB files
3. Generates 960-dim global ESM-C embeddings
4. Saves embeddings as NPZ files for dataset loading

Usage:
    python extract_esmc_embeddings.py --split train --data_dir data/processed_crossdock_noH_full_temp
"""

import argparse
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from Bio.PDB import PDBParser
from esm.sdk import client
from esm.sdk.api import ESMProtein, LogitsConfig


# Amino acid three-letter to one-letter code
AA_3TO1 = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
    'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
    'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
}


def extract_sequence_from_pdb(pdb_path):
    """Extract amino acid sequence from PDB file."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('pocket', pdb_path)

    # Get all residues and extract sequence
    residues = []
    for model in structure:
        for chain in model:
            for residue in chain:
                # Skip water and hetero atoms
                if residue.id[0] == ' ':
                    res_name = residue.get_resname()
                    if res_name in AA_3TO1:
                        residues.append(AA_3TO1[res_name])
                    else:
                        # Unknown amino acid, use X
                        residues.append('X')

    sequence = ''.join(residues)
    return sequence


def generate_esmc_embedding(sequence, model):
    """Generate global ESM-C embedding for a sequence."""
    # Create protein and encode
    protein = ESMProtein(sequence=sequence)
    encoded = model.encode(protein)

    # Get embeddings: [batch=1, seq_len, hidden_dim=960]
    logits_output = model.logits(
        encoded,
        LogitsConfig(sequence=True, return_embeddings=True)
    )
    per_residue_emb = logits_output.embeddings

    # Global embedding via mean pooling (excluding BOS/EOS tokens)
    global_emb = per_residue_emb[0, 1:-1, :].mean(dim=0)  # [960]

    # Convert to numpy (bfloat16 -> float32)
    global_emb_np = global_emb.cpu().to(torch.float32).numpy()

    return global_emb_np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, required=True,
                        choices=['train', 'val', 'test'],
                        help='Dataset split to process')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to processed data directory')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for embeddings (default: data_dir/esmc_embeddings)')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples to process (for testing)')
    parser.add_argument('--token_file', type=str, default='.env',
                        help='File containing ESM API token')
    args = parser.parse_args()

    # Setup paths
    data_dir = Path(args.data_dir)
    npz_path = data_dir / f"{args.split}.npz"

    if args.output_dir is None:
        output_dir = data_dir / "esmc_embeddings"
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"{args.split}_esmc_embeddings.npz"

    print(f"Processing {args.split} split...")
    print(f"Data file: {npz_path}")
    print(f"Output file: {output_file}")

    # Load dataset
    with np.load(npz_path, allow_pickle=True) as f:
        names = f['names']

    n_samples = len(names) if args.max_samples is None else min(len(names), args.max_samples)
    print(f"\nTotal samples: {len(names)}")
    print(f"Processing: {n_samples}")

    # Initialize ESM-C client
    print("\nInitializing ESM-C client...")
    with open(args.token_file, 'r') as f:
        token = f.read().strip()

    esmc_client = client(
        model="esmc-300m-2024-12",
        url="https://forge.evolutionaryscale.ai",
        token=token
    )
    print("✓ ESM-C client ready")

    # Extract embeddings
    embeddings = []
    sequences = []
    failed_samples = []

    print(f"\nExtracting embeddings...")
    for idx in tqdm(range(n_samples)):
        name = names[idx]

        # Parse pocket PDB path from name
        # Format: "prefix/pocket.pdb_prefix/ligand.sdf"
        pdb_rel_path = name.split('_')[0] + '.pdb'  # Simplified - may need adjustment

        # Try to find PDB file in split subdirectory
        pdb_path = data_dir / args.split / pdb_rel_path

        if not pdb_path.exists():
            # Try alternative path parsing
            # Extract just the pocket filename from the complex name
            parts = name.split('/')
            if len(parts) > 0:
                pocket_name = parts[-1].split('_')[0]
                # Search for matching pdb file
                search_pattern = f"*{pocket_name}*pocket10.pdb"
                matches = list((data_dir / args.split).glob(search_pattern))
                if matches:
                    pdb_path = matches[0]

        if not pdb_path.exists():
            print(f"\nWarning: PDB file not found for sample {idx}: {name}")
            failed_samples.append((idx, name))
            # Use zero embedding as placeholder
            embeddings.append(np.zeros(960, dtype=np.float32))
            sequences.append("")
            continue

        try:
            # Extract sequence from PDB
            sequence = extract_sequence_from_pdb(pdb_path)

            if len(sequence) == 0:
                print(f"\nWarning: Empty sequence for sample {idx}: {pdb_path}")
                failed_samples.append((idx, name))
                embeddings.append(np.zeros(960, dtype=np.float32))
                sequences.append("")
                continue

            # Generate ESM-C embedding
            embedding = generate_esmc_embedding(sequence, esmc_client)

            embeddings.append(embedding)
            sequences.append(sequence)

        except Exception as e:
            print(f"\nError processing sample {idx} ({pdb_path}): {e}")
            failed_samples.append((idx, name))
            embeddings.append(np.zeros(960, dtype=np.float32))
            sequences.append("")

    # Convert to numpy array
    embeddings = np.array(embeddings, dtype=np.float32)
    sequences = np.array(sequences, dtype=object)

    # Save embeddings
    print(f"\nSaving embeddings to {output_file}...")
    np.savez_compressed(
        output_file,
        embeddings=embeddings,
        sequences=sequences,
        names=names[:n_samples]
    )
    print("✓ Saved")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Embeddings dtype: {embeddings.dtype}")
    print(f"Successful: {n_samples - len(failed_samples)}/{n_samples}")
    print(f"Failed: {len(failed_samples)}/{n_samples}")

    if embeddings.shape[0] > 0:
        # Calculate stats only for non-zero embeddings
        non_zero_mask = ~np.all(embeddings == 0, axis=1)
        if non_zero_mask.sum() > 0:
            valid_embeddings = embeddings[non_zero_mask]
            print(f"\nEmbedding statistics (excluding failures):")
            print(f"  Mean: {valid_embeddings.mean():.6f}")
            print(f"  Std: {valid_embeddings.std():.6f}")
            print(f"  Min: {valid_embeddings.min():.6f}")
            print(f"  Max: {valid_embeddings.max():.6f}")

    if failed_samples:
        print(f"\nFailed samples saved to: {output_dir}/{args.split}_failed.txt")
        with open(output_dir / f"{args.split}_failed.txt", 'w') as f:
            for idx, name in failed_samples:
                f.write(f"{idx}\t{name}\n")

    print("\n✓ Done!")


if __name__ == "__main__":
    main()
