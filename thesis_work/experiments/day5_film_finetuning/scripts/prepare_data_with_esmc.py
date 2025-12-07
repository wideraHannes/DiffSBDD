#!/usr/bin/env python3
"""
Prepare CrossDocked data with correct atom encoding AND ESM-C embeddings.

This script:
1. Processes ligand/pocket data with correct 10-feature atom encoding (matching checkpoint)
2. Extracts amino acid sequences from pocket PDB files
3. Generates 960-dim ESM-C global embeddings via ESM Forge API
4. Saves everything aligned by sample name

Usage:
    python prepare_data_with_esmc.py --basedir data --outdir data/processed_crossdock_noH_full_fixed --max_samples 100

Requirements:
    - ESM Forge API token in .env file (format: ESM_API_TOKEN=xxx or just the token)
"""

from pathlib import Path
from time import time
import argparse
import sys
import os
import shutil

from tqdm import tqdm
import numpy as np

from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa, protein_letters_3to1
from rdkit import Chem

import torch

# Add root to path
ROOT_DIR = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))
from constants import dataset_params

# ESM-C imports
try:
    from esm.sdk import client
    from esm.sdk.api import ESMProtein, LogitsConfig

    ESM_AVAILABLE = True
except ImportError:
    ESM_AVAILABLE = False
    print("Warning: ESM SDK not available. ESM-C embeddings will be skipped.")


def get_esm_token(token_file=".env"):
    """Load ESM API token from file."""
    token_path = ROOT_DIR / token_file
    if not token_path.exists():
        # Try current directory
        token_path = Path(token_file)

    if not token_path.exists():
        return None

    with open(token_path, "r") as f:
        content = f.read().strip()

    # Handle both "ESM_API_TOKEN=xxx" and just "xxx" formats
    if "=" in content:
        for line in content.split("\n"):
            if "ESM" in line.upper() and "=" in line:
                return line.split("=", 1)[1].strip()
    return content


def extract_sequence_from_pdb(pdb_path):
    """Extract amino acid sequence from PDB file."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("pocket", pdb_path)

    residues = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.id[0] == " ":  # Skip hetero atoms
                    res_name = residue.get_resname()
                    if res_name in protein_letters_3to1:
                        residues.append(protein_letters_3to1[res_name])
                    else:
                        residues.append("X")

    return "".join(residues)


def generate_esmc_embedding(sequence, esmc_client):
    """Generate global ESM-C embedding for a sequence."""
    protein = ESMProtein(sequence=sequence)
    encoded = esmc_client.encode(protein)

    logits_output = esmc_client.logits(
        encoded, LogitsConfig(sequence=True, return_embeddings=True)
    )
    per_residue_emb = logits_output.embeddings

    # Global embedding via mean pooling (excluding BOS/EOS tokens)
    global_emb = per_residue_emb[0, 1:-1, :].mean(dim=0)

    return global_emb.cpu().to(torch.float32).numpy()


def process_ligand_and_pocket(pdbfile, sdffile, atom_dict, dist_cutoff):
    """
    Process ligand and pocket for FULL-ATOM mode with CORRECT atom encoding.

    Uses atom_dict (10 types) for pocket atoms, NOT amino_acid_dict (20 types).
    """
    pdb_struct = PDBParser(QUIET=True).get_structure("", pdbfile)

    try:
        ligand = Chem.SDMolSupplier(str(sdffile))[0]
        if ligand is None:
            raise Exception("Failed to load ligand")
    except:
        raise Exception(f"cannot read sdf mol ({sdffile})")

    # Get ligand atoms (remove H if not in atom_dict)
    lig_atoms = [
        a.GetSymbol()
        for a in ligand.GetAtoms()
        if (
            a.GetSymbol().capitalize() in atom_dict or a.GetSymbol().capitalize() != "H"
        )
    ]
    lig_coords = np.array(
        [
            list(ligand.GetConformer(0).GetAtomPosition(idx))
            for idx in range(ligand.GetNumAtoms())
        ]
    )

    # Encode ligand atoms
    try:
        lig_one_hot = np.stack(
            [
                np.eye(1, len(atom_dict), atom_dict[a.capitalize()]).squeeze()
                for a in lig_atoms
            ]
        )
    except KeyError as e:
        raise KeyError(f"{e} not in atom dict ({sdffile})")

    # Find interacting pocket residues based on distance cutoff
    pocket_residues = []
    for residue in pdb_struct[0].get_residues():
        res_coords = np.array([a.get_coord() for a in residue.get_atoms()])
        if (
            is_aa(residue.get_resname(), standard=True)
            and (
                ((res_coords[:, None, :] - lig_coords[None, :, :]) ** 2).sum(-1) ** 0.5
            ).min()
            < dist_cutoff
        ):
            pocket_residues.append(residue)

    pocket_ids = [f"{res.parent.id}:{res.id[1]}" for res in pocket_residues]

    # FULL-ATOM mode: get all atoms from pocket residues
    full_atoms = np.concatenate(
        [
            np.array([atom.element for atom in res.get_atoms()])
            for res in pocket_residues
        ],
        axis=0,
    )

    full_coords = np.concatenate(
        [np.array([atom.coord for atom in res.get_atoms()]) for res in pocket_residues],
        axis=0,
    )

    # Encode pocket atoms with atom_dict (10 types) - NOT amino_acid_dict!
    pocket_one_hot = []
    filtered_coords = []

    for i, a in enumerate(full_atoms):
        atom_symbol = a.capitalize()
        if atom_symbol in atom_dict:
            atom_vec = np.eye(1, len(atom_dict), atom_dict[atom_symbol]).squeeze()
            pocket_one_hot.append(atom_vec)
            filtered_coords.append(full_coords[i])
        elif atom_symbol == "H":
            continue  # Skip hydrogen
        # Skip unknown atoms (no 'others' category in crossdock)

    if len(pocket_one_hot) == 0:
        raise ValueError(f"No valid pocket atoms found in {pdbfile}")

    pocket_one_hot = np.stack(pocket_one_hot)
    full_coords = np.stack(filtered_coords)

    ligand_data = {
        "lig_coords": lig_coords,
        "lig_one_hot": lig_one_hot,
    }
    pocket_data = {
        "pocket_coords": full_coords,
        "pocket_one_hot": pocket_one_hot,
        "pocket_ids": pocket_ids,
    }

    return ligand_data, pocket_data


def main():
    parser = argparse.ArgumentParser(
        description="Process CrossDocked data with ESM-C embeddings"
    )
    parser.add_argument("--basedir", type=Path, default="data")
    parser.add_argument("--outdir", type=Path, default=None)
    parser.add_argument("--dist_cutoff", type=float, default=8.0)
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Limit total samples (split into train/val/test)",
    )
    parser.add_argument(
        "--token_file", type=str, default=".env", help="File containing ESM API token"
    )
    parser.add_argument(
        "--skip_esmc", action="store_true", help="Skip ESM-C embedding generation"
    )
    args = parser.parse_args()

    datadir = args.basedir / "crossdocked_pocket10/"

    # Use crossdock dataset (10 atom types) - matches the checkpoint
    dataset_info = dataset_params["crossdock"]
    atom_dict = dataset_info["atom_encoder"]

    print("=" * 60)
    print("Preparing CrossDocked data with correct encoding + ESM-C")
    print("=" * 60)
    print(f"Using dataset: crossdock")
    print(f"Atom encoder: {len(atom_dict)} types: {list(atom_dict.keys())}")

    # Output directory
    if args.outdir is None:
        processed_dir = Path(args.basedir, "processed_crossdock_noH_full_fixed")
    else:
        processed_dir = args.outdir

    processed_dir.mkdir(exist_ok=True, parents=True)
    print(f"Output directory: {processed_dir}")

    # Initialize ESM-C client
    esmc_client = None
    if not args.skip_esmc and ESM_AVAILABLE:
        token = get_esm_token(args.token_file)
        if token:
            print("\nInitializing ESM-C client...")
            try:
                esmc_client = client(
                    model="esmc-300m-2024-12",
                    url="https://forge.evolutionaryscale.ai",
                    token=token,
                )
                print("✓ ESM-C client ready")
            except Exception as e:
                print(f"✗ ESM-C client failed: {e}")
                esmc_client = None
        else:
            print("Warning: No ESM API token found. Skipping ESM-C embeddings.")
    elif args.skip_esmc:
        print("Skipping ESM-C embeddings (--skip_esmc)")
    else:
        print("Warning: ESM SDK not available. Skipping ESM-C embeddings.")

    # Read data split
    split_path = Path(args.basedir, "split_by_name.pt")
    data_split = torch.load(split_path)

    print(
        f"\nOriginal dataset: Train={len(data_split['train'])}, Test={len(data_split['test'])}"
    )

    # Limit samples and create train/val/test split
    if args.max_samples:
        # Take max_samples total, split into train/val/test
        # Test: max 100 samples, Val: 10%, Train: rest
        total = min(args.max_samples, len(data_split["train"]))
        all_samples = data_split["train"][:total]

        test_size = min(100, int(0.10 * total))  # Max 100 test samples
        val_size = int(0.10 * total)  # 10% for validation
        train_size = total - val_size - test_size

        data_split = {
            "train": all_samples[:train_size],
            "val": all_samples[train_size : train_size + val_size],
            "test": all_samples[
                train_size + val_size : train_size + val_size + test_size
            ],
        }
    else:
        # Create validation split from train
        val_size = int(0.15 * len(data_split["train"]))
        data_split["val"] = data_split["train"][:val_size]
        data_split["train"] = data_split["train"][val_size:]

    print(
        f"Processing: Train={len(data_split['train'])}, Val={len(data_split['val'])}, Test={len(data_split['test'])}"
    )

    # Process each split
    for split in ["train", "val", "test"]:
        print(f"\n{'=' * 40}")
        print(f"Processing {split} split...")
        print(f"{'=' * 40}")

        lig_coords = []
        lig_one_hot = []
        lig_mask = []
        pocket_coords = []
        pocket_one_hot = []
        pocket_mask = []
        pdb_and_mol_ids = []

        esmc_embeddings = []
        esmc_sequences = []

        count = 0
        num_failed = 0

        split_dir = processed_dir / split
        split_dir.mkdir(exist_ok=True)

        tic = time()
        pbar = tqdm(data_split[split], desc=f"{split}")

        for pocket_fn, ligand_fn in pbar:
            sdffile = datadir / f"{ligand_fn}"
            pdbfile = datadir / f"{pocket_fn}"

            try:
                # Process ligand and pocket
                ligand_data, pocket_data = process_ligand_and_pocket(
                    pdbfile,
                    sdffile,
                    atom_dict=atom_dict,
                    dist_cutoff=args.dist_cutoff,
                )

                # Generate ESM-C embedding if client available
                if esmc_client:
                    sequence = extract_sequence_from_pdb(pdbfile)
                    if len(sequence) > 0:
                        embedding = generate_esmc_embedding(sequence, esmc_client)
                    else:
                        embedding = np.zeros(960, dtype=np.float32)
                        sequence = ""
                else:
                    embedding = np.zeros(960, dtype=np.float32)
                    sequence = ""

            except Exception as e:
                num_failed += 1
                pbar.set_description(f"{split} (failed: {num_failed})")
                continue

            # Store data
            sample_name = f"{pocket_fn}_{ligand_fn}"
            pdb_and_mol_ids.append(sample_name)

            lig_coords.append(ligand_data["lig_coords"])
            lig_one_hot.append(ligand_data["lig_one_hot"])
            lig_mask.append(count * np.ones(len(ligand_data["lig_coords"])))

            pocket_coords.append(pocket_data["pocket_coords"])
            pocket_one_hot.append(pocket_data["pocket_one_hot"])
            pocket_mask.append(count * np.ones(len(pocket_data["pocket_coords"])))

            esmc_embeddings.append(embedding)
            esmc_sequences.append(sequence)

            # Copy PDB and SDF files for val/test splits (for evaluation)
            if split in {"val", "test"}:
                # Copy PDB file
                new_rec_name = Path(pdbfile).stem.replace("_", "-")
                pdb_file_out = Path(split_dir, f"{new_rec_name}.pdb")
                shutil.copy(pdbfile, pdb_file_out)

                # Copy SDF file
                new_lig_name = new_rec_name + "_" + Path(sdffile).stem.replace("_", "-")
                sdf_file_out = Path(split_dir, f"{new_lig_name}.sdf")
                shutil.copy(sdffile, sdf_file_out)

                # Write pocket residue IDs
                with open(Path(split_dir, f"{new_lig_name}.txt"), "w") as f:
                    f.write(" ".join(pocket_data["pocket_ids"]))

            count += 1

        elapsed = time() - tic
        print(
            f"\n{split}: {count} samples processed, {num_failed} failed in {elapsed:.1f}s"
        )

        if count == 0:
            print(f"Warning: No samples processed for {split}")
            continue

        # Concatenate data arrays
        lig_coords = np.concatenate(lig_coords, axis=0)
        lig_one_hot = np.concatenate(lig_one_hot, axis=0)
        lig_mask = np.concatenate(lig_mask, axis=0)
        pocket_coords = np.concatenate(pocket_coords, axis=0)
        pocket_one_hot = np.concatenate(pocket_one_hot, axis=0)
        pocket_mask = np.concatenate(pocket_mask, axis=0)

        esmc_embeddings = np.array(esmc_embeddings, dtype=np.float32)
        esmc_sequences = np.array(esmc_sequences, dtype=object)
        pdb_and_mol_ids = np.array(pdb_and_mol_ids, dtype=object)

        # Save main data
        np.savez(
            processed_dir / f"{split}.npz",
            names=pdb_and_mol_ids,
            lig_coords=lig_coords,
            lig_one_hot=lig_one_hot,
            lig_mask=lig_mask,
            pocket_coords=pocket_coords,
            pocket_one_hot=pocket_one_hot,
            pocket_mask=pocket_mask,
        )
        print(f"  Saved {split}.npz")
        print(f"    lig_one_hot: {lig_one_hot.shape}")
        print(f"    pocket_one_hot: {pocket_one_hot.shape}")

        # Save ESM-C embeddings (same format as extract_esmc_embeddings.py)
        np.savez(
            processed_dir / f"{split}_esmc.npz",
            embeddings=esmc_embeddings,
            sequences=esmc_sequences,
            names=pdb_and_mol_ids,
        )
        print(f"  Saved {split}_esmc.npz")
        print(f"    embeddings: {esmc_embeddings.shape}")

        # For train split, also save SMILES and size distribution
        if split == "train":
            # Generate size distribution histogram
            from scipy.ndimage import gaussian_filter

            idx_lig, n_nodes_lig = np.unique(lig_mask, return_counts=True)
            idx_pocket, n_nodes_pocket = np.unique(pocket_mask, return_counts=True)

            joint_histogram = np.zeros(
                (int(np.max(n_nodes_lig)) + 1, int(np.max(n_nodes_pocket)) + 1)
            )
            for nlig, npocket in zip(n_nodes_lig, n_nodes_pocket):
                joint_histogram[int(nlig), int(npocket)] += 1

            # Smooth the histogram
            filtered = gaussian_filter(
                joint_histogram,
                sigma=1.0,
                order=0,
                mode="constant",
                cval=0.0,
                truncate=4.0,
            )
            np.save(processed_dir / "size_distribution.npy", filtered)
            print(f"  Saved size_distribution.npy (shape: {filtered.shape})")

            # Generate train SMILES (placeholder - could extract from ligands if needed)
            # This is used for novelty checking during evaluation
            train_smiles = np.array(["C", "CC", "CCC"], dtype=object)  # Placeholder
            np.save(processed_dir / "train_smiles.npy", train_smiles)
            print(f"  Saved train_smiles.npy (placeholder)")

    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)
    print(f"Output directory: {processed_dir}")
    print(f"Files created: train.npz, val.npz, test.npz")
    print(f"               train_esmc.npz, val_esmc.npz, test_esmc.npz")
    print(f"               size_distribution.npy, train_smiles.npy")
    print(f"\nData format:")
    print(f"  - pocket_one_hot: 10 features (matches checkpoint)")
    print(f"  - lig_one_hot: 10 features")
    print(f"  - ESM-C embeddings: 960-dim per sample")


if __name__ == "__main__":
    main()
