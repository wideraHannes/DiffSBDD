"""
Process CrossDocked data to match the checkpoint's expected format.

The original process_crossdock.py has a bug: for full-atom pocket representation,
it uses amino_acid_dict (20 types) to encode pocket atoms, but the checkpoint
was trained with atom_dict (10 types) for pocket encoding.

This script correctly processes data for:
- dataset: crossdock (10 atom types)
- pocket_representation: full-atom (using atom_dict for pocket, not amino_acid_dict)

Usage:
    python process_data_for_checkpoint.py --basedir data --outdir data/processed_crossdock_noH_full_fixed
"""

from pathlib import Path
from time import time
import argparse
import sys

from tqdm import tqdm
import numpy as np

from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa
from rdkit import Chem

import torch

# Add root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))
from constants import dataset_params


def process_ligand_and_pocket(pdbfile, sdffile, atom_dict, dist_cutoff):
    """
    Process ligand and pocket for FULL-ATOM mode with CORRECT atom encoding.

    Key difference from original: pocket atoms are encoded with atom_dict (10 types)
    instead of amino_acid_dict (20 types).
    """
    pdb_struct = PDBParser(QUIET=True).get_structure("", pdbfile)

    try:
        ligand = Chem.SDMolSupplier(str(sdffile))[0]
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
    # CORRECTLY encode with atom_dict (10 types), not amino_acid_dict!
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

    try:
        pocket_one_hot = []
        for a in full_atoms:
            atom_symbol = a.capitalize()
            if atom_symbol in atom_dict:
                # Known atom type
                atom_vec = np.eye(1, len(atom_dict), atom_dict[atom_symbol]).squeeze()
            elif atom_symbol != "H":
                # Unknown atom (not hydrogen) - skip it
                # The checkpoint doesn't have an "others" category
                continue
            else:
                # Hydrogen - skip
                continue
            pocket_one_hot.append(atom_vec)

        # Filter coords to match (remove H and unknown atoms)
        filtered_coords = []
        for i, a in enumerate(full_atoms):
            atom_symbol = a.capitalize()
            if atom_symbol in atom_dict:
                filtered_coords.append(full_coords[i])
            elif atom_symbol != "H":
                continue
            else:
                continue

        pocket_one_hot = np.stack(pocket_one_hot)
        full_coords = np.stack(filtered_coords)

    except KeyError as e:
        raise KeyError(f"{e} not in atom dict ({pdbfile})")

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--basedir", type=Path, default="data")
    parser.add_argument("--outdir", type=Path, default=None)
    parser.add_argument("--dist_cutoff", type=float, default=8.0)
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Limit number of samples per split (for testing)",
    )
    parser.add_argument(
        "--use_esmc_samples",
        type=Path,
        default=None,
        help="Path to existing processed dir with ESM-C - use same samples",
    )
    args = parser.parse_args()

    datadir = args.basedir / "crossdocked_pocket10/"

    # Use crossdock dataset (10 atom types) - this matches the checkpoint
    dataset_info = dataset_params["crossdock"]
    atom_dict = dataset_info["atom_encoder"]

    print(f"Using dataset: crossdock")
    print(f"Atom encoder has {len(atom_dict)} types: {list(atom_dict.keys())}")

    # Output directory
    if args.outdir is None:
        processed_dir = Path(args.basedir, "processed_crossdock_noH_full_fixed")
    else:
        processed_dir = args.outdir

    processed_dir.mkdir(exist_ok=True, parents=True)
    print(f"Output directory: {processed_dir}")

    # Read data split
    split_path = Path(args.basedir, "split_by_name.pt")
    data_split = torch.load(split_path)

    print(
        f"Original dataset sizes - Train: {len(data_split['train'])}, Test: {len(data_split['test'])}"
    )

    # Limit samples if requested
    if args.max_samples:
        train_size = min(args.max_samples, len(data_split["train"]))
        test_size = min(args.max_samples // 5, len(data_split["test"]))
        data_split["train"] = data_split["train"][:train_size]
        data_split["test"] = data_split["test"][:test_size]

    # If using existing ESM-C samples, load those names and filter to match
    if args.use_esmc_samples:
        print(f"\nUsing sample names from existing ESM-C data: {args.use_esmc_samples}")
        esmc_names = {}
        for split_name in ["train", "val", "test"]:
            esmc_file = args.use_esmc_samples / f"{split_name}_esmc.npz"
            if esmc_file.exists():
                esmc_data = np.load(esmc_file, allow_pickle=True)
                esmc_names[split_name] = set(esmc_data["names"])
                print(
                    f"  {split_name}: {len(esmc_names[split_name])} samples from ESM-C"
                )

        # We need to filter from the full split to get matching samples
        # Rebuild data_split to match ESM-C samples
        full_data = data_split["train"] + data_split["test"]

        # Build a mapping from name to (pocket_fn, ligand_fn)
        name_to_files = {}
        for pocket_fn, ligand_fn in full_data:
            name = f"{pocket_fn}_{ligand_fn}"
            name_to_files[name] = (pocket_fn, ligand_fn)

        # Rebuild splits from ESM-C names
        for split_name in ["train", "val", "test"]:
            if split_name in esmc_names:
                data_split[split_name] = [
                    name_to_files[name]
                    for name in esmc_names[split_name]
                    if name in name_to_files
                ]

        print(f"\nFiltered to ESM-C samples:")
    else:
        # Create validation split (15% of train)
        val_size = int(0.15 * len(data_split["train"]))
        data_split["val"] = data_split["train"][:val_size]
        data_split["train"] = data_split["train"][val_size:]

    print(
        f"Processing sizes - Train: {len(data_split['train'])}, Val: {len(data_split['val'])}, Test: {len(data_split['test'])}"
    )

    for split in data_split.keys():
        lig_coords = []
        lig_one_hot = []
        lig_mask = []
        pocket_coords = []
        pocket_one_hot = []
        pocket_mask = []
        pdb_and_mol_ids = []
        count = 0

        split_dir = processed_dir / split
        split_dir.mkdir(exist_ok=True)

        tic = time()
        num_failed = 0
        pbar = tqdm(data_split[split])
        pbar.set_description(f"Processing {split}")

        for pocket_fn, ligand_fn in pbar:
            sdffile = datadir / f"{ligand_fn}"
            pdbfile = datadir / f"{pocket_fn}"

            try:
                ligand_data, pocket_data = process_ligand_and_pocket(
                    pdbfile,
                    sdffile,
                    atom_dict=atom_dict,
                    dist_cutoff=args.dist_cutoff,
                )
            except (
                KeyError,
                AssertionError,
                FileNotFoundError,
                IndexError,
                ValueError,
                Exception,
            ) as e:
                num_failed += 1
                pbar.set_description(f"#failed: {num_failed}")
                continue

            pdb_and_mol_ids.append(f"{pocket_fn}_{ligand_fn}")
            lig_coords.append(ligand_data["lig_coords"])
            lig_one_hot.append(ligand_data["lig_one_hot"])
            lig_mask.append(count * np.ones(len(ligand_data["lig_coords"])))
            pocket_coords.append(pocket_data["pocket_coords"])
            pocket_one_hot.append(pocket_data["pocket_one_hot"])
            pocket_mask.append(count * np.ones(len(pocket_data["pocket_coords"])))
            count += 1

        print(
            f"{split}: {count} samples processed, {num_failed} failed in {time() - tic:.2f}s"
        )

        if count == 0:
            print(f"Warning: No samples processed for {split}")
            continue

        # Concatenate all data
        lig_coords = np.concatenate(lig_coords, axis=0)
        lig_one_hot = np.concatenate(lig_one_hot, axis=0)
        lig_mask = np.concatenate(lig_mask, axis=0)
        pocket_coords = np.concatenate(pocket_coords, axis=0)
        pocket_one_hot = np.concatenate(pocket_one_hot, axis=0)
        pocket_mask = np.concatenate(pocket_mask, axis=0)

        # Save
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

        print(f"Saved {split}.npz")
        print(f"  lig_one_hot shape: {lig_one_hot.shape}")
        print(f"  pocket_one_hot shape: {pocket_one_hot.shape}")

    print("\nDone! Data processed with correct atom encoding for checkpoint.")


if __name__ == "__main__":
    main()
