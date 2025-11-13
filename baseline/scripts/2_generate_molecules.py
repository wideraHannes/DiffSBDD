"""
Generate molecules from baseline model for 10-pocket subset.

Usage:
    python 2_generate_molecules.py
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from lightning_modules import LigandPocketDDPM
from analysis.molecule_builder import build_molecule, process_molecule
from utils import write_sdf_file


def generate_molecules(
    checkpoint_path="checkpoints/crossdocked_fullatom_cond.ckpt",
    test_subset_path="baseline/data/test_subset_10.npz",
    n_samples=1,
    output_dir="baseline/results/molecules",
    device="cuda",
):
    """Generate molecules for 10-pocket baseline test."""

    print("=" * 80)
    print("STEP 2: GENERATE MOLECULES")
    print("=" * 80)

    # Check checkpoint
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        print(f"\nERROR: Checkpoint not found at {checkpoint_path}")
        print("\nPlease download the baseline checkpoint:")
        print("  - From Zenodo: https://zenodo.org/record/8183747")
        print("  - Or check if you have it elsewhere")
        return False

    # Check test subset
    test_subset_path = Path(test_subset_path)
    if not test_subset_path.exists():
        print(f"\nERROR: Test subset not found at {test_subset_path}")
        print("Please run: python 1_extract_subset.py first")
        return False

    # Check GPU
    if device == "cuda" and not torch.cuda.is_available():
        print("\nWARNING: CUDA not available, falling back to CPU")
        print("This will be SLOW (~10x slower)")
        device = "cpu"

    # Load model
    print(f"\nLoading checkpoint: {checkpoint_path}")
    model = LigandPocketDDPM.load_from_checkpoint(checkpoint_path, map_location=device)
    model = model.eval().to(device)
    print(f"✓ Model loaded successfully")
    print(f"  Mode: {model.hparams.mode}")
    print(f"  Device: {device}")

    # Check what dimension the model expects
    expected_atom_dim = len(model.dataset_info.get("atom_encoder", {}))
    expected_residue_dim = len(model.dataset_info.get("aa_encoder", {}))
    print(
        f"  Model expects: {expected_atom_dim} atom types, {expected_residue_dim} residue types"
    )

    # Load test subset
    print(f"\nLoading test subset: {test_subset_path}")
    data = np.load(test_subset_path, allow_pickle=True)
    n_pockets = len(data["names"])
    print(f"✓ Loaded {n_pockets} pockets")

    # Check data dimensions
    sample_pocket_one_hot = data["pocket_one_hot"][data["pocket_mask"] == 0]
    sample_lig_one_hot = data["lig_one_hot"][data["lig_mask"] == 0]
    print(f"  Data dimensions:")
    print(f"    Pocket features: {sample_pocket_one_hot.shape[1]}")
    print(f"    Ligand features: {sample_lig_one_hot.shape[1]}")

    # Warn if mismatch
    if sample_pocket_one_hot.shape[1] != expected_atom_dim:
        print(f"\n  ⚠ WARNING: Dimension mismatch detected!")
        print(f"    Data has {sample_pocket_one_hot.shape[1]} pocket features")
        print(f"    Model expects {expected_atom_dim} features")
        print(
            f"    Will trim to {expected_atom_dim} features (dropping unknown atom type)"
        )
    if sample_lig_one_hot.shape[1] != expected_atom_dim:
        print(f"    Data has {sample_lig_one_hot.shape[1]} ligand features")
        print(f"    Model expects {expected_atom_dim} features")
        print(
            f"    Will trim to {expected_atom_dim} features (dropping unknown atom type)"
        )

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate for each pocket
    print(f"\nGenerating {n_samples} molecules per pocket...")
    all_results = []
    total_generated = 0
    total_valid = 0

    for pocket_idx in tqdm(range(n_pockets), desc="Pockets"):
        # Extract pocket data
        pocket_mask_bool = data["pocket_mask"] == pocket_idx
        pocket_coords = torch.from_numpy(
            data["pocket_coords"][pocket_mask_bool]
        ).float()
        pocket_one_hot = torch.from_numpy(
            data["pocket_one_hot"][pocket_mask_bool]
        ).float()

        # FIX: Handle dimension mismatch (data has 11 features, model expects 10)
        # The 11th feature is for unknown atoms - we need to drop it
        if pocket_one_hot.shape[1] > expected_atom_dim:
            pocket_one_hot = pocket_one_hot[:, :expected_atom_dim]

        # Prepare pocket dict
        pocket = {
            "x": pocket_coords.to(device),
            "one_hot": pocket_one_hot.to(device),
            "size": torch.tensor([len(pocket_coords)], device=device),
            "mask": torch.zeros(len(pocket_coords), dtype=torch.long, device=device),
        }

        # Repeat for n_samples
        pocket_repeated = {
            "x": pocket["x"].repeat(n_samples, 1),
            "one_hot": pocket["one_hot"].repeat(n_samples, 1),
            "size": torch.tensor([len(pocket_coords)] * n_samples, device=device),
            "mask": torch.arange(n_samples, device=device).repeat_interleave(
                len(pocket_coords)
            ),
        }

        # Sample ligand size
        with torch.no_grad():
            num_nodes_lig = model.ddpm.size_distribution.sample_conditional(
                n1=None, n2=pocket_repeated["size"]
            )

            # Generate molecules
            xh_lig, _, _, _ = model.ddpm.sample_given_pocket(
                pocket_repeated, num_nodes_lig
            )

        # Build RDKit molecules
        molecules = []
        for sample_idx in range(n_samples):
            sample_mask = pocket_repeated["mask"] == sample_idx
            coords = xh_lig[sample_mask, :3].cpu().numpy()
            atom_types = torch.argmax(xh_lig[sample_mask, 3:], dim=1).cpu().numpy()

            try:
                mol = build_molecule(
                    coords, atom_types, model.dataset_info, use_openbabel=True
                )

                if mol is not None:
                    # Process molecule
                    mol = process_molecule(
                        mol,
                        sanitize=True,
                        relax_iter=0,  # Skip relaxation for speed
                        largest_frag=True,
                    )

                    if mol is not None:
                        molecules.append(mol)
                        total_valid += 1

            except Exception as e:
                # Skip failed molecules
                pass

            total_generated += 1

        # Save molecules for this pocket
        pocket_name = f"pocket_{pocket_idx}"
        sdf_path = output_dir / f"{pocket_name}.sdf"

        if molecules:
            write_sdf_file(sdf_path, molecules)

        all_results.append(
            {
                "pocket_idx": pocket_idx,
                "pocket_name": pocket_name,
                "n_requested": n_samples,
                "n_generated": len(molecules),
                "sdf_path": str(sdf_path) if molecules else None,
            }
        )

    # Summary
    print(f"\n{'=' * 80}")
    print("GENERATION SUMMARY")
    print("=" * 80)
    print(f"Total pockets: {n_pockets}")
    print(f"Molecules requested: {total_generated}")
    print(
        f"Valid molecules: {total_valid} ({total_valid / total_generated * 100:.1f}%)"
    )
    print(f"Output directory: {output_dir}")

    print(f"\nPer-pocket breakdown:")
    for result in all_results:
        print(
            f"  {result['pocket_name']}: {result['n_generated']}/{result['n_requested']} valid"
        )

    # Save summary
    import json

    summary_path = output_dir.parent / "generation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(
            {
                "total_pockets": n_pockets,
                "n_samples_per_pocket": n_samples,
                "total_requested": total_generated,
                "total_valid": total_valid,
                "validity_rate": total_valid / total_generated,
                "pockets": all_results,
            },
            f,
            indent=2,
        )

    print(f"\n✓ Summary saved to: {summary_path}")

    return True


if __name__ == "__main__":
    success = generate_molecules()

    if success:
        print("\n" + "=" * 80)
        print("STEP 2 COMPLETE")
        print("=" * 80)
        print("\nNext step: Run 3_compute_metrics.py")
    else:
        print("\n" + "=" * 80)
        print("STEP 2 FAILED")
        print("=" * 80)
        sys.exit(1)
