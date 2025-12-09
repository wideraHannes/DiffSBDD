#!/usr/bin/env python3
"""
Create ground truth properties CSV with Vina docking scores from test dataset.

This extends create_ground_truth.py by adding AutoDock Vina docking scores
for each ligand-receptor pair in the test dataset.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Descriptors, QED, Crippen
from analysis.SA_Score.sascorer import calculateScore
from analysis.vina_docking import vina_score
import subprocess
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')


def prepare_receptor(pdb_file: Path, output_dir: Path) -> Path:
    """
    Convert PDB receptor to PDBQT format using OpenBabel.

    Args:
        pdb_file: Input PDB file
        output_dir: Directory for output files

    Returns:
        Path to created PDBQT file
    """
    pdbqt_file = output_dir / f"{pdb_file.stem}.pdbqt"

    # Skip if already exists
    if pdbqt_file.exists():
        return pdbqt_file

    # Convert PDB to PDBQT using obabel with -xr flag (rigid receptor)
    cmd = f'obabel {pdb_file} -O {pdbqt_file} -xr'
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"Receptor preparation failed: {result.stderr}")

    return pdbqt_file


def find_receptor_ligand_pairs(test_dir: Path):
    """
    Find matching receptor (PDB) and ligand (SDF) pairs in test directory.

    Expected naming pattern:
    - Receptor: {name}-pocket10.pdb
    - Ligand: {name}-pocket10_{name}.sdf

    Returns:
        List of (receptor_path, ligand_path, pocket_name) tuples
    """
    pairs = []

    # Find all PDB files (receptors)
    pdb_files = sorted(test_dir.glob("*-pocket10.pdb"))

    for pdb_file in pdb_files:
        # Extract pocket name (remove -pocket10 suffix)
        pocket_name = pdb_file.stem.replace("-pocket10", "")

        # Find corresponding SDF file - pattern: {name}-pocket10_{name}.sdf
        sdf_pattern = f"{pocket_name}-pocket10_{pocket_name}.sdf"
        sdf_file = test_dir / sdf_pattern

        if sdf_file.exists():
            pairs.append((pdb_file, sdf_file, pocket_name))
        else:
            print(f"Warning: No ligand found for receptor {pdb_file.name}")
            print(f"  Expected: {sdf_pattern}")

    return pairs


def calculate_vina_for_pair(receptor_pdb: Path, ligand_sdf: Path,
                            receptor_pdbqt_dir: Path,
                            exhaustiveness: int = 8,
                            score_only: bool = False):
    """
    Calculate Vina docking score for a receptor-ligand pair.

    Args:
        receptor_pdb: Receptor PDB file
        ligand_sdf: Ligand SDF file
        receptor_pdbqt_dir: Directory for prepared receptors
        exhaustiveness: Vina exhaustiveness parameter
        score_only: If True, use score_only mode (faster but less accurate)

    Returns:
        Vina score (kcal/mol) or np.nan if failed
    """
    try:
        # Prepare receptor if needed
        receptor_pdbqt = prepare_receptor(receptor_pdb, receptor_pdbqt_dir)

        # Load ligand
        suppl = Chem.SDMolSupplier(str(ligand_sdf), sanitize=False)
        mol = next((m for m in suppl if m is not None), None)

        if mol is None:
            return np.nan

        # Get ligand center for box placement
        conf = mol.GetConformer()
        center = tuple(conf.GetPositions().mean(axis=0))

        # Run Vina docking
        score = vina_score(
            mol,
            receptor_pdbqt,
            center=center,
            box_size=(25, 25, 25),
            exhaustiveness=exhaustiveness,
            score_only=score_only
        )

        # If full docking (returns list), take best score
        if isinstance(score, list):
            return score[0] if score else np.nan

        return score

    except Exception as e:
        print(f"  Error calculating Vina score: {e}")
        return np.nan


def main():
    """Main analysis workflow."""

    # Configuration
    test_dir = Path("data/dummy_testing_dataset_10_tests/test")
    output_dir = Path("data/dummy_testing_dataset_10_tests")
    receptor_pdbqt_dir = output_dir / "prepared_receptors"
    receptor_pdbqt_dir.mkdir(exist_ok=True)

    # Vina parameters
    USE_SCORE_ONLY = False  # Set to True for faster screening, False for accurate scores
    EXHAUSTIVENESS = 16     # Higher = more thorough (8=default, 16=recommended, 32=publication)

    print("=" * 80)
    print("Ground Truth Analysis with Vina Docking")
    print("=" * 80)
    print(f"\nTest directory: {test_dir}")
    print(f"Vina mode: {'score_only (fast)' if USE_SCORE_ONLY else 'full docking (accurate)'}")
    print(f"Exhaustiveness: {EXHAUSTIVENESS}")
    print()

    # Find receptor-ligand pairs
    print("Finding receptor-ligand pairs...")
    pairs = find_receptor_ligand_pairs(test_dir)
    print(f"Found {len(pairs)} receptor-ligand pairs")
    print()

    if len(pairs) == 0:
        print("Error: No receptor-ligand pairs found!")
        return 1

    # Initialize results storage
    properties = {
        "pocket_name": [],
        "receptor_file": [],
        "ligand_file": [],
        "num_atoms": [],
        "num_heavy_atoms": [],
        "molecular_weight": [],
        "logp": [],
        "hbd": [],
        "hba": [],
        "rotatable_bonds": [],
        "qed": [],
        "sa_score": [],
        "vina_score": [],
        "vina_mode": [],
    }

    # Process each pair
    print("Processing receptor-ligand pairs...")
    print("-" * 80)

    for receptor_pdb, ligand_sdf, pocket_name in tqdm(pairs, desc="Docking"):
        try:
            # Load ligand
            suppl = Chem.SDMolSupplier(str(ligand_sdf), sanitize=False)
            mol = next((m for m in suppl if m is not None), None)

            if mol is None:
                print(f"Warning: Could not load molecule from {ligand_sdf}")
                continue

            # Sanitize molecule
            Chem.SanitizeMol(mol)

            # Calculate molecular properties
            properties["pocket_name"].append(pocket_name)
            properties["receptor_file"].append(receptor_pdb.name)
            properties["ligand_file"].append(ligand_sdf.name)
            properties["num_atoms"].append(mol.GetNumAtoms())
            properties["num_heavy_atoms"].append(mol.GetNumHeavyAtoms())
            properties["molecular_weight"].append(Descriptors.MolWt(mol))
            properties["logp"].append(Crippen.MolLogP(mol))
            properties["hbd"].append(Descriptors.NumHDonors(mol))
            properties["hba"].append(Descriptors.NumHAcceptors(mol))
            properties["rotatable_bonds"].append(Descriptors.NumRotatableBonds(mol))

            # QED
            try:
                properties["qed"].append(QED.qed(mol))
            except:
                properties["qed"].append(np.nan)

            # SA Score
            try:
                sa = calculateScore(mol)
                # Convert to 0-1 scale (higher is better)
                sa_normalized = (10 - sa) / 9
                properties["sa_score"].append(sa_normalized)
            except:
                properties["sa_score"].append(np.nan)

            # Vina docking score
            vina = calculate_vina_for_pair(
                receptor_pdb,
                ligand_sdf,
                receptor_pdbqt_dir,
                exhaustiveness=EXHAUSTIVENESS,
                score_only=USE_SCORE_ONLY
            )
            properties["vina_score"].append(vina)
            properties["vina_mode"].append("score_only" if USE_SCORE_ONLY else "full_docking")

        except Exception as e:
            print(f"Error processing {pocket_name}: {e}")
            continue

    print()
    print("-" * 80)

    # Create DataFrame
    df = pd.DataFrame(properties)

    # Save to CSV
    mode_suffix = "score_only" if USE_SCORE_ONLY else f"exhaustiveness{EXHAUSTIVENESS}"
    output_file = output_dir / f"ground_truth_with_vina_{mode_suffix}.csv"
    df.to_csv(output_file, index=False)

    print(f"\n✓ Results saved to: {output_file}")
    print(f"  Total molecules: {len(df)}")

    # Print summary statistics
    print("\n" + "=" * 80)
    print("Summary Statistics")
    print("=" * 80)
    print()

    # Molecular properties
    print("Molecular Properties:")
    print(f"  Heavy atoms: {df['num_heavy_atoms'].mean():.1f} ± {df['num_heavy_atoms'].std():.1f}")
    print(f"  Molecular weight: {df['molecular_weight'].mean():.1f} ± {df['molecular_weight'].std():.1f}")
    print(f"  LogP: {df['logp'].mean():.2f} ± {df['logp'].std():.2f}")
    print(f"  Rotatable bonds: {df['rotatable_bonds'].mean():.1f} ± {df['rotatable_bonds'].std():.1f}")
    print()

    # Drug-likeness metrics
    print("Drug-likeness Metrics:")
    print(f"  QED: {df['qed'].mean():.3f} ± {df['qed'].std():.3f}")
    print(f"  SA Score: {df['sa_score'].mean():.3f} ± {df['sa_score'].std():.3f}")
    print()

    # Vina scores
    vina_valid = df['vina_score'].dropna()
    if len(vina_valid) > 0:
        print("Vina Docking Scores:")
        print(f"  Mean: {vina_valid.mean():.3f} ± {vina_valid.std():.3f} kcal/mol")
        print(f"  Median: {vina_valid.median():.3f} kcal/mol")
        print(f"  Range: [{vina_valid.min():.3f}, {vina_valid.max():.3f}] kcal/mol")
        print(f"  Valid scores: {len(vina_valid)}/{len(df)} ({100*len(vina_valid)/len(df):.1f}%)")
        print()

        # Categorize by binding strength
        excellent = (vina_valid < -10.0).sum()
        strong = ((vina_valid >= -10.0) & (vina_valid < -9.0)).sum()
        good = ((vina_valid >= -9.0) & (vina_valid < -7.0)).sum()
        moderate = ((vina_valid >= -7.0) & (vina_valid < -5.0)).sum()
        weak = ((vina_valid >= -5.0) & (vina_valid < -3.0)).sum()
        poor = (vina_valid >= -3.0).sum()

        print("  Binding strength distribution:")
        print(f"    Excellent (< -10.0): {excellent} ({100*excellent/len(vina_valid):.1f}%)")
        print(f"    Strong (-10.0 to -9.0): {strong} ({100*strong/len(vina_valid):.1f}%)")
        print(f"    Good (-9.0 to -7.0): {good} ({100*good/len(vina_valid):.1f}%)")
        print(f"    Moderate (-7.0 to -5.0): {moderate} ({100*moderate/len(vina_valid):.1f}%)")
        print(f"    Weak (-5.0 to -3.0): {weak} ({100*weak/len(vina_valid):.1f}%)")
        print(f"    Poor (> -3.0): {poor} ({100*poor/len(vina_valid):.1f}%)")
    else:
        print("Vina Docking Scores: No valid scores calculated")

    print()
    print("=" * 80)
    print("✓ Analysis complete!")
    print("=" * 80)

    return 0


if __name__ == '__main__':
    sys.exit(main())
