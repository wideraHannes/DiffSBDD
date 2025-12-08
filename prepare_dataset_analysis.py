#!/usr/bin/env python3
"""
Prepare ground truth analysis for dataset
Calculates: QED, SA, LogP, Lipinski for comparison with generated molecules

This should be run automatically when creating a new dataset.
"""

import pandas as pd
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Descriptors, QED, Crippen
from analysis.SA_Score.sascorer import calculateScore
import logging
import argparse
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def lipinski_compliance(mol):
    """Calculate Lipinski rule compliance (0-4 rules satisfied)"""
    try:
        Chem.SanitizeMol(mol)
        violations = 0
        if Descriptors.MolWt(mol) > 500:
            violations += 1
        if Crippen.MolLogP(mol) > 5:
            violations += 1
        if Descriptors.NumHDonors(mol) > 5:
            violations += 1
        if Descriptors.NumHAcceptors(mol) > 10:
            violations += 1
        return 4 - violations
    except:
        return 0


def prepare_ground_truth_analysis(dataset_dir, split="test"):
    """
    Analyze ground truth ligands from dataset

    Args:
        dataset_dir: Path to dataset directory (e.g., data/dummy_testing_dataset_10_tests)
        split: Which split to analyze ('test', 'val', or 'train')

    Returns:
        DataFrame with molecular properties distributions
    """

    dataset_dir = Path(dataset_dir)
    test_dir = dataset_dir / split

    if not test_dir.exists():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")

    logging.info(f"Analyzing ground truth molecules from: {test_dir}")

    # Collect all properties
    properties = {
        "pocket_name": [],
        "qed_scores": [],
        "sa_scores": [],
        "logp_values": [],
        "lipinski_scores": [],
        "molecular_weights": [],
        "hbd_counts": [],
        "hba_counts": [],
        "rotatable_bonds": [],
        "num_atoms": [],
    }

    # Process each SDF file (each pocket has 1 ground truth ligand)
    sdf_files = list(test_dir.glob("*.sdf"))
    logging.info(f"Found {len(sdf_files)} ground truth molecules")

    for sdf_file in tqdm(sdf_files, desc="Processing ground truth"):
        try:
            suppl = Chem.SDMolSupplier(str(sdf_file), sanitize=False)
            mol = suppl[0]  # Each SDF should have one molecule

            if mol is None:
                logging.warning(f"Could not load molecule from {sdf_file.name}")
                continue

            # Sanitize and calculate properties
            Chem.SanitizeMol(mol)

            properties["pocket_name"].append(sdf_file.stem)
            properties["qed_scores"].append(QED.qed(mol))
            properties["sa_scores"].append(calculateScore(mol))
            properties["logp_values"].append(Crippen.MolLogP(mol))
            properties["lipinski_scores"].append(lipinski_compliance(mol))
            properties["molecular_weights"].append(Descriptors.MolWt(mol))
            properties["hbd_counts"].append(Descriptors.NumHDonors(mol))
            properties["hba_counts"].append(Descriptors.NumHAcceptors(mol))
            properties["rotatable_bonds"].append(Descriptors.NumRotatableBonds(mol))
            properties["num_atoms"].append(mol.GetNumAtoms())

        except Exception as e:
            logging.warning(f"Error processing {sdf_file.name}: {e}")
            continue

    # Create DataFrame
    df = pd.DataFrame(properties)

    if len(df) == 0:
        logging.error("No valid molecules found in ground truth!")
        return None

    # Create analysis directory in dataset
    analysis_dir = dataset_dir / "analysis"
    analysis_dir.mkdir(exist_ok=True)

    # Save ground truth properties
    output_file = analysis_dir / f"{split}_ground_truth_properties.csv"
    df.to_csv(output_file, index=False)

    logging.info(f"✅ Ground truth analysis saved to: {output_file}")

    # Print summary statistics
    print(f"\n{'='*60}")
    print(f"GROUND TRUTH ANALYSIS: {dataset_dir.name} ({split} set)")
    print(f"{'='*60}")
    print(f"Total molecules: {len(df)}")
    print(f"\nProperty Distributions:")
    print(f"  QED:        {df['qed_scores'].mean():.3f} ± {df['qed_scores'].std():.3f}")
    print(f"  SA Score:   {df['sa_scores'].mean():.2f} ± {df['sa_scores'].std():.2f}")
    print(f"  LogP:       {df['logp_values'].mean():.2f} ± {df['logp_values'].std():.2f}")
    print(f"  Lipinski:   {df['lipinski_scores'].mean():.1f}/4")
    print(
        f"  Mol Weight: {df['molecular_weights'].mean():.1f} ± {df['molecular_weights'].std():.1f} Da"
    )
    print(
        f"  Num Atoms:  {df['num_atoms'].mean():.1f} ± {df['num_atoms'].std():.1f}"
    )
    print(f"{'='*60}\n")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare ground truth analysis for dataset"
    )
    parser.add_argument("dataset_dir", type=Path, help="Path to dataset directory")
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["test", "val", "train"],
        help="Which split to analyze (default: test)",
    )
    args = parser.parse_args()

    prepare_ground_truth_analysis(args.dataset_dir, args.split)
