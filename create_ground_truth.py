#!/usr/bin/env python3
"""
Create ground truth properties CSV from test dataset ligands
"""

import pandas as pd
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Descriptors, QED, Crippen
from analysis.SA_Score.sascorer import calculateScore

# Test dataset directory
test_dir = Path("data/dummy_testing_dataset_10_tests/test")

# Find all SDF files (ground truth ligands)
sdf_files = sorted(test_dir.glob("*.sdf"))

print(f"Found {len(sdf_files)} ground truth ligand files")

# Calculate properties
properties = {
    "pocket_name": [],
    "molecular_weights": [],
    "logp_values": [],
    "hbd_counts": [],
    "hba_counts": [],
    "rotatable_bonds": [],
    "num_atoms": [],
    "num_heavy_atoms": [],
    "qed_scores": [],
    "sa_scores": [],
}

for sdf_file in sdf_files:
    suppl = Chem.SDMolSupplier(str(sdf_file), sanitize=False)

    for mol in suppl:
        if mol is None:
            continue

        try:
            Chem.SanitizeMol(mol)

            pocket_name = sdf_file.stem.replace("_" + sdf_file.stem.split("_")[-1], "")
            properties["pocket_name"].append(pocket_name)
            properties["molecular_weights"].append(Descriptors.MolWt(mol))
            properties["logp_values"].append(Crippen.MolLogP(mol))
            properties["hbd_counts"].append(Descriptors.NumHDonors(mol))
            properties["hba_counts"].append(Descriptors.NumHAcceptors(mol))
            properties["rotatable_bonds"].append(Descriptors.NumRotatableBonds(mol))
            properties["num_atoms"].append(mol.GetNumAtoms())
            properties["num_heavy_atoms"].append(mol.GetNumHeavyAtoms())

            try:
                properties["qed_scores"].append(QED.qed(mol))
            except:
                properties["qed_scores"].append(0.0)

            try:
                properties["sa_scores"].append(calculateScore(mol))
            except:
                properties["sa_scores"].append(None)

        except Exception as e:
            print(f"Error processing {sdf_file}: {e}")
            continue

# Create DataFrame
df = pd.DataFrame(properties)

# Save to CSV
output_file = Path("data/dummy_testing_dataset_10_tests/ground_truth_properties.csv")
df.to_csv(output_file, index=False)

print(f"\nGround truth properties saved to: {output_file}")
print(f"Total molecules: {len(df)}")
print("\nSummary statistics:")
print(df.describe())
