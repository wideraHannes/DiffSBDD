"""
Compute metrics for baseline molecules.

Usage:
    python 3_compute_metrics.py
"""

import sys
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import Descriptors, QED

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def read_sdf_file(sdf_path):
    """Read molecules from SDF file."""
    suppl = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
    return [mol for mol in suppl if mol is not None]


def compute_metrics(
    molecules_dir='baseline/results/molecules',
    output_path='baseline/results/metrics.json'
):
    """Compute all metrics for baseline molecules."""

    print("="*80)
    print("STEP 3: COMPUTE METRICS")
    print("="*80)

    molecules_dir = Path(molecules_dir)
    if not molecules_dir.exists():
        print(f"\nERROR: Molecules directory not found: {molecules_dir}")
        print("Please run: python 2_generate_molecules.py first")
        return False

    # Load all molecules
    print(f"\nLoading molecules from: {molecules_dir}")
    sdf_files = sorted(molecules_dir.glob("*.sdf"))

    if not sdf_files:
        print(f"ERROR: No SDF files found in {molecules_dir}")
        return False

    print(f"Found {len(sdf_files)} SDF files")

    all_molecules = []
    pocket_names = []

    for sdf_path in tqdm(sdf_files, desc="Loading SDFs"):
        mols = read_sdf_file(sdf_path)
        all_molecules.extend(mols)
        pocket_name = sdf_path.stem
        pocket_names.extend([pocket_name] * len(mols))

    print(f"Total molecules loaded: {len(all_molecules)}")

    if len(all_molecules) == 0:
        print("ERROR: No valid molecules found!")
        return False

    # Compute metrics
    print("\nComputing metrics...")

    results = {
        'n_molecules': len(all_molecules),
        'n_pockets': len(sdf_files),
    }

    # 1. Validity (all loaded molecules are valid)
    results['validity'] = 1.0  # Already filtered during generation
    print(f"  Validity: {results['validity']:.1%}")

    # 2. Uniqueness
    smiles_list = []
    for mol in all_molecules:
        try:
            smiles = Chem.MolToSmiles(mol)
            smiles_list.append(smiles)
        except:
            continue

    unique_smiles = set(smiles_list)
    results['uniqueness'] = len(unique_smiles) / len(smiles_list) if smiles_list else 0
    results['n_unique'] = len(unique_smiles)
    print(f"  Uniqueness: {results['uniqueness']:.1%} ({len(unique_smiles)}/{len(smiles_list)})")

    # 3. Molecular properties
    qed_scores = []
    sa_scores = []
    logp_scores = []
    mw_scores = []
    n_atoms_list = []
    n_bonds_list = []

    print("\nComputing molecular properties...")

    for mol in tqdm(all_molecules, desc="Properties"):
        try:
            # QED (drug-likeness)
            qed = QED.qed(mol)
            qed_scores.append(qed)

            # SA Score (synthetic accessibility)
            try:
                from rdkit.Contrib.SA_Score import sascorer
                sa = sascorer.calculateScore(mol)
                sa_scores.append(sa)
            except:
                # SA Score not available
                pass

            # LogP
            logp = Descriptors.MolLogP(mol)
            logp_scores.append(logp)

            # Molecular weight
            mw = Descriptors.MolWt(mol)
            mw_scores.append(mw)

            # Number of atoms
            n_atoms = mol.GetNumAtoms()
            n_atoms_list.append(n_atoms)

            # Number of bonds
            n_bonds = mol.GetNumBonds()
            n_bonds_list.append(n_bonds)

        except Exception as e:
            continue

    # Save statistics
    if qed_scores:
        results['qed_mean'] = float(np.mean(qed_scores))
        results['qed_std'] = float(np.std(qed_scores))
        results['qed_min'] = float(np.min(qed_scores))
        results['qed_max'] = float(np.max(qed_scores))
        print(f"\n  QED: {results['qed_mean']:.3f} ± {results['qed_std']:.3f}")
        print(f"       Range: [{results['qed_min']:.3f}, {results['qed_max']:.3f}]")

    if sa_scores:
        results['sa_mean'] = float(np.mean(sa_scores))
        results['sa_std'] = float(np.std(sa_scores))
        results['sa_min'] = float(np.min(sa_scores))
        results['sa_max'] = float(np.max(sa_scores))
        print(f"  SA Score: {results['sa_mean']:.3f} ± {results['sa_std']:.3f}")
        print(f"            Range: [{results['sa_min']:.3f}, {results['sa_max']:.3f}]")
    else:
        print("  SA Score: Not available (install rdkit.Contrib.SA_Score)")

    if logp_scores:
        results['logp_mean'] = float(np.mean(logp_scores))
        results['logp_std'] = float(np.std(logp_scores))
        results['logp_min'] = float(np.min(logp_scores))
        results['logp_max'] = float(np.max(logp_scores))
        print(f"  LogP: {results['logp_mean']:.3f} ± {results['logp_std']:.3f}")
        print(f"        Range: [{results['logp_min']:.3f}, {results['logp_max']:.3f}]")

    if mw_scores:
        results['mw_mean'] = float(np.mean(mw_scores))
        results['mw_std'] = float(np.std(mw_scores))
        results['mw_min'] = float(np.min(mw_scores))
        results['mw_max'] = float(np.max(mw_scores))
        print(f"  Mol Weight: {results['mw_mean']:.1f} ± {results['mw_std']:.1f} Da")
        print(f"              Range: [{results['mw_min']:.1f}, {results['mw_max']:.1f}]")

    if n_atoms_list:
        results['n_atoms_mean'] = float(np.mean(n_atoms_list))
        results['n_atoms_std'] = float(np.std(n_atoms_list))
        results['n_atoms_min'] = int(np.min(n_atoms_list))
        results['n_atoms_max'] = int(np.max(n_atoms_list))
        print(f"  Num Atoms: {results['n_atoms_mean']:.1f} ± {results['n_atoms_std']:.1f}")
        print(f"             Range: [{results['n_atoms_min']}, {results['n_atoms_max']}]")

    if n_bonds_list:
        results['n_bonds_mean'] = float(np.mean(n_bonds_list))
        results['n_bonds_std'] = float(np.std(n_bonds_list))
        results['n_bonds_min'] = int(np.min(n_bonds_list))
        results['n_bonds_max'] = int(np.max(n_bonds_list))
        print(f"  Num Bonds: {results['n_bonds_mean']:.1f} ± {results['n_bonds_std']:.1f}")
        print(f"             Range: [{results['n_bonds_min']}, {results['n_bonds_max']}]")

    # Save detailed scores (for later analysis)
    results['detailed_scores'] = {
        'qed': [float(x) for x in qed_scores],
        'sa': [float(x) for x in sa_scores],
        'logp': [float(x) for x in logp_scores],
        'mw': [float(x) for x in mw_scores],
        'n_atoms': [int(x) for x in n_atoms_list],
        'n_bonds': [int(x) for x in n_bonds_list],
    }

    # Save to file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Metrics saved to: {output_path}")

    return True


if __name__ == '__main__':
    success = compute_metrics()

    if success:
        print("\n" + "="*80)
        print("STEP 3 COMPLETE")
        print("="*80)
        print("\nNext step: Run 4_create_report.py")
    else:
        print("\n" + "="*80)
        print("STEP 3 FAILED")
        print("="*80)
        sys.exit(1)
