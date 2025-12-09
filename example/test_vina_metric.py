"""
Example: Using AutoDock Vina docking metric for evaluating generated ligands.

This demonstrates how to integrate Vina scoring into the existing metrics workflow.
"""

from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rdkit import Chem
from analysis.vina_docking import vina_score, dock_molecules
from analysis.metrics import MoleculeProperties


def main():
    """Run comprehensive docking evaluation."""

    # Paths
    example_dir = Path(__file__).parent
    receptor = example_dir / '3rfm_receptor.pdbqt'
    ligand_sdf = example_dir / '3rfm_mol.sdf'

    print("=" * 60)
    print("AutoDock Vina Docking Metric Example")
    print("=" * 60)
    print()

    # Check files exist
    if not receptor.exists():
        print(f"ERROR: Receptor not found: {receptor}")
        print("Please prepare receptor using:")
        print("  obabel 3rfm.pdb -O 3rfm_receptor.pdbqt -xr")
        return 1

    if not ligand_sdf.exists():
        print(f"ERROR: Ligand not found: {ligand_sdf}")
        return 1

    # Load ligand
    print(f"Loading ligand from: {ligand_sdf.name}")
    suppl = Chem.SDMolSupplier(str(ligand_sdf), sanitize=False)
    mols = [mol for mol in suppl if mol is not None]

    if not mols:
        print("ERROR: No valid molecules in SDF file")
        return 1

    print(f"  ✓ Loaded {len(mols)} molecule(s)")
    print()

    # Get molecule properties
    mol = mols[0]
    print(f"Molecule properties:")
    print(f"  Atoms: {mol.GetNumAtoms()}")
    print(f"  Bonds: {mol.GetNumBonds()}")

    conf = mol.GetConformer()
    center = tuple(conf.GetPositions().mean(axis=0))
    print(f"  Center: ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f})")
    print()

    # Calculate molecular properties
    props = MoleculeProperties()
    try:
        Chem.SanitizeMol(mol)
        qed = props.calculate_qed(mol)
        sa = props.calculate_sa(mol)
        logp = props.calculate_logp(mol)
        lipinski = props.calculate_lipinski(mol)

        print("Molecular properties:")
        print(f"  QED (drug-likeness): {qed:.3f}")
        print(f"  SA (synthetic accessibility): {sa:.3f}")
        print(f"  LogP (lipophilicity): {logp:.3f}")
        print(f"  Lipinski rules passed: {lipinski}/5")
        print()
    except Exception as e:
        print(f"Warning: Could not calculate properties: {e}")
        print()

    # === Docking Evaluation ===
    print("-" * 60)
    print("DOCKING EVALUATION")
    print("-" * 60)
    print()

    # Method 1: Quick scoring (no optimization)
    print("1. Score-only mode (fast, no optimization):")
    print("   Use case: Quick filtering of many molecules")
    print()
    score_quick = vina_score(
        mol,
        receptor,
        center=center,
        box_size=(25, 25, 25),
        score_only=True
    )
    print(f"   Score: {score_quick:.3f} kcal/mol")
    print(f"   Note: High score indicates poor fit (clashes/bad geometry)")
    print()

    # Method 2: Full docking with optimization
    print("2. Full docking mode (slower, with optimization):")
    print("   Use case: Final evaluation of best candidates")
    print()
    scores_full = vina_score(
        mol,
        receptor,
        center=center,
        box_size=(25, 25, 25),
        exhaustiveness=16,  # Higher = more thorough
        score_only=False
    )
    print(f"   Best binding affinity: {scores_full[0]:.3f} kcal/mol")
    if len(scores_full) > 1:
        print(f"   Alternative modes: {[f'{s:.2f}' for s in scores_full[1:]]}")
    print()

    # Interpretation
    print("-" * 60)
    print("INTERPRETATION")
    print("-" * 60)
    print()
    print("Binding affinity scale (kcal/mol):")
    print("  < -10.0  : Very strong binding (rare)")
    print("   -9.0 to -10.0 : Strong binding")
    print("   -7.0 to -9.0  : Good binding (typical for drugs)")
    print("   -5.0 to -7.0  : Moderate binding")
    print("   -3.0 to -5.0  : Weak binding")
    print("   > -3.0  : Very weak/no binding")
    print()

    best_score = scores_full[0]
    if best_score < -9.0:
        category = "EXCELLENT"
    elif best_score < -7.0:
        category = "GOOD"
    elif best_score < -5.0:
        category = "MODERATE"
    elif best_score < -3.0:
        category = "WEAK"
    else:
        category = "POOR"

    print(f"This molecule: {best_score:.2f} kcal/mol → {category} binding")
    print()

    # === Batch processing example ===
    print("-" * 60)
    print("BATCH PROCESSING")
    print("-" * 60)
    print()

    if len(mols) > 1:
        print(f"Processing {len(mols)} molecules...")
        batch_scores = dock_molecules(
            mols,
            receptor,
            center=center,
            box_size=(25, 25, 25),
            score_only=False
        )
        for i, score in enumerate(batch_scores):
            print(f"  Molecule {i+1}: {score:.3f} kcal/mol")
    else:
        print("Only one molecule in file. Batch example skipped.")
        print()
        print("To test batch processing, provide an SDF with multiple molecules:")
        print("  scores = dock_molecules(mols, receptor, ...)")

    print()
    print("=" * 60)
    print("✓ Docking metric test completed successfully!")
    print("=" * 60)

    return 0


if __name__ == '__main__':
    sys.exit(main())
