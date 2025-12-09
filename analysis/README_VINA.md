# AutoDock Vina Docking Metric

Clean implementation of AutoDock Vina docking for evaluating generated ligands.

## Overview

This module provides Python interfaces to AutoDock Vina (v1.2.7+) for scoring and docking molecular structures against protein receptors.

**Key Features:**
- ✅ Clean API matching existing metrics (QED, SA, LogP, etc.)
- ✅ Both score-only (fast) and full docking (thorough) modes
- ✅ Batch processing of multiple molecules
- ✅ Automatic file format conversions (RDKit → PDBQT)
- ✅ Comprehensive error handling

## Requirements

- **AutoDock Vina** binary available in PATH (tested with v1.2.7)
  - Install: `brew install autodock-vina` (macOS) or download from [GitHub](https://github.com/ccsb-scripps/AutoDock-Vina/releases)
  - Verify: `vina --version`

- **OpenBabel** for file conversions (already available via RDKit)

## Quick Start

### 1. Prepare Receptor

Convert your protein PDB to PDBQT format:

```bash
obabel protein.pdb -O protein_receptor.pdbqt -xr
```

**Note:** The `-xr` flag is crucial - it marks this as a rigid receptor (no ROOT tags).

### 2. Score Molecules

```python
from rdkit import Chem
from analysis.vina_docking import vina_score

# Load molecule
mol = Chem.MolFromSmiles('CC(=O)Oc1ccccc1C(=O)O')  # Aspirin

# Quick scoring (fast, no optimization)
score = vina_score(
    mol,
    receptor_file='protein_receptor.pdbqt',
    center=(10.0, 15.0, 20.0),  # Binding site center
    box_size=(25, 25, 25),       # Search box dimensions
    score_only=True
)

print(f"Binding affinity: {score:.2f} kcal/mol")
```

### 3. Full Docking (with optimization)

```python
# Full docking with pose optimization
scores = vina_score(
    mol,
    receptor_file='protein_receptor.pdbqt',
    center=(10.0, 15.0, 20.0),
    box_size=(25, 25, 25),
    exhaustiveness=16,  # Higher = more thorough
    score_only=False    # Enable full docking
)

print(f"Best pose: {scores[0]:.2f} kcal/mol")
print(f"Alternative poses: {scores[1:]}")
```

### 4. Batch Processing

```python
from analysis.vina_docking import dock_molecules

# Score multiple molecules at once
mols = [mol1, mol2, mol3]  # List of RDKit molecules

scores = dock_molecules(
    mols,
    receptor_file='protein_receptor.pdbqt',
    center=(10.0, 15.0, 20.0),
    box_size=(25, 25, 25),
    score_only=True
)

for i, score in enumerate(scores):
    print(f"Molecule {i+1}: {score:.2f} kcal/mol")
```

## API Reference

### Main Functions

#### `vina_score(rdmols, receptor_file, **kwargs)`

Convenience wrapper for scoring molecules.

**Args:**
- `rdmols`: Single RDKit Mol or list of Mols
- `receptor_file`: Path to receptor PDBQT file
- `center`: (x, y, z) tuple for box center in Angstroms
- `box_size`: (x, y, z) tuple for box dimensions (default: 20, 20, 20)
- `exhaustiveness`: Search thoroughness, 1-100 (default: 8)
- `score_only`: If True, skip optimization (default: True)

**Returns:**
- Single score (if single mol + score_only=True)
- List of scores otherwise

---

#### `dock_molecules(rdmols, receptor_file, **kwargs)`

Batch docking with automatic error handling.

**Args:**
- Same as `vina_score` but expects list of molecules
- `cleanup`: Remove temporary files (default: True)

**Returns:**
- List of scores (np.nan for failed molecules)

---

#### `calculate_vina_score(receptor_file, ligand_file, center, box_size, ...)`

Low-level interface for pre-prepared PDBQT files.

## Score Interpretation

Binding affinity is reported in **kcal/mol** (more negative = stronger binding):

| Score Range | Category | Description |
|-------------|----------|-------------|
| < -10.0 | Excellent | Very strong binding (rare) |
| -9.0 to -10.0 | Strong | Clinical drug candidates |
| -7.0 to -9.0 | Good | Typical for approved drugs |
| -5.0 to -7.0 | Moderate | Lead compounds, optimizable |
| -3.0 to -5.0 | Weak | Early hits, need improvement |
| > -3.0 | Poor | Likely non-binders |

**Important Notes:**
1. Scores are **estimates**, not experimentally measured affinities
2. Use for **ranking** molecules, not absolute predictions
3. Combine with other metrics (QED, SA, diversity) for holistic evaluation

## Advanced Usage

### Determining Box Center

```python
# Option 1: Use reference ligand centroid
ref_mol = Chem.SDMolSupplier('reference_ligand.sdf')[0]
conf = ref_mol.GetConformer()
center = tuple(conf.GetPositions().mean(axis=0))

# Option 2: Specify binding site manually (e.g., from literature)
center = (15.2, -8.7, 22.4)
```

### Saving Docked Poses

```python
from pathlib import Path
from analysis.vina_docking import calculate_vina_score, rdmol_to_pdbqt

# Prepare ligand PDBQT
ligand_pdbqt = Path('ligand.pdbqt')
rdmol_to_pdbqt(mol, ligand_pdbqt)

# Dock and save output
scores = calculate_vina_score(
    receptor_file=Path('receptor.pdbqt'),
    ligand_file=ligand_pdbqt,
    center=(10, 15, 20),
    box_size=(25, 25, 25),
    output_file=Path('docked_poses.pdbqt')  # Save result
)

print(f"Docked poses saved to: docked_poses.pdbqt")
```

### Custom Exhaustiveness

For challenging targets, increase search thoroughness:

```python
# Quick screening
score = vina_score(mol, receptor, exhaustiveness=1, ...)  # Fast

# Standard evaluation
score = vina_score(mol, receptor, exhaustiveness=8, ...)  # Default

# Publication-quality
score = vina_score(mol, receptor, exhaustiveness=32, ...)  # Thorough
```

## Integration with Existing Metrics

Combine Vina with molecular properties:

```python
from analysis.metrics import MoleculeProperties
from analysis.vina_docking import vina_score

props = MoleculeProperties()

# Evaluate molecule
qed = props.calculate_qed(mol)
sa = props.calculate_sa(mol)
vina = vina_score(mol, receptor, ...)

print(f"QED: {qed:.3f}, SA: {sa:.3f}, Vina: {vina:.2f}")

# Filter criteria
is_drug_like = qed > 0.5 and sa > 0.5 and vina < -6.0
```

## Example

See `example/test_vina_metric.py` for a comprehensive demonstration:

```bash
cd example
uv run python test_vina_metric.py
```

## Troubleshooting

### "PDBQT parsing error: Unknown or inappropriate tag found in rigid receptor"

**Problem:** Receptor file contains ROOT tags (ligand format).

**Solution:** Re-prepare receptor with proper flags:
```bash
obabel protein.pdb -O protein_receptor.pdbqt -xr
```

### "Could not parse scores from Vina output"

**Problem:** Vina failed silently or produced unexpected output.

**Solution:** Check Vina stderr for errors. Common issues:
- Box doesn't overlap with receptor
- Ligand has invalid structure
- Receptor has missing atoms

### Very high scores (> 100 kcal/mol)

**Problem:** Molecule has severe clashes/bad geometry.

**Solution:**
- Use `score_only=False` to enable optimization
- Check molecule structure (sanitization, hydrogens)
- Ensure box is properly centered on binding site

## Performance

| Mode | Speed | Use Case |
|------|-------|----------|
| score_only=True | ~1-2s/mol | Quick filtering, initial screening |
| score_only=False, exhaustiveness=8 | ~10-30s/mol | Standard evaluation |
| score_only=False, exhaustiveness=32 | ~1-5min/mol | Final validation |

**Tip:** Use score_only for initial filtering, then full docking for top candidates.

## References

1. **Vina 1.2.0:** Eberhardt et al., J. Chem. Inf. Model. (2021)
   DOI: [10.1021/acs.jcim.1c00203](https://doi.org/10.1021/acs.jcim.1c00203)

2. **Original Vina:** Trott & Olson, J. Comp. Chem. (2010)
   DOI: [10.1002/jcc.21334](https://doi.org/10.1002/jcc.21334)

3. **Documentation:** https://autodock-vina.readthedocs.io/

## License

This wrapper is part of the DiffSBDD project. AutoDock Vina is licensed under Apache License 2.0.
