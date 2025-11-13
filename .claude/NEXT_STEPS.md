# ESM-C Integration: Next Steps

**Status**: Validation Complete ✅ (6/6 tests passed)
**Date**: 2025-11-13
**Ready to proceed**: YES - all technical blockers resolved

---

## 🚨 Critical Path: Data Re-processing

**Blocker**: Current NPZ files are missing `pocket_residue_ids` field
**Impact**: Cannot broadcast ESM-C embeddings without atom→residue mapping
**Priority**: HIGHEST - must complete before any other steps

---

## Phase 0A: Baseline Evaluation (Week 1 - PARALLEL TASK)

**IMPORTANT**: Before making any changes, establish baseline performance for comparison!

### Step 0: Evaluate Existing Baseline Model

**Purpose**: Document baseline performance to compare against ESM-C model

**Checkpoint**: You should already have the pre-trained baseline model:
- `checkpoints/crossdocked_fullatom_cond.ckpt` (from Zenodo)
- If not, download from: https://zenodo.org/record/8183747

**Action Items**:
- [ ] Verify baseline checkpoint exists
- [ ] Load and test checkpoint
- [ ] Generate molecules on test set
- [ ] Compute all metrics
- [ ] Save results for comparison

---

#### Step 0.1: Verify Baseline Checkpoint

```bash
# Check if checkpoint exists
ls -lh checkpoints/crossdocked_fullatom_cond.ckpt

# If not found, download from Zenodo
# (See paper supplementary materials)
wget https://zenodo.org/record/8183747/files/crossdocked_fullatom_cond.ckpt \
    -O checkpoints/crossdocked_fullatom_cond.ckpt
```

**Test checkpoint loads**:
```python
from lightning_modules import LigandPocketDDPM

# Load checkpoint
checkpoint_path = "checkpoints/crossdocked_fullatom_cond.ckpt"
model = LigandPocketDDPM.load_from_checkpoint(checkpoint_path)
print(f"✓ Loaded model: {model.__class__.__name__}")
print(f"  Mode: {model.hparams.mode}")
print(f"  Dataset: {model.hparams.dataset}")
```

**Expected Output**:
```
✓ Loaded model: LigandPocketDDPM
  Mode: pocket_conditioning
  Dataset: crossdock
```

**Action Items**:
- [ ] Download or locate baseline checkpoint
- [ ] Verify checkpoint loads without errors
- [ ] Check model configuration matches expectations

---

#### Step 0.2: Generate Baseline Molecules (Test Set)

**Option A: Using existing test set**
```bash
# Generate 100 molecules per test pocket
python scripts/generate_baseline_test_set.py \
    --checkpoint checkpoints/crossdocked_fullatom_cond.ckpt \
    --test_set data/processed_crossdock_noH_full/test.npz \
    --n_samples 100 \
    --outdir results/baseline/molecules/ \
    --save_format sdf
```

**Option B: Manual generation script** (if script doesn't exist)

Create `scripts/generate_baseline_test_set.py`:

```python
"""
Generate molecules from baseline model for all test pockets.
"""

import os
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse

from lightning_modules import LigandPocketDDPM
from dataset import ProcessedLigandPocketDataset
from utils import write_sdf_file


def generate_baseline(checkpoint_path, test_set_path, n_samples, outdir):
    """Generate molecules from baseline for all test pockets."""

    # Load model
    print(f"Loading checkpoint: {checkpoint_path}")
    model = LigandPocketDDPM.load_from_checkpoint(checkpoint_path)
    model = model.eval().cuda()

    # Load test set
    print(f"Loading test set: {test_set_path}")
    test_data = np.load(test_set_path, allow_pickle=True)
    n_test_samples = len(test_data['names'])

    print(f"Found {n_test_samples} test pockets")
    print(f"Generating {n_samples} molecules per pocket")

    # Create output directory
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Generate for each test pocket
    all_results = []

    for i in tqdm(range(n_test_samples), desc="Generating"):
        # Extract pocket for this sample
        pocket_mask = (test_data['pocket_mask'] == i)
        pocket_coords = torch.from_numpy(test_data['pocket_coords'][pocket_mask]).float()
        pocket_one_hot = torch.from_numpy(test_data['pocket_one_hot'][pocket_mask]).float()

        # Prepare pocket dict
        pocket = {
            'x': pocket_coords.cuda(),
            'one_hot': pocket_one_hot.cuda(),
            'size': torch.tensor([len(pocket_coords)]).cuda(),
            'mask': torch.zeros(len(pocket_coords), dtype=torch.long).cuda(),
        }

        # Repeat for n_samples
        pocket_repeated = {
            'x': pocket['x'].repeat(n_samples, 1),
            'one_hot': pocket['one_hot'].repeat(n_samples, 1),
            'size': torch.tensor([len(pocket_coords)] * n_samples).cuda(),
            'mask': torch.arange(n_samples).repeat_interleave(len(pocket_coords)).cuda(),
        }

        # Sample ligand size
        num_nodes_lig = model.ddpm.size_distribution.sample_conditional(
            n1=None,
            n2=pocket_repeated['size']
        )

        # Generate molecules
        with torch.no_grad():
            xh_lig, _, _, _ = model.ddpm.sample_given_pocket(
                pocket_repeated, num_nodes_lig
            )

        # Build RDKit molecules
        from analysis.molecule_builder import build_molecule

        molecules = []
        for j in range(n_samples):
            sample_mask = (pocket_repeated['mask'] == j)
            coords = xh_lig[sample_mask, :3].cpu().numpy()
            atom_types = torch.argmax(xh_lig[sample_mask, 3:], dim=1).cpu().numpy()

            mol = build_molecule(
                coords, atom_types, model.dataset_info,
                use_openbabel=True
            )

            if mol is not None:
                molecules.append(mol)

        # Save molecules
        pocket_name = test_data['names'][i].replace('/', '_')
        sdf_path = outdir / f"{pocket_name}_baseline.sdf"
        write_sdf_file(sdf_path, molecules)

        all_results.append({
            'pocket_name': pocket_name,
            'n_generated': len(molecules),
            'sdf_path': str(sdf_path),
        })

    # Save summary
    import json
    with open(outdir / 'generation_summary.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n✓ Generation complete!")
    print(f"  Saved {len(all_results)} pockets to {outdir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--test_set', type=str, required=True)
    parser.add_argument('--n_samples', type=int, default=100)
    parser.add_argument('--outdir', type=str, default='results/baseline/molecules')

    args = parser.parse_args()

    generate_baseline(
        checkpoint_path=args.checkpoint,
        test_set_path=args.test_set,
        n_samples=args.n_samples,
        outdir=args.outdir,
    )
```

**Run generation**:
```bash
python scripts/generate_baseline_test_set.py \
    --checkpoint checkpoints/crossdocked_fullatom_cond.ckpt \
    --test_set data/processed_crossdock_noH_full/test.npz \
    --n_samples 100 \
    --outdir results/baseline/molecules/
```

**Expected time**: 2-6 hours (depends on test set size and GPU)

**Action Items**:
- [ ] Create generation script if needed
- [ ] Run generation on full test set
- [ ] Verify molecules were generated
- [ ] Check output directory has SDF files

---

#### Step 0.3: Compute Baseline Metrics

**Create `scripts/evaluate_baseline.py`**:

```python
"""
Evaluate baseline model on test set.
Computes: validity, uniqueness, novelty, QED, SA score, etc.
"""

import os
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
import argparse
import json
from rdkit import Chem

from analysis.metrics import BasicMolecularMetrics, MoleculeProperties
from utils import read_sdf_file


def compute_metrics(molecules_dir, test_set_path, output_path):
    """Compute all metrics for baseline molecules."""

    print(f"Loading molecules from: {molecules_dir}")
    molecules_dir = Path(molecules_dir)

    # Load all SDF files
    all_molecules = []
    all_pocket_names = []

    sdf_files = sorted(molecules_dir.glob("*.sdf"))
    print(f"Found {len(sdf_files)} SDF files")

    for sdf_path in tqdm(sdf_files, desc="Loading molecules"):
        mols = read_sdf_file(sdf_path)
        all_molecules.extend(mols)
        pocket_name = sdf_path.stem.replace('_baseline', '')
        all_pocket_names.extend([pocket_name] * len(mols))

    print(f"\nTotal molecules loaded: {len(all_molecules)}")

    # Initialize metrics
    basic_metrics = BasicMolecularMetrics()
    mol_properties = MoleculeProperties()

    # Compute metrics
    print("\nComputing metrics...")

    results = {
        'n_molecules': len(all_molecules),
        'n_pockets': len(sdf_files),
    }

    # Validity
    valid_mols = [m for m in all_molecules if m is not None]
    results['validity'] = len(valid_mols) / len(all_molecules)
    print(f"  Validity: {results['validity']:.3f}")

    # Uniqueness
    smiles_list = []
    for mol in valid_mols:
        try:
            smiles = Chem.MolToSmiles(mol)
            smiles_list.append(smiles)
        except:
            continue

    unique_smiles = set(smiles_list)
    results['uniqueness'] = len(unique_smiles) / len(smiles_list) if smiles_list else 0
    print(f"  Uniqueness: {results['uniqueness']:.3f}")

    # Molecular properties (QED, SA, LogP, etc.)
    qed_scores = []
    sa_scores = []
    logp_scores = []

    for mol in tqdm(valid_mols, desc="Computing properties"):
        try:
            # QED (drug-likeness)
            from rdkit.Chem import QED
            qed = QED.qed(mol)
            qed_scores.append(qed)

            # SA Score (synthetic accessibility)
            from rdkit.Contrib.SA_Score import sascorer
            sa = sascorer.calculateScore(mol)
            sa_scores.append(sa)

            # LogP
            from rdkit.Chem import Descriptors
            logp = Descriptors.MolLogP(mol)
            logp_scores.append(logp)
        except:
            continue

    results['qed_mean'] = np.mean(qed_scores)
    results['qed_std'] = np.std(qed_scores)
    results['sa_mean'] = np.mean(sa_scores)
    results['sa_std'] = np.std(sa_scores)
    results['logp_mean'] = np.mean(logp_scores)
    results['logp_std'] = np.std(logp_scores)

    print(f"\n  QED: {results['qed_mean']:.3f} ± {results['qed_std']:.3f}")
    print(f"  SA Score: {results['sa_mean']:.3f} ± {results['sa_std']:.3f}")
    print(f"  LogP: {results['logp_mean']:.3f} ± {results['logp_std']:.3f}")

    # Molecular weight
    mw_scores = [Chem.Descriptors.MolWt(m) for m in valid_mols]
    results['mol_weight_mean'] = np.mean(mw_scores)
    results['mol_weight_std'] = np.std(mw_scores)
    print(f"  Mol Weight: {results['mol_weight_mean']:.1f} ± {results['mol_weight_std']:.1f}")

    # Number of atoms
    n_atoms = [m.GetNumAtoms() for m in valid_mols]
    results['n_atoms_mean'] = np.mean(n_atoms)
    results['n_atoms_std'] = np.std(n_atoms)
    print(f"  Num Atoms: {results['n_atoms_mean']:.1f} ± {results['n_atoms_std']:.1f}")

    # Save detailed results
    results['qed_scores'] = qed_scores
    results['sa_scores'] = sa_scores
    results['logp_scores'] = logp_scores
    results['mw_scores'] = mw_scores
    results['n_atoms'] = n_atoms

    # Save to file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Saved results to: {output_path}")

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--molecules_dir', type=str,
                       default='results/baseline/molecules')
    parser.add_argument('--test_set', type=str,
                       default='data/processed_crossdock_noH_full/test.npz')
    parser.add_argument('--output', type=str,
                       default='results/baseline/baseline_metrics.json')

    args = parser.parse_args()

    compute_metrics(
        molecules_dir=args.molecules_dir,
        test_set_path=args.test_set,
        output_path=args.output,
    )
```

**Run evaluation**:
```bash
python scripts/evaluate_baseline.py \
    --molecules_dir results/baseline/molecules/ \
    --test_set data/processed_crossdock_noH_full/test.npz \
    --output results/baseline/baseline_metrics.json
```

**Expected Output**:
```
Total molecules loaded: 10000
Computing metrics...
  Validity: 0.723
  Uniqueness: 0.947

  QED: 0.452 ± 0.142
  SA Score: 3.18 ± 0.87
  LogP: 2.34 ± 1.12
  Mol Weight: 287.3 ± 78.4
  Num Atoms: 21.4 ± 5.8

✓ Saved results to: results/baseline/baseline_metrics.json
```

**Action Items**:
- [ ] Create evaluation script
- [ ] Run metrics computation
- [ ] Save results JSON
- [ ] Document baseline numbers

---

#### Step 0.4: (Optional) Docking Evaluation

**If Vina/smina available**:

```bash
# Install smina (molecular docking)
conda install -c conda-forge smina

# Or download from: https://sourceforge.net/projects/smina/
```

**Create `scripts/evaluate_docking_baseline.py`**:

```python
"""
Evaluate baseline molecules with molecular docking (Vina/smina).
"""

import os
import subprocess
from pathlib import Path
from tqdm import tqdm
import json
import argparse


def dock_molecule(receptor_pdb, ligand_sdf, output_dir):
    """Dock single molecule with smina."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run smina scoring
    cmd = [
        'smina',
        '-r', receptor_pdb,
        '-l', ligand_sdf,
        '--score_only',  # Just score, don't optimize
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    # Parse score from output
    for line in result.stdout.split('\n'):
        if 'Affinity' in line:
            score = float(line.split()[1])
            return score

    return None


def evaluate_docking(molecules_dir, receptors_dir, output_path):
    """Evaluate docking scores for all baseline molecules."""

    molecules_dir = Path(molecules_dir)
    receptors_dir = Path(receptors_dir)

    sdf_files = sorted(molecules_dir.glob("*.sdf"))

    all_scores = []

    for sdf_path in tqdm(sdf_files, desc="Docking"):
        pocket_name = sdf_path.stem.replace('_baseline', '')

        # Find corresponding receptor PDB
        # (This depends on your data organization)
        receptor_pdb = receptors_dir / f"{pocket_name}_receptor.pdb"

        if not receptor_pdb.exists():
            print(f"Warning: Receptor not found for {pocket_name}")
            continue

        # Dock
        score = dock_molecule(receptor_pdb, sdf_path,
                             Path('results/baseline/docking'))

        if score is not None:
            all_scores.append({
                'pocket_name': pocket_name,
                'vina_score': score,
            })

    # Save results
    with open(output_path, 'w') as f:
        json.dump(all_scores, f, indent=2)

    # Print summary
    import numpy as np
    scores = [s['vina_score'] for s in all_scores]
    print(f"\nDocking Summary:")
    print(f"  N molecules: {len(scores)}")
    print(f"  Mean Vina: {np.mean(scores):.2f} ± {np.std(scores):.2f}")
    print(f"  Median: {np.median(scores):.2f}")
    print(f"  Best: {np.min(scores):.2f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--molecules_dir', type=str,
                       default='results/baseline/molecules')
    parser.add_argument('--receptors_dir', type=str,
                       default='data/receptors')  # Adjust path
    parser.add_argument('--output', type=str,
                       default='results/baseline/docking_scores.json')

    args = parser.parse_args()

    evaluate_docking(
        molecules_dir=args.molecules_dir,
        receptors_dir=args.receptors_dir,
        output_path=args.output,
    )
```

**Note**: Docking requires:
- Receptor PDB files (protein structures)
- May be time-consuming
- Optional but valuable for comparison

**Action Items**:
- [ ] Install smina (if doing docking)
- [ ] Run docking evaluation (optional)
- [ ] Save docking scores

---

#### Step 0.5: Create Baseline Report

**Create `scripts/create_baseline_report.py`**:

```python
"""
Create a comprehensive baseline report with all metrics.
"""

import json
import numpy as np
from pathlib import Path


def create_report(metrics_path, docking_path=None, output_path='results/baseline/BASELINE_REPORT.md'):
    """Generate markdown report of baseline performance."""

    # Load metrics
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)

    # Load docking if available
    docking_scores = None
    if docking_path and Path(docking_path).exists():
        with open(docking_path, 'r') as f:
            docking_data = json.load(f)
            docking_scores = [d['vina_score'] for d in docking_data]

    # Create report
    report = f"""# Baseline Model Performance Report

**Date**: {import datetime; datetime.datetime.now().strftime('%Y-%m-%d')}
**Model**: DiffSBDD Baseline (CrossDocked full-atom conditional)
**Checkpoint**: `checkpoints/crossdocked_fullatom_cond.ckpt`

---

## Summary Statistics

| Metric | Value |
|--------|-------|
| Total Molecules | {metrics['n_molecules']:,} |
| Test Pockets | {metrics['n_pockets']:,} |
| **Validity** | **{metrics['validity']:.1%}** |
| **Uniqueness** | **{metrics['uniqueness']:.1%}** |

---

## Molecular Properties

### Drug-likeness (QED)
- Mean: **{metrics['qed_mean']:.3f}** ± {metrics['qed_std']:.3f}
- Range: {min(metrics['qed_scores']):.3f} - {max(metrics['qed_scores']):.3f}

### Synthetic Accessibility (SA Score)
- Mean: **{metrics['sa_mean']:.3f}** ± {metrics['sa_std']:.3f}
- Range: {min(metrics['sa_scores']):.3f} - {max(metrics['sa_scores']):.3f}
- *Lower is better (1-10 scale)*

### Lipophilicity (LogP)
- Mean: **{metrics['logp_mean']:.3f}** ± {metrics['logp_std']:.3f}
- Range: {min(metrics['logp_scores']):.3f} - {max(metrics['logp_scores']):.3f}

### Molecular Weight
- Mean: **{metrics['mol_weight_mean']:.1f}** ± {metrics['mol_weight_std']:.1f} Da
- Range: {min(metrics['mw_scores']):.1f} - {max(metrics['mw_scores']):.1f} Da

### Number of Atoms
- Mean: **{metrics['n_atoms_mean']:.1f}** ± {metrics['n_atoms_std']:.1f}
- Range: {min(metrics['n_atoms']):.0f} - {max(metrics['n_atoms']):.0f}

---
"""

    # Add docking section if available
    if docking_scores:
        report += f"""
## Docking Scores (Vina)

| Metric | Value (kcal/mol) |
|--------|------------------|
| Mean | **{np.mean(docking_scores):.2f}** ± {np.std(docking_scores):.2f} |
| Median | {np.median(docking_scores):.2f} |
| Best (min) | {np.min(docking_scores):.2f} |
| 10th percentile | {np.percentile(docking_scores, 10):.2f} |
| 90th percentile | {np.percentile(docking_scores, 90):.2f} |

*Lower (more negative) scores indicate better predicted binding affinity*

---
"""

    report += """
## Distribution Plots

(To be generated with matplotlib)

---

## Comparison Target for ESM-C Model

**Success Criteria**:
- Validity improvement: >5% (current: {:.1%})
- QED improvement: >0.05 (current: {:.3f})
- SA Score improvement: <-0.2 (current: {:.3f})
- Vina improvement: <-0.5 kcal/mol (if docking available)

**Statistical Test**: Wilcoxon signed-rank test (p < 0.05)

---

## Files

- Molecules: `results/baseline/molecules/*.sdf`
- Metrics: `results/baseline/baseline_metrics.json`
- Report: `results/baseline/BASELINE_REPORT.md`
""".format(metrics['validity'], metrics['qed_mean'], metrics['sa_mean'])

    # Save report
    with open(output_path, 'w') as f:
        f.write(report)

    print(f"✓ Baseline report saved to: {output_path}")


if __name__ == '__main__':
    create_report(
        metrics_path='results/baseline/baseline_metrics.json',
        docking_path='results/baseline/docking_scores.json',  # Optional
        output_path='results/baseline/BASELINE_REPORT.md',
    )
```

**Run**:
```bash
python scripts/create_baseline_report.py
```

**Action Items**:
- [ ] Create report generation script
- [ ] Generate baseline report
- [ ] Review metrics
- [ ] Save report for comparison

---

#### Step 0.6: Document Baseline Numbers

**Critical**: Save these baseline numbers for comparison!

**Create `results/baseline/BASELINE_SUMMARY.txt`**:

```
BASELINE MODEL PERFORMANCE (DiffSBDD)
=====================================

Checkpoint: checkpoints/crossdocked_fullatom_cond.ckpt
Test Set: CrossDocked test split
Date: 2025-11-13

KEY METRICS:
-----------
Validity:    XX.X%
Uniqueness:  XX.X%
QED:         X.XXX ± X.XXX
SA Score:    X.XX ± X.XX
Vina Score:  -X.XX ± X.XX kcal/mol (if available)

THESE ARE THE TARGET TO BEAT WITH ESM-C!

Statistical significance required: p < 0.05 (Wilcoxon test)
```

**Action Items**:
- [ ] Fill in actual baseline numbers
- [ ] Commit to git (important!)
- [ ] Reference in thesis

---

### Summary of Phase 0A

**Deliverables**:
1. Baseline checkpoint verified ✓
2. Test set molecules generated ✓
3. All metrics computed ✓
4. Baseline report created ✓
5. Numbers documented for comparison ✓

**Time Estimate**: 4-8 hours (mostly generation time)

**Can run in parallel with**: Data re-processing (Phase 0B)

**Files Created**:
- `scripts/generate_baseline_test_set.py`
- `scripts/evaluate_baseline.py`
- `scripts/evaluate_docking_baseline.py` (optional)
- `scripts/create_baseline_report.py`
- `results/baseline/baseline_metrics.json`
- `results/baseline/BASELINE_REPORT.md`
- `results/baseline/molecules/*.sdf`

---

## Phase 0B: Data Re-processing (Week 1)

### Step 1: Modify `process_crossdock.py`

**File**: `process_crossdock.py`
**Location**: Lines 105-138 (full-atom branch)
**Reference**: See `experiments/example_residue_id_tracking.py` for exact code

#### Changes Required:

```python
# BEFORE (lines 105-118):
full_atoms = np.concatenate([
    np.array([atom.element for atom in res.get_atoms()])
    for res in pocket_residues
], axis=0)

full_coords = np.concatenate([
    np.array([atom.coord for atom in res.get_atoms()])
    for res in pocket_residues
], axis=0)

# AFTER (modified):
# Track residue IDs per atom
full_atoms = []
full_coords = []
residue_ids = []  # NEW

for res_idx, res in enumerate(pocket_residues):
    for atom in res.get_atoms():
        full_atoms.append(atom.element)
        full_coords.append(atom.coord)
        residue_ids.append(res_idx)  # NEW: Track parent residue

full_atoms = np.array(full_atoms)
full_coords = np.array(full_coords)
residue_ids = np.array(residue_ids, dtype=np.int32)  # NEW
```

```python
# BEFORE (line ~134-138):
pocket_data = {
    "pocket_coords": full_coords,
    "pocket_one_hot": pocket_one_hot,
    "pocket_ids": pocket_ids,
}

# AFTER (modified):
pocket_data = {
    "pocket_coords": full_coords,
    "pocket_one_hot": pocket_one_hot,
    "pocket_ids": pocket_ids,
    "pocket_residue_ids": residue_ids,  # NEW
}
```

**Action Items**:
- [ ] Open `process_crossdock.py`
- [ ] Apply changes to lines 105-138
- [ ] Save modified file
- [ ] Test on single PDB (see Step 2)

---

### Step 2: Test on Single PDB

**Purpose**: Verify modifications work before re-processing full dataset

```bash
# Test on a single structure
python process_crossdock.py \
    /path/to/test.pdb \
    /path/to/test.sdf \
    --no_H \
    --dist_cutoff 10.0

# Verify output
python -c "
import numpy as np
data = np.load('test_output.npz', allow_pickle=True)
print('Keys:', list(data.keys()))
print('Has residue IDs:', 'pocket_residue_ids' in data.keys())
if 'pocket_residue_ids' in data.keys():
    print('Residue IDs:', data['pocket_residue_ids'])
"
```

**Expected Output**:
```
Keys: [..., 'pocket_residue_ids', ...]
Has residue IDs: True
Residue IDs: [0 0 0 0 1 1 1 1 1 2 2 2 ...]
```

**Action Items**:
- [ ] Run on single PDB file
- [ ] Verify `pocket_residue_ids` field exists
- [ ] Check residue IDs are contiguous and start from 0
- [ ] Verify number of residues matches `len(pocket_ids)`

---

### Step 3: Re-process Full Dataset

**Dataset**: CrossDock (~100K samples)
**Time estimate**: 8-16 hours (depends on hardware)
**Storage**: Keep both old and new NPZ for safety

```bash
# Create output directory
mkdir -p data/processed_crossdock_noH_full_esmc

# Re-process train set
python process_crossdock.py \
    /path/to/crossdocked/train/ \
    --no_H \
    --dist_cutoff 10.0 \
    --ca_only False \
    --output data/processed_crossdock_noH_full_esmc/train.npz

# Re-process val set
python process_crossdock.py \
    /path/to/crossdocked/val/ \
    --no_H \
    --dist_cutoff 10.0 \
    --ca_only False \
    --output data/processed_crossdock_noH_full_esmc/val.npz

# Re-process test set
python process_crossdock.py \
    /path/to/crossdocked/test/ \
    --no_H \
    --dist_cutoff 10.0 \
    --ca_only False \
    --output data/processed_crossdock_noH_full_esmc/test.npz
```

**HPC Submission** (if processing is slow):
```bash
# Create PBS script
cat > .claude/hpc/reprocess_data.pbs << 'EOF'
#!/bin/bash
#PBS -N reprocess_crossdock
#PBS -l select=1:ncpus=4:mem=32gb
#PBS -l walltime=24:00:00
#PBS -A your_project_id

cd $PBS_O_WORKDIR
module load python/3.9

python process_crossdock.py \
    /path/to/crossdocked/train/ \
    --no_H \
    --dist_cutoff 10.0 \
    --ca_only False \
    --output data/processed_crossdock_noH_full_esmc/train.npz
EOF

# Submit
qsub .claude/hpc/reprocess_data.pbs
```

**Action Items**:
- [ ] Run re-processing (locally or on HPC)
- [ ] Monitor progress (check log files)
- [ ] Verify all three splits (train/val/test) complete
- [ ] Check file sizes are similar to original NPZ files
- [ ] Run validation script on new data (see Step 4)

---

### Step 4: Validate Re-processed Data

```bash
# Check new NPZ structure
python -c "
import numpy as np

for split in ['train', 'val', 'test']:
    print(f'\n=== {split.upper()} ===')
    data = np.load(f'data/processed_crossdock_noH_full_esmc/{split}.npz', allow_pickle=True)

    print(f'Keys: {list(data.keys())}')
    print(f'Samples: {len(data[\"names\"])}')
    print(f'Pocket atoms: {len(data[\"pocket_coords\"])}')

    if 'pocket_residue_ids' in data.keys():
        res_ids = data['pocket_residue_ids']
        print(f'Residue IDs: {len(res_ids)} atoms')
        print(f'Unique residues per sample: {len(set(res_ids[:100]))}')  # First sample
        print('✓ Residue IDs present!')
    else:
        print('✗ ERROR: pocket_residue_ids missing!')
"
```

**Expected Output**:
```
=== TRAIN ===
Keys: ['names', 'lig_coords', ..., 'pocket_residue_ids']
Samples: 100000
Pocket atoms: 35301000
Residue IDs: 35301000 atoms
Unique residues per sample: ~100
✓ Residue IDs present!
```

**Action Items**:
- [ ] Verify `pocket_residue_ids` exists in all splits
- [ ] Check residue ID ranges are correct (start from 0 per sample)
- [ ] Verify total atom counts match
- [ ] Spot-check a few samples manually

---

## Phase 1: ESM-C Pre-computation (Week 1-2)

### Step 5: Install ESM-C

```bash
# Install ESM (fair-esm package)
uv pip install fair-esm

# Test installation
python -c "
from esm import pretrained
print('ESM installation successful!')
print('Available models:')
for name in dir(pretrained):
    if not name.startswith('_'):
        print(f'  - {name}')
"
```

**Model Selection**:
- `esm2_t33_650M_UR50D`: ESM-2 650M params (recommended, 1280-dim)
- `esmc_300M`: ESM-C 300M params (960-dim, if available)
- `esmc_600M`: ESM-C 600M params (960-dim, if available)

**Action Items**:
- [ ] Install fair-esm
- [ ] Test model loading locally
- [ ] Check available models
- [ ] Decide on model version (ESM-2 or ESM-C)

---

### Step 6: Write `scripts/precompute_esmc.py`

**File**: `scripts/precompute_esmc.py`

```python
"""
Pre-compute ESM-C embeddings for all pockets in dataset.

Usage:
    python scripts/precompute_esmc.py \
        --input data/processed_crossdock_noH_full_esmc/train.npz \
        --output data/processed_crossdock_esmc/train.npz \
        --model esm2_t33_650M_UR50D \
        --batch_size 32 \
        --device cuda
"""

import numpy as np
import torch
from esm import pretrained
from tqdm import tqdm
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from constants import dataset_params


def extract_sequences_from_pocket_ids(pocket_ids):
    """
    Extract amino acid sequences from pocket_ids.

    Args:
        pocket_ids: List of residue identifiers (e.g., ["A:123", "A:124", ...])

    Returns:
        sequence: String of amino acids (e.g., "MKTAYIAK...")
    """
    # TODO: Implement sequence extraction
    # Need to map residue IDs to amino acid types
    # May need to read from original PDB or infer from pocket_one_hot
    raise NotImplementedError("Sequence extraction not yet implemented")


def broadcast_to_atoms(esmc_residue_embeddings, residue_ids):
    """
    Broadcast per-residue embeddings to per-atom embeddings.

    Args:
        esmc_residue_embeddings: (n_residues, esmc_dim)
        residue_ids: (n_atoms,) - which residue each atom belongs to

    Returns:
        esmc_atom_embeddings: (n_atoms, esmc_dim)
    """
    return esmc_residue_embeddings[residue_ids]


def compute_esmc_embeddings(npz_path, model_name, batch_size, device, output_path):
    """
    Compute ESM-C embeddings for all samples in NPZ file.
    """
    print(f"Loading data from {npz_path}...")
    data = np.load(npz_path, allow_pickle=True)

    # Check for residue IDs
    if 'pocket_residue_ids' not in data.keys():
        raise ValueError("NPZ file missing 'pocket_residue_ids' field. "
                        "Re-process data with modified process_crossdock.py first!")

    # Load ESM model
    print(f"Loading ESM model: {model_name}...")
    model, alphabet = getattr(pretrained, model_name)()
    model = model.eval().to(device)
    esmc_dim = model.embed_dim

    print(f"ESM embedding dimension: {esmc_dim}")

    # Extract sequences
    print("Extracting sequences...")
    sequences = []
    n_samples = len(data['names'])

    for i in tqdm(range(n_samples)):
        pocket_ids_sample = data['pocket_ids'][i]
        sequence = extract_sequences_from_pocket_ids(pocket_ids_sample)
        sequences.append(sequence)

    # Compute embeddings in batches
    print(f"Computing ESM-C embeddings (batch_size={batch_size})...")
    all_embeddings = []

    for i in tqdm(range(0, len(sequences), batch_size)):
        batch_seqs = sequences[i:i+batch_size]

        # Tokenize
        batch_tokens = [alphabet.encode(seq) for seq in batch_seqs]
        max_len = max(len(t) for t in batch_tokens)
        batch_tokens_padded = torch.full((len(batch_tokens), max_len),
                                        alphabet.padding_idx, dtype=torch.long)

        for j, tokens in enumerate(batch_tokens):
            batch_tokens_padded[j, :len(tokens)] = tokens

        # Forward pass
        with torch.no_grad():
            batch_tokens_padded = batch_tokens_padded.to(device)
            results = model(batch_tokens_padded, repr_layers=[model.num_layers])
            embeddings = results['representations'][model.num_layers]  # (batch, len, dim)

        # Extract per-residue embeddings (skip BOS/EOS tokens)
        for j, seq in enumerate(batch_seqs):
            # tokens: <cls> seq <eos>
            # embeddings[j, 1:-1, :] are the residue embeddings
            residue_embeddings = embeddings[j, 1:len(seq)+1, :].cpu().numpy()
            all_embeddings.append(residue_embeddings)

    # Broadcast to atoms
    print("Broadcasting to atoms...")
    pocket_esmc = []
    pocket_mask = data['pocket_mask']
    pocket_residue_ids_all = data['pocket_residue_ids']

    for i in tqdm(range(n_samples)):
        # Get atoms for this sample
        sample_mask = (pocket_mask == i)
        sample_residue_ids = pocket_residue_ids_all[sample_mask]

        # Broadcast
        sample_embeddings = all_embeddings[i]
        sample_esmc = broadcast_to_atoms(sample_embeddings, sample_residue_ids)
        pocket_esmc.append(sample_esmc)

    pocket_esmc = np.concatenate(pocket_esmc, axis=0)

    # Save augmented NPZ
    print(f"Saving to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_path,
        **{k: v for k, v in data.items()},  # Copy all original fields
        pocket_esmc=pocket_esmc,  # Add ESM-C embeddings
    )

    print(f"✓ Done! Saved {n_samples} samples with ESM-C embeddings.")
    print(f"  pocket_esmc shape: {pocket_esmc.shape}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Input NPZ file')
    parser.add_argument('--output', type=str, required=True, help='Output NPZ file')
    parser.add_argument('--model', type=str, default='esm2_t33_650M_UR50D',
                       help='ESM model name')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--device', type=str, default='cuda', help='Device')

    args = parser.parse_args()

    compute_esmc_embeddings(
        npz_path=Path(args.input),
        model_name=args.model,
        batch_size=args.batch_size,
        device=args.device,
        output_path=Path(args.output),
    )


if __name__ == '__main__':
    main()
```

**Action Items**:
- [ ] Create `scripts/precompute_esmc.py`
- [ ] Implement `extract_sequences_from_pocket_ids()` function
- [ ] Test on small subset (10 samples)
- [ ] Profile runtime and memory usage

---

### Step 7: Run ESM-C Pre-computation

**Local Test** (small subset):
```bash
# Test on 100 samples
python scripts/precompute_esmc.py \
    --input data/processed_crossdock_noH_full_esmc/train_subset_100.npz \
    --output data/processed_crossdock_esmc/train_subset_100.npz \
    --model esm2_t33_650M_UR50D \
    --batch_size 16 \
    --device cuda
```

**HPC Submission** (full dataset):
```bash
# Create PBS script for each split
cat > .claude/hpc/precompute_esmc_train.pbs << 'EOF'
#!/bin/bash
#PBS -N esmc_train
#PBS -l select=1:ncpus=4:ngpus=1:mem=32gb
#PBS -l walltime=24:00:00
#PBS -A your_project_id

cd $PBS_O_WORKDIR
module load cuda/11.8
module load python/3.9

python scripts/precompute_esmc.py \
    --input data/processed_crossdock_noH_full_esmc/train.npz \
    --output data/processed_crossdock_esmc/train.npz \
    --model esm2_t33_650M_UR50D \
    --batch_size 32 \
    --device cuda
EOF

# Submit
qsub .claude/hpc/precompute_esmc_train.pbs
```

**Time Estimate**:
- 100K samples × ~100 residues/sample = 10M residues
- ESM-2 throughput: ~500 residues/sec on V100
- Total time: 10M / 500 = 20K seconds ≈ 5-6 hours

**Action Items**:
- [ ] Test on subset (100 samples)
- [ ] Profile runtime per sample
- [ ] Submit HPC jobs for train/val/test splits
- [ ] Monitor job progress
- [ ] Verify output files (see Step 8)

---

### Step 8: Validate ESM-C Embeddings

```bash
# Check ESM-C augmented NPZ
python -c "
import numpy as np

data = np.load('data/processed_crossdock_esmc/train.npz', allow_pickle=True)

print('Keys:', list(data.keys()))
print('Has ESM-C:', 'pocket_esmc' in data.keys())

if 'pocket_esmc' in data.keys():
    esmc = data['pocket_esmc']
    print(f'ESM-C shape: {esmc.shape}')
    print(f'Expected: ({len(data[\"pocket_coords\"])}, 960 or 1280)')

    # Verify broadcasting worked
    pocket_mask = data['pocket_mask']
    pocket_residue_ids = data['pocket_residue_ids']

    # Check first sample
    sample_mask = (pocket_mask == 0)
    sample_esmc = esmc[sample_mask]
    sample_res_ids = pocket_residue_ids[sample_mask]

    # Atoms from same residue should have identical embeddings
    for res_id in np.unique(sample_res_ids)[:5]:
        atom_indices = np.where(sample_res_ids == res_id)[0]
        embeddings = sample_esmc[atom_indices]
        all_equal = np.allclose(embeddings, embeddings[0])
        print(f'Residue {res_id}: {len(atom_indices)} atoms, embeddings equal: {all_equal}')
"
```

**Expected Output**:
```
Keys: [..., 'pocket_esmc']
Has ESM-C: True
ESM-C shape: (35301000, 960)
Expected: (35301000, 960 or 1280)
Residue 0: 4 atoms, embeddings equal: True
Residue 1: 5 atoms, embeddings equal: True
...
```

**Action Items**:
- [ ] Verify `pocket_esmc` field exists in all splits
- [ ] Check embedding dimensions (960 or 1280 depending on model)
- [ ] Verify broadcasting (atoms from same residue have identical embeddings)
- [ ] Spot-check embeddings are not all zeros or NaNs

---

## Phase 2: Model Implementation (Week 2)

### Step 9: Modify `equivariant_diffusion/dynamics.py`

**File**: `equivariant_diffusion/dynamics.py`

**Changes**:

```python
# Line 11: Add esmc_dim parameter
class EGNNDynamics(nn.Module):
    def __init__(self, atom_nf, residue_nf,
                 n_dims, joint_nf=16, hidden_nf=64, device='cpu',
                 act_fn=torch.nn.SiLU(), n_layers=4, attention=False,
                 condition_time=True, tanh=False, mode='egnn_dynamics',
                 norm_constant=0, inv_sublayers=2, sin_embedding=False,
                 normalization_factor=100, aggregation_method='sum',
                 update_pocket_coords=True, edge_cutoff_ligand=None,
                 edge_cutoff_pocket=None, edge_cutoff_interaction=None,
                 reflection_equivariant=True, edge_embedding_dim=None,
                 esmc_dim=0):  # NEW PARAMETER
        super().__init__()

        self.esmc_dim = esmc_dim  # NEW
        self.use_esmc = (esmc_dim > 0)  # NEW

        # ... (atom encoder unchanged)

        # Line 39-43: Modify residue encoder
        encoder_input_dim = residue_nf + esmc_dim  # NEW: augmented input
        self.residue_encoder = nn.Sequential(
            nn.Linear(encoder_input_dim, 2 * encoder_input_dim),  # MODIFIED
            act_fn,
            nn.Linear(2 * encoder_input_dim, joint_nf)  # MODIFIED
        )

        # Decoder unchanged (still outputs residue_nf)

# Line 87: Add esmc_embeddings parameter
    def forward(self, xh_atoms, xh_residues, t, mask_atoms, mask_residues,
                esmc_embeddings=None):  # NEW PARAMETER

        x_residues = xh_residues[:, :self.n_dims].clone()
        h_residues = xh_residues[:, self.n_dims:].clone()

        # ... (atom encoding unchanged)

        # NEW: Augment with ESM-C before encoding
        if self.use_esmc and esmc_embeddings is not None:
            h_residues = torch.cat([h_residues, esmc_embeddings], dim=-1)

        h_residues = self.residue_encoder(h_residues)

        # ... (rest unchanged)
```

**Action Items**:
- [ ] Add `esmc_dim` parameter to `__init__`
- [ ] Modify `encoder_input_dim` calculation
- [ ] Update encoder Linear layers
- [ ] Add `esmc_embeddings` parameter to `forward()`
- [ ] Add concatenation logic before encoding
- [ ] Test forward pass with dummy data

---

### Step 10: Modify `dataset.py`

**File**: `dataset.py`

**Changes**:

```python
# Line 8: Add use_esmc parameter
class ProcessedLigandPocketDataset(Dataset):
    def __init__(self, npz_path, center=True, transform=None, use_esmc=False):  # NEW

        self.transform = transform
        self.use_esmc = use_esmc  # NEW

        with np.load(npz_path, allow_pickle=True) as f:
            data = {key: val for key, val in f.items()}

        # ... (split data as before)

        # NEW: Check for ESM-C embeddings if requested
        if self.use_esmc and 'pocket_esmc' not in data.keys():
            raise ValueError(f"use_esmc=True but NPZ missing 'pocket_esmc' field!")

# Line 46: Return ESM-C embeddings in __getitem__
    def __getitem__(self, idx):
        data = {key: val[idx] for key, val in self.data.items()}

        # NEW: Handle ESM-C embeddings
        if self.use_esmc:
            data['pocket_esmc'] = self.data['pocket_esmc'][idx]

        if self.transform is not None:
            data = self.transform(data)
        return data

# Line 53: Handle in collate_fn
    @staticmethod
    def collate_fn(batch):
        out = {}
        for prop in batch[0].keys():

            if prop == 'names' or prop == 'receptors':
                out[prop] = [x[prop] for x in batch]
            elif prop == 'num_lig_atoms' or prop == 'num_pocket_nodes' \
                    or prop == 'num_virtual_atoms':
                out[prop] = torch.tensor([x[prop] for x in batch])
            elif 'mask' in prop:
                out[prop] = torch.cat([i * torch.ones(len(x[prop]))
                                       for i, x in enumerate(batch)], dim=0)
            else:
                out[prop] = torch.cat([x[prop] for x in batch], dim=0)

        return out
```

**Action Items**:
- [ ] Add `use_esmc` parameter to `__init__`
- [ ] Add ESM-C validation check
- [ ] Return `pocket_esmc` in `__getitem__`
- [ ] Test data loading with ESM-C

---

### Step 11: Modify `lightning_modules.py`

**File**: `lightning_modules.py`

**Changes**:

```python
# Line ~50: Add use_esmc to __init__
class LigandPocketDDPM(pl.LightningModule):
    def __init__(self, ..., use_esmc=False, esmc_dim=960):  # NEW
        super().__init__()

        self.use_esmc = use_esmc  # NEW

        # ... (dataset info)

        # Line ~80: Pass esmc_dim to dynamics
        self.dynamics = EGNNDynamics(
            atom_nf=len(self.dataset_info['atom_encoder']),
            residue_nf=len(self.dataset_info['aa_encoder']),  # Still 20
            esmc_dim=esmc_dim if use_esmc else 0,  # NEW
            # ... (other params)
        )

# Line ~130: Modify setup() to pass use_esmc to dataset
    def setup(self, stage=None):
        self.train_dataset = ProcessedLigandPocketDataset(
            os.path.join(self.data_path, 'train.npz'),
            center=True,
            transform=self.transform,
            use_esmc=self.use_esmc,  # NEW
        )
        # ... (val/test datasets similarly)

# Line ~274: Extract ESM-C in training_step
    def training_step(self, batch, batch_idx):
        ligand = {
            'x': batch['lig_coords'],
            'one_hot': batch['lig_one_hot'],
            'size': batch['num_lig_atoms'],
            'mask': batch['lig_mask'].long(),
        }
        pocket = {
            'x': batch['pocket_coords'],
            'one_hot': batch['pocket_one_hot'],
            'size': batch['num_pocket_nodes'],
            'mask': batch['pocket_mask'].long(),
        }

        # NEW: Extract ESM-C embeddings
        esmc = batch.get('pocket_esmc', None) if self.use_esmc else None

        # Pass to ddpm (need to modify ddpm.forward to accept esmc)
        delta_log_px, error_t_lig, error_t_pocket, SNR_weight, \
            loss_0_x_ligand, loss_0_x_pocket, loss_0_h, \
            neg_log_const_0, kl_prior = self.ddpm(ligand, pocket, esmc=esmc)  # NEW

        # ... (rest unchanged)
```

**Note**: Also need to modify `EnVariationalDiffusion` and `ConditionalDDPM` to accept and pass `esmc` to dynamics.

**Action Items**:
- [ ] Add `use_esmc` and `esmc_dim` parameters
- [ ] Pass `esmc_dim` to dynamics initialization
- [ ] Pass `use_esmc` to dataset
- [ ] Extract ESM-C from batch in training/validation steps
- [ ] Modify DDPM classes to accept and pass ESM-C

---

### Step 12: Create ESM-C Config File

**File**: `configs/crossdock_fullatom_cond_esmc.yml`

```yaml
run_name: "ESM-C-augmented"
logdir: "logs"
wandb_params:
  mode: "online"
  entity: "your-wandb-username"
  group: "crossdock-esmc"

dataset: "crossdock"
datadir: "data/processed_crossdock_esmc"  # NEW: ESM-C augmented data
use_esmc: True  # NEW FLAG
esmc_dim: 960  # NEW: ESM-2 is 1280, ESM-C is 960

mode: "pocket_conditioning"
pocket_representation: "full-atom"
virtual_nodes: False

batch_size: 8  # Reduced (larger encoder)
lr: 1.0e-3
n_epochs: 500
num_workers: 0
gpus: 4
clip_grad: True

egnn_params:
  joint_nf: 128
  hidden_nf: 256
  n_layers: 6
  attention: True
  edge_cutoff_pocket: 5.0
  edge_cutoff_interaction: 5.0

diffusion_params:
  diffusion_steps: 500
  diffusion_noise_schedule: "polynomial_2"
  diffusion_loss_type: "l2"
  normalize_factors: [1, 4]

eval_epochs: 25
visualize_sample_epoch: 50
eval_params:
  n_eval_samples: 100
  eval_batch_size: 50
```

**Action Items**:
- [ ] Create config file
- [ ] Update `datadir` to ESM-C augmented path
- [ ] Set `use_esmc: True` and `esmc_dim` appropriately
- [ ] Reduce batch size if needed (start with 8)

---

## Phase 3: Testing (Week 3)

### Step 13: Debug Run (Small Subset)

**Purpose**: Verify implementation works before full training

**Create debug config**: `configs/crossdock_fullatom_cond_esmc_debug.yml`
```yaml
# Copy from crossdock_fullatom_cond_esmc.yml
# Modify:
n_epochs: 10
batch_size: 4
eval_epochs: 5
```

**Create small subset**:
```bash
python -c "
import numpy as np

# Load full data
data = np.load('data/processed_crossdock_esmc/train.npz', allow_pickle=True)

# Take first 100 samples
subset = {}
for key in data.keys():
    if key in ['names', 'receptors']:
        subset[key] = data[key][:100]
    elif 'mask' in key:
        # Find cutoff for 100 samples
        cutoff = np.where(data[key] >= 100)[0][0]
        subset[key] = data[key][:cutoff]
    else:
        if key.startswith('lig'):
            cutoff = np.where(data['lig_mask'] >= 100)[0][0]
        else:
            cutoff = np.where(data['pocket_mask'] >= 100)[0][0]
        subset[key] = data[key][:cutoff]

np.savez('data/processed_crossdock_esmc/train_debug.npz', **subset)
print('Created debug subset: 100 samples')
"
```

**Run debug training**:
```bash
python train.py \
    --config configs/crossdock_fullatom_cond_esmc_debug.yml \
    --datadir data/processed_crossdock_esmc
```

**What to check**:
- [ ] No shape errors in forward pass
- [ ] No NaN losses
- [ ] GPU memory usage acceptable
- [ ] Loss decreases (even slightly)
- [ ] Validation runs without errors
- [ ] Sample generation works

**Common Issues & Fixes**:

| Issue | Fix |
|-------|-----|
| Shape mismatch in encoder | Check `encoder_input_dim = residue_nf + esmc_dim` |
| OOM error | Reduce batch_size to 2 or 4 |
| NaN losses | Adjust `normalize_factors` to [0.1, 1] |
| ESM-C not found in batch | Verify dataset `use_esmc=True` |

---

### Step 14: Small-scale Test (1K Samples)

**Purpose**: Verify training dynamics are reasonable

**Create 1K subset**:
```bash
# Similar to debug subset but with 1000 samples
python create_subset.py --n_samples 1000 --output train_1k.npz
```

**Config**: `configs/crossdock_fullatom_cond_esmc_1k.yml`
```yaml
# Same as main config but:
n_epochs: 50
datadir: "data/processed_crossdock_esmc_1k"
```

**Run**:
```bash
python train.py --config configs/crossdock_fullatom_cond_esmc_1k.yml
```

**Monitor**:
- [ ] Training loss curve (should decrease)
- [ ] Validation loss (should track training)
- [ ] Generated molecules (validity > 0%)
- [ ] No catastrophic failures

**Success Criteria**:
- Trains without crashes for 50 epochs
- Validation loss decreases or plateaus
- Can generate at least some valid molecules (>10% validity)

---

## Phase 4: Full Training (Week 3-5)

### Step 15: Submit HPC Training Job

**PBS Script**: `.claude/hpc/train_esmc.pbs`

```bash
#!/bin/bash
#PBS -N train_esmc
#PBS -l select=1:ncpus=16:ngpus=4:mem=128gb
#PBS -l walltime=336:00:00  # 14 days
#PBS -A your_project_id

cd $PBS_O_WORKDIR
module load cuda/11.8
module load python/3.9

# Activate environment
source venv/bin/activate

# Run training
python train.py \
    --config configs/crossdock_fullatom_cond_esmc.yml \
    --gpus 4 \
    --num_workers 4

echo "Training complete!"
```

**Submit**:
```bash
qsub .claude/hpc/train_esmc.pbs
```

**Monitor**:
```bash
# Check job status
qstat -u $USER

# Monitor WandB
# https://wandb.ai/your-entity/crossdock-esmc

# Check log files
tail -f train_esmc.o<job_id>
```

**Action Items**:
- [ ] Create PBS script
- [ ] Submit job
- [ ] Monitor via WandB
- [ ] Check for errors in first few epochs
- [ ] Let run for ~10-14 days

---

### Step 16: Baseline Comparison (Parallel)

**Important**: While ESM-C model trains, evaluate baseline for comparison

**Use existing checkpoint**:
```bash
# Baseline is already trained
checkpoint="checkpoints/crossdocked_fullatom_cond.ckpt"

# Generate on test set
python generate_ligands.py $checkpoint \
    --test_set data/processed_crossdock_noH_full/test.npz \
    --n_samples 100 \
    --outdir results/baseline/
```

**Compute metrics**:
```bash
python analyze_results.py \
    --checkpoint $checkpoint \
    --test_set data/processed_crossdock_noH_full/test.npz \
    --output results/baseline_metrics.json
```

**Action Items**:
- [ ] Load existing baseline checkpoint
- [ ] Generate 100 molecules per test pocket
- [ ] Compute validity, uniqueness, QED, SA scores
- [ ] Run Vina docking if available
- [ ] Save results for comparison

---

## Phase 5: Evaluation (Week 5-6)

### Step 17: Generate with ESM-C Model

**After training completes**:

```bash
# Load best checkpoint
checkpoint="logs/ESM-C-augmented/checkpoints/best.ckpt"

# Generate on test set
python generate_ligands.py $checkpoint \
    --test_set data/processed_crossdock_esmc/test.npz \
    --n_samples 100 \
    --outdir results/esmc/
```

---

### Step 18: Comparative Analysis

**Metrics to compute**:
- Validity (% RDKit-valid)
- Uniqueness (% unique SMILES)
- Novelty (% not in training)
- QED (drug-likeness)
- SA Score (synthetic accessibility)
- Vina docking scores
- Tanimoto similarity to reference

**Statistical tests**:
```python
from scipy.stats import wilcoxon

# Paired comparison (same pockets)
baseline_vina = [...]  # Baseline Vina scores
esmc_vina = [...]      # ESM-C Vina scores

statistic, p_value = wilcoxon(baseline_vina, esmc_vina)
print(f"p-value: {p_value}")
if p_value < 0.05:
    print("✓ Statistically significant improvement!")
```

**Action Items**:
- [ ] Compute all metrics for both models
- [ ] Run statistical tests
- [ ] Create comparison tables and plots
- [ ] Analyze failure cases

---

## Timeline Summary

| Week | Phase | Key Deliverables | Estimated Hours |
|------|-------|------------------|-----------------|
| 1 | Data Re-processing | Modified NPZ with residue IDs | 8-16 |
| 1-2 | ESM-C Pre-computation | Augmented NPZ with embeddings | 4-8 |
| 2 | Implementation | Modified codebase | 8-16 |
| 3 | Testing | Debug + 1K sample runs | 12 |
| 3-5 | Full Training | Trained model (HPC) | - |
| 5-6 | Evaluation | Metrics + comparison | 8-16 |

**Total hands-on time**: 40-68 hours
**Total elapsed time**: 5-6 weeks

---

## Success Criteria

### Minimum (Thesis-worthy)
- ✅ Model trains without crashes
- ✅ Data processing works correctly
- ✅ Any metric improves by >5%

### Good
- ✅ 2+ metrics improve (p < 0.05)
- ✅ Validity or QED increases
- ✅ Vina scores improve in tail distribution

### Excellent
- ✅ All metrics improve
- ✅ Mean Vina shift >0.5 kcal/mol
- ✅ Novel high-affinity molecules generated

---

## Getting Help

**Validation scripts**:
- `experiments/validate_esmc_integration.py` - Run all validation tests
- `experiments/example_residue_id_tracking.py` - Code modification examples

**Documentation**:
- `.claude/ESM_C_INTEGRATION_STRATEGY.md` - Full technical strategy
- `.claude/PLAN.md` - Thesis plan and timeline
- `experiments/VALIDATION_SUMMARY.md` - Detailed validation results

**Common issues**:
- See `experiments/VALIDATION_SUMMARY.md` section "Potential Issues & Mitigations"

---

## Checklist

### Phase 0A: Baseline Evaluation (PARALLEL - Week 1)
- [ ] Step 0.1: Verify baseline checkpoint
- [ ] Step 0.2: Generate baseline molecules (test set)
- [ ] Step 0.3: Compute baseline metrics
- [ ] Step 0.4: (Optional) Docking evaluation
- [ ] Step 0.5: Create baseline report
- [ ] Step 0.6: Document baseline numbers

### Phase 0B: Data Re-processing (Week 1)
- [ ] Step 1: Modify process_crossdock.py
- [ ] Step 2: Test on single PDB
- [ ] Step 3: Re-process full dataset
- [ ] Step 4: Validate re-processed data

### Phase 1: ESM-C Pre-computation (Week 1-2)
- [ ] Step 5: Install ESM-C
- [ ] Step 6: Write precompute_esmc.py
- [ ] Step 7: Run ESM-C pre-computation
- [ ] Step 8: Validate ESM-C embeddings

### Phase 2: Implementation (Week 2)
- [ ] Step 9: Modify dynamics.py
- [ ] Step 10: Modify dataset.py
- [ ] Step 11: Modify lightning_modules.py
- [ ] Step 12: Create ESM-C config file

### Phase 3: Testing (Week 3)
- [ ] Step 13: Debug run (100 samples)
- [ ] Step 14: Small-scale test (1K samples)

### Phase 4: Full Training (Week 3-5)
- [ ] Step 15: Submit HPC training job
- [ ] Step 16: Monitor training progress

### Phase 5: Evaluation (Week 5-6)
- [ ] Step 17: Generate with ESM-C model
- [ ] Step 18: Comparative analysis (vs baseline)

---

**Current Status**: Ready to start Phase 0A and 0B (in parallel)
**Next Actions**:
1. **Step 0.1**: Verify baseline checkpoint (can start immediately)
2. **Step 1**: Modify `process_crossdock.py` (can start immediately)
