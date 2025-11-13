# Baseline Evaluation Guide

**Purpose**: Establish baseline performance before making any changes to the model

**Time Estimate**: 4-8 hours
**Can run in parallel with**: Data re-processing

---

## Why Baseline Evaluation is Critical

Before integrating ESM-C, you MUST know:
1. **What is the current performance?** (validity, QED, SA, Vina scores)
2. **What is the target to beat?** (minimum improvement: >5%)
3. **Is the improvement statistically significant?** (p < 0.05)

Without baseline metrics, you cannot prove ESM-C works!

---

## Quick Start

### Step 1: Check for Baseline Checkpoint

```bash
# Look for pre-trained checkpoint
ls -lh checkpoints/crossdocked_fullatom_cond.ckpt

# If not found, check if you have it elsewhere
find . -name "*crossdocked*fullatom*.ckpt" 2>/dev/null

# Download from Zenodo if needed
# https://zenodo.org/record/8183747
```

**Expected**: File size ~500 MB

---

### Step 2: Verify Checkpoint Loads

```python
# Test script
from lightning_modules import LigandPocketDDPM

checkpoint_path = "checkpoints/crossdocked_fullatom_cond.ckpt"
model = LigandPocketDDPM.load_from_checkpoint(checkpoint_path)

print(f"✓ Model loaded: {model.__class__.__name__}")
print(f"  Mode: {model.hparams.mode}")
print(f"  Dataset: {model.hparams.dataset}")

# Expected output:
# ✓ Model loaded: LigandPocketDDPM
#   Mode: pocket_conditioning
#   Dataset: crossdock
```

---

### Step 3: Generate Baseline Molecules

**Create** `scripts/generate_baseline_test_set.py`:

See `.claude/NEXT_STEPS.md` Step 0.2 for full code.

**Run**:
```bash
python scripts/generate_baseline_test_set.py \
    --checkpoint checkpoints/crossdocked_fullatom_cond.ckpt \
    --test_set data/processed_crossdock_noH_full/test.npz \
    --n_samples 100 \
    --outdir results/baseline/molecules/
```

**Expected time**: 2-6 hours (depends on test set size and GPU)

**Expected output**:
```
Found 200 test pockets
Generating 100 molecules per pocket
Generating: 100%|██████████| 200/200
✓ Generation complete!
  Saved 200 pockets to results/baseline/molecules
```

---

### Step 4: Compute Metrics

**Create** `scripts/evaluate_baseline.py`:

See `.claude/NEXT_STEPS.md` Step 0.3 for full code.

**Run**:
```bash
python scripts/evaluate_baseline.py \
    --molecules_dir results/baseline/molecules/ \
    --test_set data/processed_crossdock_noH_full/test.npz \
    --output results/baseline/baseline_metrics.json
```

**Expected output**:
```
Total molecules loaded: 20000
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

**SAVE THESE NUMBERS!** You'll compare ESM-C against them.

---

### Step 5: (Optional) Docking Evaluation

**Only if you have**:
- smina or Vina installed
- Receptor PDB files
- Extra time (can be slow)

See `.claude/NEXT_STEPS.md` Step 0.4 for details.

---

### Step 6: Create Report

**Create** `scripts/create_baseline_report.py`:

See `.claude/NEXT_STEPS.md` Step 0.5 for full code.

**Run**:
```bash
python scripts/create_baseline_report.py
```

**Creates**: `results/baseline/BASELINE_REPORT.md` with:
- Summary statistics
- Molecular properties
- Distribution plots (if matplotlib available)
- Comparison targets for ESM-C

---

### Step 7: Document Baseline Numbers

**Create** `results/baseline/BASELINE_SUMMARY.txt`:

```
BASELINE MODEL PERFORMANCE (DiffSBDD)
=====================================

Checkpoint: checkpoints/crossdocked_fullatom_cond.ckpt
Test Set: CrossDocked test split
Date: 2025-11-13

KEY METRICS:
-----------
Validity:    72.3%
Uniqueness:  94.7%
QED:         0.452 ± 0.142
SA Score:    3.18 ± 0.87
LogP:        2.34 ± 1.12
Mol Weight:  287.3 ± 78.4 Da
Num Atoms:   21.4 ± 5.8

TARGETS FOR ESM-C MODEL:
------------------------
Validity:    >77% (>5% improvement)
QED:         >0.50 (>0.05 improvement)
SA Score:    <3.0 (<-0.2 improvement, lower is better)

Statistical significance required: p < 0.05 (Wilcoxon test)
```

**CRITICAL**: Commit this to git!

```bash
git add results/baseline/
git commit -m "Add baseline evaluation results"
```

---

## What You'll Have After This

**Files created**:
```
scripts/
  ├── generate_baseline_test_set.py
  ├── evaluate_baseline.py
  ├── evaluate_docking_baseline.py (optional)
  └── create_baseline_report.py

results/baseline/
  ├── molecules/
  │   ├── pocket_1_baseline.sdf
  │   ├── pocket_2_baseline.sdf
  │   └── ...
  ├── baseline_metrics.json
  ├── BASELINE_REPORT.md
  └── BASELINE_SUMMARY.txt
```

**Metrics documented**:
- ✅ Validity (% chemically valid)
- ✅ Uniqueness (% unique SMILES)
- ✅ QED (drug-likeness)
- ✅ SA Score (synthetic accessibility)
- ✅ LogP (lipophilicity)
- ✅ Molecular weight
- ✅ Number of atoms
- ✅ Vina scores (optional)

---

## Common Issues

### Issue 1: Checkpoint not found

**Solution**:
- Check if you have the checkpoint from the paper authors
- Download from Zenodo: https://zenodo.org/record/8183747
- Or train baseline model yourself (takes 2-3 weeks)

### Issue 2: Generation is slow

**Solution**:
- Reduce `n_samples` to 10 or 20 for quick test
- Use GPU for generation
- Run on HPC if available

### Issue 3: Low validity (<50%)

**Possible causes**:
- Wrong checkpoint (check it's conditional mode)
- Data mismatch (check test set path)
- Bug in molecule builder

**Solution**: Compare with paper's reported validity (~72%)

### Issue 4: Metrics computation fails

**Solution**:
- Check RDKit is installed: `pip install rdkit`
- Check SA Score available: `from rdkit.Contrib.SA_Score import sascorer`
- Verify molecules are valid RDKit objects

---

## Success Criteria

After completing baseline evaluation:

✅ **You have**:
1. Generated ~10K-20K baseline molecules
2. Computed all key metrics
3. Documented baseline numbers
4. Created comparison targets

✅ **You know**:
1. What validity to expect (~72%)
2. What QED to expect (~0.45)
3. What SA Score to expect (~3.2)
4. What improvement is needed (>5%)

✅ **You can**:
1. Run statistical tests (Wilcoxon)
2. Compare ESM-C to baseline
3. Prove improvement is significant
4. Write results section of thesis

---

## Next Steps

**After baseline evaluation**:

1. **Continue with data re-processing** (Phase 0B)
   - Modify `process_crossdock.py`
   - Re-process dataset with residue IDs

2. **Pre-compute ESM-C embeddings** (Phase 1)
   - Install fair-esm
   - Compute embeddings for all pockets

3. **Implement ESM-C integration** (Phase 2)
   - Modify dynamics.py
   - Train model with ESM-C

4. **Compare to baseline** (Phase 5)
   - Generate with ESM-C model
   - Run same evaluation scripts
   - Statistical comparison

---

## Quick Reference

**Generate baseline**:
```bash
python scripts/generate_baseline_test_set.py \
    --checkpoint checkpoints/crossdocked_fullatom_cond.ckpt \
    --test_set data/processed_crossdock_noH_full/test.npz \
    --n_samples 100 \
    --outdir results/baseline/molecules/
```

**Evaluate baseline**:
```bash
python scripts/evaluate_baseline.py \
    --molecules_dir results/baseline/molecules/ \
    --output results/baseline/baseline_metrics.json
```

**Create report**:
```bash
python scripts/create_baseline_report.py
```

**Key files**:
- Full guide: `.claude/NEXT_STEPS.md` (Step 0.1-0.6)
- Code examples: See NEXT_STEPS.md
- Results: `results/baseline/`

---

**Questions?** See `.claude/NEXT_STEPS.md` Phase 0A for complete details.
