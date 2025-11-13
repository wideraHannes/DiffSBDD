# Baseline Evaluation

**Purpose**: Small-scale baseline evaluation on 10 test pockets to establish performance baseline

**Status**: Ready to run

---

## Directory Structure

```
baseline/
├── README.md                 # This file
├── scripts/                  # Evaluation scripts
│   ├── 1_extract_subset.py
│   ├── 2_generate_molecules.py
│   ├── 3_compute_metrics.py
│   ├── 4_create_report.py
│   └── run_all.sh           # Run entire pipeline
├── data/                     # Test subset data
│   └── test_subset_10.npz   # 10 pocket subset
└── results/                  # Generated results
    ├── molecules/            # Generated SDF files
    ├── metrics.json          # Computed metrics
    └── REPORT.md             # Final report
```

---

## Quick Start

### Option 1: Run Complete Pipeline (Recommended)

```bash
cd baseline/scripts
bash run_all.sh
```

**Time**: ~30-60 minutes

### Option 2: Run Step-by-Step

```bash
# Step 1: Extract 10-pocket subset
cd baseline/scripts
python 1_extract_subset.py

# Step 2: Generate molecules (takes ~10-20 min)
python 2_generate_molecules.py

# Step 3: Compute metrics
python 3_compute_metrics.py

# Step 4: Create report
python 4_create_report.py
```

---

## What This Does

1. **Extracts** 10 random pockets from test set
2. **Generates** 20 molecules per pocket (200 total)
3. **Computes** all metrics:
   - Validity
   - Uniqueness
   - QED
   - SA Score
   - LogP
   - Molecular weight
   - Number of atoms
4. **Creates** detailed report with statistics

---

## Expected Results

After running, you'll have:

```
baseline/results/
├── molecules/
│   ├── pocket_0.sdf         # 20 molecules
│   ├── pocket_1.sdf
│   └── ... (10 files)
├── metrics.json              # All computed metrics
└── REPORT.md                 # Summary report
```

**Example metrics** (expected):
- Validity: ~70-75%
- Uniqueness: ~90-95%
- QED: ~0.45 ± 0.15
- SA Score: ~3.2 ± 0.9

---

## After Small-Scale Test

If results look good:

1. **Scale up** to full test set (100+ pockets)
2. **Run full baseline** evaluation
3. **Compare** ESM-C model to these baseline numbers

See `.claude/NEXT_STEPS.md` Step 0.2-0.6 for full evaluation.

---

## Troubleshooting

### Checkpoint not found
```bash
# Check for checkpoint
ls -lh ../checkpoints/crossdocked_fullatom_cond.ckpt

# If missing, download from Zenodo or check paper
```

### Generation fails
- Check GPU is available: `nvidia-smi`
- Reduce `n_samples` to 10 in script
- Check test set path is correct

### Low validity
- Expected: ~70-75%
- If <50%, check checkpoint is correct mode

---

## Files Reference

- **Scripts**: `baseline/scripts/*.py`
- **Data**: `baseline/data/test_subset_10.npz`
- **Results**: `baseline/results/`
- **Full guide**: `.claude/BASELINE_EVALUATION_GUIDE.md`
