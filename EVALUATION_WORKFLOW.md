# Simple Evaluation Workflow with Wasserstein Distances

## 📋 Overview

This workflow evaluates molecule generation models by comparing their output to ground truth using Wasserstein distances.

**Lower Wasserstein distance = Better (closer to real drug-like molecules)**

---

## 🚀 Quick Start: Add New Model to Comparison

### Step 1: Generate Molecules (if not done yet)
```bash
# Run your model and generate molecules
uv run python test.py <checkpoint> \
    --test_dir data/dummy_testing_dataset_10_tests/test \
    --outdir thesis_work/experiments/day5_film_finetuning/outputs/baseline_comparison_v2/<model_name> \
    --n_samples 5 \
    --batch_size 1 \
    --sanitize
```

### Step 2: Analyze with Ground Truth
```bash
# This calculates Wasserstein distances and saves to analysis_summary.csv
uv run python analyze_results.py \
    thesis_work/experiments/day5_film_finetuning/outputs/baseline_comparison_v2/<model_name> \
    --ground_truth data/dummy_testing_dataset_10_tests/analysis/test_ground_truth_properties.csv
```

### Step 3: Update Comparison Table
```bash
# Regenerates comparison_table.csv with ALL models in the directory
uv run python create_comparison_table.py \
    --baseline_dir thesis_work/experiments/day5_film_finetuning/outputs/baseline_comparison_v2
```

**That's it!** Your new model is now in `comparison_table.csv`

---

## 📁 File Structure

```
thesis_work/experiments/day5_film_finetuning/outputs/baseline_comparison_v2/
├── no_film/                    # Baseline 1: Pretrained without FiLM
│   ├── processed/*.sdf         # Generated molecules
│   └── analysis_summary.csv    # Metrics + Wasserstein distances
├── identity/                   # Baseline 2: Identity FiLM (untrained)
│   ├── processed/*.sdf
│   └── analysis_summary.csv
├── random/                     # Baseline 3: Random FiLM (negative control)
│   ├── processed/*.sdf
│   └── analysis_summary.csv
├── film_trained/               # Your trained FiLM model (add this next!)
│   ├── processed/*.sdf
│   └── analysis_summary.csv
└── comparison_table.csv        # Final comparison of ALL models
```

---

## 🔧 Core Scripts (3 files only!)

### 1. `prepare_dataset_analysis.py`
**Purpose**: Calculate ground truth properties (run once per dataset)

```bash
# Create ground truth reference for a dataset
uv run python prepare_dataset_analysis.py data/dummy_testing_dataset_10_tests --split test
```

**Output**: `data/dummy_testing_dataset_10_tests/analysis/test_ground_truth_properties.csv`

### 2. `analyze_results.py`
**Purpose**: Analyze generated molecules and compute Wasserstein distances

```bash
# Analyze a single model's results
uv run python analyze_results.py <results_dir> --ground_truth <ground_truth_csv>
```

**Output**: `<results_dir>/analysis_summary.csv` (with Wasserstein distances)

### 3. `create_comparison_table.py`
**Purpose**: Aggregate all models into one comparison table

```bash
# Create comparison table from all models in directory
uv run python create_comparison_table.py --baseline_dir <directory>
```

**Output**: `<directory>/comparison_table.csv`

---

## 📊 Example: Adding Your Trained FiLM Model

### Scenario: You trained FiLM and want to compare it to baselines

```bash
# 1. Generate molecules with your trained model
uv run python test.py checkpoints/film_trained_epoch10.ckpt \
    --test_dir data/dummy_testing_dataset_10_tests/test \
    --outdir thesis_work/experiments/day5_film_finetuning/outputs/baseline_comparison_v2/film_trained \
    --n_samples 5 \
    --use_film

# 2. Analyze with ground truth
uv run python analyze_results.py \
    thesis_work/experiments/day5_film_finetuning/outputs/baseline_comparison_v2/film_trained \
    --ground_truth data/dummy_testing_dataset_10_tests/analysis/test_ground_truth_properties.csv

# 3. Update comparison table
uv run python create_comparison_table.py \
    --baseline_dir thesis_work/experiments/day5_film_finetuning/outputs/baseline_comparison_v2

# Done! Now comparison_table.csv includes: no_film, identity, random, film_trained
```

---

## 🎯 Expected Results

### Before Training (Current State)
All baselines perform similarly (~0.05-0.08 Wasserstein) because they use the same frozen pretrained EGNN.

| Model | QED (W↓) | SA (W↓) | Why Similar? |
|-------|----------|---------|--------------|
| no_film | 0.048 | 0.57 | Pretrained EGNN only |
| identity | 0.067 | 0.73 | Untrained FiLM (no-op) |
| random | 0.077 | 0.50 | Random FiLM (garbage) |

### After Training FiLM (Expected)
Trained FiLM should show **lower** Wasserstein distances (improvement over pretrained).

| Model | QED (W↓) | SA (W↓) | Expected Outcome |
|-------|----------|---------|------------------|
| no_film | 0.048 | 0.57 | Baseline |
| **film_trained** | **0.02-0.03** ✨ | **0.30-0.40** ✨ | **Better** (ESM-C helps!) |
| identity | 0.067 | 0.73 | Same as no_film |
| random | 0.077 | 0.50 | Worse (garbage) |

---

## 📝 Notes

### Ground Truth Properties Calculated:
- **QED**: Quantitative Estimate of Drug-likeness
- **SA Score**: Synthetic Accessibility (1-10, lower = easier to synthesize)
- **LogP**: Lipophilicity (octanol-water partition coefficient)
- **Molecular Weight**: Da
- **Number of Atoms**: Size distribution

### Wasserstein Distance Interpretation:
- Measures "distance" between two probability distributions
- **Lower** = generated molecules have similar properties to real ligands
- **Higher** = generated molecules are chemically different from real ligands

### Data Flow:
```
Ground Truth SDF files
    ↓ [prepare_dataset_analysis.py]
Ground Truth CSV (properties distributions)
    ↓ [Used by analyze_results.py]
Generated Molecules → analysis_summary.csv (with Wasserstein distances)
    ↓ [create_comparison_table.py aggregates all]
comparison_table.csv (final comparison)
```

---

## ⚡ Pro Tips

1. **Rerun evaluation anytime**: Just run steps 2-3, no need to regenerate molecules
2. **Add multiple models**: Put them all in the same baseline_dir, run step 3 once
3. **Different datasets**: Run `prepare_dataset_analysis.py` once per dataset
4. **Quick check**: Look at `comparison_table.csv` sorted by QED Wasserstein (lower = better)

---

## 🐛 Troubleshooting

**Q: Comparison table missing my model?**
A: Run `analyze_results.py` first to create `analysis_summary.csv`

**Q: Ground truth file not found?**
A: Run `prepare_dataset_analysis.py` on your dataset first

**Q: All models have similar Wasserstein distances?**
A: Normal for untrained models! Train FiLM to see differences.

---

## 📚 Summary

**3 commands to add any model to comparison:**
```bash
# 1. Analyze
uv run python analyze_results.py <model_dir> --ground_truth <ground_truth.csv>

# 2. Update table
uv run python create_comparison_table.py --baseline_dir <all_models_dir>

# 3. Check results
cat <all_models_dir>/comparison_table.csv
```

**Keep It Simple!** ✨
