# Day 5 Evaluation Guide: Adding Models to Comparison Table

## 🎯 Goal
Compare different models using Wasserstein distances to ground truth molecules.

**Lower Wasserstein distance = Better (closer to real drug-like molecules)**

---

## 📊 Current Baselines

| Model | Description | Status |
|-------|-------------|--------|
| `no_film` | Pretrained without FiLM | ✅ Done |
| `identity` | Identity FiLM (γ=1, β=0) | ✅ Done |
| `random` | Random FiLM (negative control) | ✅ Done |
| `film_trained` | **Your trained FiLM model** | 🔜 Next step! |

---

## 🚀 Quick Workflow: Add Trained FiLM to Comparison

### Step 0: Train Your Model
```bash
# Train FiLM using your config
uv run python train.py --config thesis_work/experiments/day5_film_finetuning/configs/film_finetuning.yml

# This creates checkpoint: lightning_logs/version_X/checkpoints/epoch=Y-step=Z.ckpt
```

### Step 1: Generate Molecules with Trained Model
```bash
uv run python test.py lightning_logs/version_X/checkpoints/epoch=Y-step=Z.ckpt \
    --test_dir data/dummy_testing_dataset_10_tests/test \
    --outdir thesis_work/experiments/day5_film_finetuning/outputs/baseline_comparison_v2/film_trained \
    --n_samples 5 \
    --batch_size 1 \
    --sanitize \
    --use_film
```

### Step 2: Analyze with Ground Truth (Calculates Wasserstein Distances)
```bash
uv run python analyze_results.py \
    thesis_work/experiments/day5_film_finetuning/outputs/baseline_comparison_v2/film_trained \
    --ground_truth data/dummy_testing_dataset_10_tests/analysis/test_ground_truth_properties.csv
```

### Step 3: Update Comparison Table
```bash
uv run python create_comparison_table.py \
    --baseline_dir thesis_work/experiments/day5_film_finetuning/outputs/baseline_comparison_v2
```

### Step 4: Check Results
```bash
cat thesis_work/experiments/day5_film_finetuning/outputs/baseline_comparison_v2/comparison_table.csv
```

---

## 📈 Expected Results

### Before Training (Current Baselines)
All three are similar because they use the same frozen pretrained EGNN:

| Model | QED (W↓) | SA (W↓) | LogP (W↓) |
|-------|----------|---------|-----------|
| no_film | 0.0476 | 0.5712 | 0.8604 |
| identity | 0.0663 | 0.7347 | 0.8815 |
| random | 0.0766 | 0.5010 | 0.9566 |

### After Training (Expected)
If FiLM training works, you should see **improvement**:

| Model | QED (W↓) | SA (W↓) | LogP (W↓) | Result |
|-------|----------|---------|-----------|--------|
| no_film | 0.0476 | 0.5712 | 0.8604 | Baseline |
| **film_trained** | **0.02-0.03** ✨ | **0.30-0.45** ✨ | **0.60-0.80** ✨ | **Better!** |
| identity | 0.0663 | 0.7347 | 0.8815 | Same as baseline |
| random | 0.0766 | 0.5010 | 0.9566 | Worse (garbage) |

**Success criteria**: `film_trained` has lower Wasserstein distances than `no_film`

---

## 📁 File Structure

```
thesis_work/experiments/day5_film_finetuning/
├── configs/
│   └── film_finetuning.yml              # Your training config
├── outputs/
│   └── baseline_comparison_v2/
│       ├── no_film/                      # Baseline 1
│       │   ├── processed/*.sdf           # 50 molecules (10 pockets × 5 samples)
│       │   └── analysis_summary.csv      # Metrics + Wasserstein distances
│       ├── identity/                     # Baseline 2
│       │   ├── processed/*.sdf
│       │   └── analysis_summary.csv
│       ├── random/                       # Baseline 3
│       │   ├── processed/*.sdf
│       │   └── analysis_summary.csv
│       ├── film_trained/                 # Your trained model (ADD THIS!)
│       │   ├── processed/*.sdf
│       │   └── analysis_summary.csv
│       └── comparison_table.csv          # Final comparison table
└── EVALUATION_GUIDE.md                   # This file
```

---

## 🔧 Core Commands Reference

### Regenerate Comparison (Anytime)
If you want to update the comparison table without regenerating molecules:

```bash
# Just run analysis + comparison
uv run python analyze_results.py \
    thesis_work/experiments/day5_film_finetuning/outputs/baseline_comparison_v2/<model_name> \
    --ground_truth data/dummy_testing_dataset_10_tests/analysis/test_ground_truth_properties.csv

uv run python create_comparison_table.py \
    --baseline_dir thesis_work/experiments/day5_film_finetuning/outputs/baseline_comparison_v2
```

### Check Ground Truth Stats
```bash
cat data/dummy_testing_dataset_10_tests/analysis/test_ground_truth_properties.csv
```

Ground truth (10 real ligands):
- QED: 0.452 ± 0.244
- SA Score: 3.86 ± 1.36
- LogP: 0.72 ± 2.42
- MolWt: 387.8 ± 120.1 Da
- NumAtoms: 26.5 ± 7.8

---

## 📊 Understanding Wasserstein Distance

**What it measures**: How different two probability distributions are

**Example**:
- Ground truth QED distribution: [0.3, 0.5, 0.6, 0.4, 0.7, ...]
- Generated QED distribution: [0.4, 0.5, 0.5, 0.6, 0.6, ...]
- Wasserstein distance: 0.0476 (very similar!)

**Interpretation**:
- **0.00-0.05**: Excellent match to ground truth
- **0.05-0.10**: Good match
- **0.10-0.20**: Moderate match
- **>0.20**: Poor match

---

## 🎓 Research Question Validation

Your thesis hypothesis:
> **"Can ESM-C evolutionary context improve structure-based drug design?"**

**How to validate with this evaluation**:

1. ✅ Baseline established: `no_film` (no evolutionary context) → W = 0.0476 (QED)
2. 🔜 Trained model: `film_trained` (with ESM-C) → W = ? (should be lower!)
3. 📊 Compare: If `film_trained` < `no_film` → **ESM-C helps!** ✨

**Statistical significance**:
- Run on larger test set (100+ pockets)
- Calculate confidence intervals
- Perform t-test or Wilcoxon test

---

## 🐛 Troubleshooting

### Problem: "Ground truth file not found"
**Solution**:
```bash
uv run python prepare_dataset_analysis.py data/dummy_testing_dataset_10_tests --split test
```

### Problem: "Comparison table doesn't include my model"
**Solution**: Make sure `analysis_summary.csv` exists in your model's directory. Run step 2 first.

### Problem: "All models have similar Wasserstein distances"
**Answer**: This is expected for **untrained** models! They all use the same pretrained EGNN. After training FiLM, you should see improvement.

### Problem: "Trained model is WORSE than baseline"
**Possible causes**:
- Training didn't converge yet (check loss curves)
- Learning rate too high (molecules got worse)
- ESM-C embeddings not properly loaded
- FiLM network not properly integrated

---

## 📝 Adding Different Datasets

To evaluate on a different dataset (e.g., official test set):

```bash
# 1. Prepare ground truth for new dataset
uv run python prepare_dataset_analysis.py data/official_test --split test

# 2. Generate molecules on new test set
uv run python test.py <checkpoint> \
    --test_dir data/official_test/test \
    --outdir thesis_work/experiments/day5_film_finetuning/outputs/official_test_comparison/film_trained \
    --n_samples 5

# 3. Analyze with new ground truth
uv run python analyze_results.py \
    thesis_work/experiments/day5_film_finetuning/outputs/official_test_comparison/film_trained \
    --ground_truth data/official_test/analysis/test_ground_truth_properties.csv

# 4. Create comparison table for this dataset
uv run python create_comparison_table.py \
    --baseline_dir thesis_work/experiments/day5_film_finetuning/outputs/official_test_comparison
```

---

## ✅ Summary

**To add your trained FiLM model:**

```bash
# 1. Generate molecules
uv run python test.py <trained_ckpt> --test_dir data/dummy_testing_dataset_10_tests/test --outdir outputs/baseline_comparison_v2/film_trained --n_samples 5 --use_film

# 2. Analyze
uv run python analyze_results.py outputs/baseline_comparison_v2/film_trained --ground_truth data/dummy_testing_dataset_10_tests/analysis/test_ground_truth_properties.csv

# 3. Update table
uv run python create_comparison_table.py --baseline_dir outputs/baseline_comparison_v2
```

**Check results**: Lower Wasserstein distance = FiLM training worked! 🎉

---

## 📚 Related Files

- **`EVALUATION_WORKFLOW.md`** (project root) - General evaluation workflow
- **`configs/film_finetuning.yml`** - Training configuration
- **`scripts/run_baselines.sh`** - Automated baseline comparison script
