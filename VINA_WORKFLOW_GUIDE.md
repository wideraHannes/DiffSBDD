# Complete Vina Analysis Workflow

## Overview

This guide shows how to integrate Vina docking scores into your baseline comparison pipeline.

## Workflow

### Step 1: Create Ground Truth with Vina (One-time)

```bash
# Generate ground truth CSV with Vina scores
uv run python create_docking_ground_truth.py
```

**Output:** `data/dummy_testing_dataset_10_tests/ground_truth_with_vina_exhaustiveness16.csv`

**Includes:**
- All molecular properties (QED, SA, LogP, etc.)
- Vina binding affinity scores
- Ready for Wasserstein distance calculations

---

### Step 2: Analyze Generated Molecules with Vina

For each baseline experiment:

```bash
# Basic usage (fast, score_only mode)
uv run python analyze_results_with_vina.py \
    thesis_work/experiments/day5_film_finetuning/outputs/baseline_comparison_v2/film_trained \
    --ground_truth data/dummy_testing_dataset_10_tests/ground_truth_with_vina_exhaustiveness16.csv

# Full docking mode (more accurate, slower)
uv run python analyze_results_with_vina.py \
    thesis_work/experiments/day5_film_finetuning/outputs/baseline_comparison_v2/film_trained \
    --ground_truth data/dummy_testing_dataset_10_tests/ground_truth_with_vina_exhaustiveness16.csv \
    --vina_full_docking \
    --vina_exhaustiveness 16

# Disable Vina (use original script behavior)
uv run python analyze_results_with_vina.py \
    thesis_work/experiments/day5_film_finetuning/outputs/baseline_comparison_v2/film_trained \
    --ground_truth data/dummy_testing_dataset_10_tests/ground_truth_with_vina_exhaustiveness16.csv \
    --disable_vina
```

**Parameters:**
- `--vina_exhaustiveness N`: Thoroughness (8=fast, 16=recommended, 32=best)
- `--vina_full_docking`: Enable full docking (default: score_only for speed)
- `--disable_vina`: Skip Vina calculations entirely
- `--test_data_dir PATH`: Custom path to test data receptors

**Output:**
- `{results_dir}/analysis_summary.csv` - Summary with Vina metrics
- `{results_dir}/analysis_summary.detailed.csv` - Per-molecule details
- Console output with statistics

---

### Step 3: Create Comparison Table

```bash
uv run python create_comparison_table.py \
    --baseline_dir thesis_work/experiments/day5_film_finetuning/outputs/baseline_comparison_v2
```

**Output:** `baseline_comparison_v2/comparison_table.csv`

**Includes:**
- Wasserstein distances for all metrics (including Vina!)
- Validity metrics
- Average scores
- Sorted by best performance

---

## Complete Example

```bash
# 1. Ground truth with Vina (one-time)
uv run python create_docking_ground_truth.py

# 2. Analyze all baselines
for dir in thesis_work/experiments/day5_film_finetuning/outputs/baseline_comparison_v2/*/; do
    echo "Analyzing $(basename $dir)..."
    uv run python analyze_results_with_vina.py \
        "$dir" \
        --ground_truth data/dummy_testing_dataset_10_tests/ground_truth_with_vina_exhaustiveness16.csv \
        --vina_exhaustiveness 16
done

# 3. Create comparison table
uv run python create_comparison_table.py \
    --baseline_dir thesis_work/experiments/day5_film_finetuning/outputs/baseline_comparison_v2
```

---

## Output Format

### Comparison Table Columns

| Column | Description | Good Value |
|--------|-------------|------------|
| `Model` | Experiment name | - |
| `QED (W↓)` | QED Wasserstein distance | Lower |
| `SA (W↓)` | SA Wasserstein distance | Lower |
| `LogP (W↓)` | LogP Wasserstein distance | Lower |
| `MolWt (W↓)` | Molecular weight Wasserstein | Lower |
| `NumAtoms (W↓)` | Atom count Wasserstein | Lower |
| **`Vina (W↓)`** | **Vina score Wasserstein** ⭐ | **Lower** |
| `Validity` | Fraction valid molecules | Higher |
| `Connectivity` | Fraction connected | Higher |
| `Uniqueness` | Fraction unique | Higher |
| `Diversity` | Structural diversity | Higher |
| `QED (avg)` | Average QED score | 0.5-0.8 |
| `SA (avg)` | Average SA score | 0.6-0.9 |
| `LogP (avg)` | Average LogP | 0-3 |
| **`Vina (avg)`** | **Average binding affinity** ⭐ | **< -7.0** |

---

## Performance Comparison

| Mode | Speed | Use Case | Command Flag |
|------|-------|----------|-------------|
| Score-only (default) | ~1-2 sec/mol | Quick screening, many experiments | (default) |
| Full docking, exhaustiveness=8 | ~5-10 sec/mol | Standard analysis | `--vina_full_docking --vina_exhaustiveness 8` |
| Full docking, exhaustiveness=16 | ~10-20 sec/mol | Recommended for comparison | `--vina_full_docking --vina_exhaustiveness 16` |
| Full docking, exhaustiveness=32 | ~30-60 sec/mol | Publication quality | `--vina_full_docking --vina_exhaustiveness 32` |

**Recommendation for thesis:**
- Initial screening: score_only mode
- Final comparison: `--vina_full_docking --vina_exhaustiveness 16`

---

## Example Output

### Console Output

```
DIFFSBDD MOLECULE GENERATION ANALYSIS (with Vina)
================================================================

📊 DATASET OVERVIEW:
  Total molecules generated: 100
  Total pockets processed: 10
  Average molecules per pocket: 10.0

🔬 VINA DOCKING SCORES:
  Mean: -7.234 ± 1.102 kcal/mol
  Range: [-9.843, -5.374] kcal/mol
  Valid scores: 98.0%
  Binding strength distribution:
    Good (-9.0 to -7.0): 52 (53.1%)
    Moderate (-7.0 to -5.0): 46 (46.9%)

📏 WASSERSTEIN DISTANCES (vs Ground Truth):
  QED: 0.0234
  SA Score: 0.1456
  LogP: 0.3421
  Molecular Weight: 45.2341
  Num Atoms: 2.1234
  Vina Score: 0.4523 ⭐

🏆 OVERALL QUALITY SCORE: 11/11 (100.0%)
```

### Comparison Table

```
Model            QED (W↓)  SA (W↓)  LogP (W↓)  Vina (W↓)  Validity  Vina (avg)
film_trained     0.0234    0.1456   0.3421     0.4523     0.980     -7.234
no_film          0.0456    0.1789   0.4123     0.5234     0.970     -6.891
pca_correct_2    0.0312    0.1567   0.3789     0.4789     0.975     -7.123
```

---

## Troubleshooting

### Issue: "No receptor found for pocket"

**Solution:** Ensure test data directory contains matching PDB files:
```bash
ls data/dummy_testing_dataset_10_tests/test/*-pocket10.pdb
```

### Issue: "Vina scoring taking too long"

**Solutions:**
1. Use score_only mode (default) instead of full docking
2. Reduce exhaustiveness: `--vina_exhaustiveness 8`
3. Disable Vina for quick tests: `--disable_vina`

### Issue: "Different Vina scores between runs"

**Explanation:** Normal variation due to stochastic search. Use higher exhaustiveness for more reproducibility.

### Issue: "High Vina Wasserstein distance"

**Interpretation:** Generated molecules have different binding affinity distribution than ground truth. This could indicate:
- Model generating different binding modes
- Need for binding affinity conditioning during training
- Opportunity for FiLM fine-tuning improvement

---

## Integration with Your Thesis

### Current Status
✅ Ground truth baseline established
✅ Vina docking implemented
✅ Wasserstein distance calculation ready
✅ Comparison table includes Vina scores

### Next Steps

1. **Reanalyze existing baselines with Vina:**
   ```bash
   ./reanalyze_all_with_vina.sh  # Script below
   ```

2. **Track Vina scores during training:**
   - Add to evaluation metrics
   - Log Vina Wasserstein distance per epoch
   - Use as model selection criterion

3. **Publication results:**
   - Run with `--vina_full_docking --vina_exhaustiveness 32`
   - Generate comparison table
   - Statistical significance testing

### Batch Reanalysis Script

Create `reanalyze_all_with_vina.sh`:

```bash
#!/bin/bash
# Reanalyze all baseline experiments with Vina

BASELINE_DIR="thesis_work/experiments/day5_film_finetuning/outputs/baseline_comparison_v2"
GROUND_TRUTH="data/dummy_testing_dataset_10_tests/ground_truth_with_vina_exhaustiveness16.csv"

echo "Reanalyzing all baselines with Vina docking..."
echo "=============================================="

for dir in "$BASELINE_DIR"/*/; do
    name=$(basename "$dir")
    echo ""
    echo "Processing: $name"
    uv run python analyze_results_with_vina.py \
        "$dir" \
        --ground_truth "$GROUND_TRUTH" \
        --vina_exhaustiveness 16
done

echo ""
echo "Creating comparison table..."
uv run python create_comparison_table.py --baseline_dir "$BASELINE_DIR"

echo ""
echo "✓ Complete! Check: $BASELINE_DIR/comparison_table.csv"
```

Make executable:
```bash
chmod +x reanalyze_all_with_vina.sh
./reanalyze_all_with_vina.sh
```

---

## Files Summary

| File | Purpose |
|------|---------|
| `create_docking_ground_truth.py` | Generate ground truth with Vina |
| `analyze_results_with_vina.py` | Analyze generated molecules + Vina |
| `create_comparison_table.py` | Create comparison table (updated) |
| `analyze_results.py` | Original (without Vina) |
| `analysis/vina_docking.py` | Core Vina module |

---

## Questions?

- **API reference:** `analysis/README_VINA.md`
- **Setup details:** `DOCKING_ANALYSIS_COMPLETE.md`
- **Quick reference:** `VINA_QUICK_REFERENCE.md`
