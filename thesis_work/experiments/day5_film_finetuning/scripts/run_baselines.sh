#!/bin/bash
# Quick script to run all three baseline experiments
# Usage: bash thesis_work/experiments/day5_film_finetuning/scripts/run_baselines.sh

set -e  # Exit on error

echo "================================"
echo "Running FiLM Baseline Comparison"
echo "================================"
echo ""

# Run the comparison script
uv run python thesis_work/experiments/day5_film_finetuning/scripts/run_baseline_comparison.py \
    --n_samples 5 \
    --batch_size 1 \
    --test_dir data/dummy_testing_dataset_10_tests/test \
    --output_dir thesis_work/experiments/day5_film_finetuning/outputs/baseline_comparison_v2

echo ""
echo "================================"
echo "✅ Baseline comparison complete!"
echo "================================"
echo ""
echo "Results saved to:"
echo "  thesis_work/experiments/day5_film_finetuning/outputs/baseline_comparison/"
echo ""
echo "View comparison:"
echo "  cat thesis_work/experiments/day5_film_finetuning/outputs/baseline_comparison/baseline_comparison.csv"
