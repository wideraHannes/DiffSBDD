#!/bin/bash
# Analyze all baseline experiments with Vina docking scores
# Calculates Vina scores and Wasserstein distances for all models

set -e  # Exit on error

# Configuration
BASELINE_DIR="thesis_work/experiments/day5_film_finetuning/outputs/baseline_comparison_v2"
GROUND_TRUTH="data/dummy_testing_dataset_10_tests/ground_truth_with_vina_exhaustiveness16.csv"
VINA_EXHAUSTIVENESS=8  # 8=fast, 16=recommended, 32=publication

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "========================================================================"
echo "  Analyzing All Baselines with Vina Docking Scores"
echo "========================================================================"
echo ""
echo "Baseline directory: $BASELINE_DIR"
echo "Ground truth: $GROUND_TRUTH"
echo "Vina exhaustiveness: $VINA_EXHAUSTIVENESS"
echo ""

# Check if ground truth exists
if [ ! -f "$GROUND_TRUTH" ]; then
    echo -e "${YELLOW}Warning: Ground truth file not found!${NC}"
    echo "Creating ground truth with Vina scores..."
    uv run python create_docking_ground_truth.py
    echo ""
fi

# Count baselines
baseline_count=$(find "$BASELINE_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l | tr -d ' ')
echo -e "${BLUE}Found $baseline_count baseline experiments${NC}"
echo ""

# Process each baseline
current=0
for dir in "$BASELINE_DIR"/*/; do
    current=$((current + 1))
    name=$(basename "$dir")

    echo "========================================================================"
    echo -e "${GREEN}[$current/$baseline_count] Processing: $name${NC}"
    echo "========================================================================"

    # Check if processed directory exists
    if [ ! -d "$dir/processed" ]; then
        echo -e "${YELLOW}  ⚠️  No processed directory found, skipping...${NC}"
        echo ""
        continue
    fi

    # Run analysis
    uv run python analyze_results_with_vina.py \
        "$dir" \
        --ground_truth "$GROUND_TRUTH" \
        --vina_exhaustiveness $VINA_EXHAUSTIVENESS

    # Check if successful
    if [ -f "$dir/analysis_summary.csv" ]; then
        echo -e "${GREEN}  ✓ Analysis complete${NC}"

        # Show quick stats
        if command -v python3 &> /dev/null; then
            vina_score=$(python3 -c "import pandas as pd; df=pd.read_csv('$dir/analysis_summary.csv'); print(f\"{df['avg_vina_score'].values[0]:.2f}\" if 'avg_vina_score' in df.columns else 'N/A')" 2>/dev/null || echo "N/A")
            vina_wass=$(python3 -c "import pandas as pd; df=pd.read_csv('$dir/analysis_summary.csv'); print(f\"{df['vina_wasserstein'].values[0]:.3f}\" if 'vina_wasserstein' in df.columns else 'N/A')" 2>/dev/null || echo "N/A")
            echo -e "  ${BLUE}Vina avg: $vina_score kcal/mol, Wasserstein: $vina_wass${NC}"
        fi
    else
        echo -e "${YELLOW}  ⚠️  Analysis may have failed${NC}"
    fi

    echo ""
done

echo "========================================================================"
echo -e "${GREEN}All baselines analyzed!${NC}"
echo "========================================================================"
echo ""

# Generate comparison table
echo "Creating comparison table..."
uv run python create_comparison_table.py \
    --baseline_dir "$BASELINE_DIR"

echo ""
echo -e "${GREEN}✓ Complete!${NC}"
echo ""
echo "Results:"
echo "  - Individual summaries: $BASELINE_DIR/*/analysis_summary.csv"
echo "  - Comparison table: $BASELINE_DIR/comparison_table.csv"
echo ""
echo "View comparison table:"
echo "  cat $BASELINE_DIR/comparison_table.csv"
echo "  # Or with formatting:"
echo "  column -t -s, $BASELINE_DIR/comparison_table.csv | less -S"
echo ""
