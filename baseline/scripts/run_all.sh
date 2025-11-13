#!/bin/bash

# Baseline Evaluation Pipeline
# Runs complete small-scale baseline evaluation (10 pockets, 20 molecules each)

set -e  # Exit on error

echo "================================================================================"
echo "BASELINE EVALUATION PIPELINE"
echo "================================================================================"
echo ""
echo "This will:"
echo "  1. Extract 10-pocket test subset"
echo "  2. Generate 20 molecules per pocket (200 total)"
echo "  3. Compute metrics (validity, QED, SA, etc.)"
echo "  4. Create detailed report"
echo ""
echo "Expected time: 30-60 minutes (depends on GPU)"
echo ""
read -p "Press Enter to continue or Ctrl+C to cancel..."

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Change to project root
cd ../..

echo ""
echo "Working directory: $(pwd)"
echo ""

# Step 1: Extract subset
echo "================================================================================"
echo "STEP 1/4: Extract Test Subset"
echo "================================================================================"
python baseline/scripts/1_extract_subset.py
if [ $? -ne 0 ]; then
    echo "ERROR: Step 1 failed!"
    exit 1
fi

echo ""
echo "Press Enter to continue to Step 2..."
read

# Step 2: Generate molecules
echo ""
echo "================================================================================"
echo "STEP 2/4: Generate Molecules"
echo "================================================================================"
echo "This may take 10-30 minutes depending on your GPU..."
echo ""
python baseline/scripts/2_generate_molecules.py
if [ $? -ne 0 ]; then
    echo "ERROR: Step 2 failed!"
    exit 1
fi

echo ""
echo "Press Enter to continue to Step 3..."
read

# Step 3: Compute metrics
echo ""
echo "================================================================================"
echo "STEP 3/4: Compute Metrics"
echo "================================================================================"
python baseline/scripts/3_compute_metrics.py
if [ $? -ne 0 ]; then
    echo "ERROR: Step 3 failed!"
    exit 1
fi

echo ""
echo "Press Enter to continue to Step 4..."
read

# Step 4: Create report
echo ""
echo "================================================================================"
echo "STEP 4/4: Create Report"
echo "================================================================================"
python baseline/scripts/4_create_report.py
if [ $? -ne 0 ]; then
    echo "ERROR: Step 4 failed!"
    exit 1
fi

# Success!
echo ""
echo "================================================================================"
echo "✓ BASELINE EVALUATION COMPLETE!"
echo "================================================================================"
echo ""
echo "Results saved in: baseline/results/"
echo ""
echo "Files created:"
echo "  - baseline/data/test_subset_10.npz          (10-pocket subset)"
echo "  - baseline/results/molecules/*.sdf          (generated molecules)"
echo "  - baseline/results/metrics.json             (computed metrics)"
echo "  - baseline/results/REPORT.md                (detailed report)"
echo "  - baseline/results/generation_summary.json  (generation stats)"
echo ""
echo "Next steps:"
echo "  1. Review report: cat baseline/results/REPORT.md"
echo "  2. If results look good, scale up to full test set"
echo "  3. Proceed with ESM-C integration"
echo ""
