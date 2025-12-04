#!/bin/bash
# Verification script for thesis structure

echo "🔍 Verifying Thesis Structure..."
echo ""

# Check main directories
echo "📁 Main Directories:"
for dir in thesis_work esmc_integration data; do
    if [ -d "$dir" ]; then
        echo "  ✅ $dir/"
    else
        echo "  ❌ $dir/ NOT FOUND"
    fi
done

echo ""
echo "📁 Thesis Work Subdirectories:"
for dir in daily_logs documentation experiments analysis configs results; do
    if [ -d "thesis_work/$dir" ]; then
        echo "  ✅ thesis_work/$dir/"
    else
        echo "  ❌ thesis_work/$dir/ NOT FOUND"
    fi
done

echo ""
echo "📁 ESM-C Integration:"
for dir in extraction tests analysis embeddings_cache; do
    if [ -d "esmc_integration/$dir" ]; then
        echo "  ✅ esmc_integration/$dir/"
    else
        echo "  ❌ esmc_integration/$dir/ NOT FOUND"
    fi
done

echo ""
echo "📄 Key Files:"
files=(
    "THESIS_README.md"
    "thesis_work/README.md"
    "thesis_work/daily_logs/INDEX.md"
    "thesis_work/documentation/progress_tracker.md"
    "thesis_work/documentation/implementation_plan.md"
    "esmc_integration/extraction/extract_esmc_embeddings.py"
)

for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✅ $file"
    else
        echo "  ❌ $file NOT FOUND"
    fi
done

echo ""
echo "✨ Structure verification complete!"
