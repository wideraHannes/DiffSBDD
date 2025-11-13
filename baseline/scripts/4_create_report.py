"""
Create baseline evaluation report.

Usage:
    python 4_create_report.py
"""

import json
import sys
from pathlib import Path
from datetime import datetime


def create_report(
    metrics_path='baseline/results/metrics.json',
    output_path='baseline/results/REPORT.md'
):
    """Generate markdown report from metrics."""

    print("="*80)
    print("STEP 4: CREATE REPORT")
    print("="*80)

    metrics_path = Path(metrics_path)
    if not metrics_path.exists():
        print(f"\nERROR: Metrics file not found: {metrics_path}")
        print("Please run: python 3_compute_metrics.py first")
        return False

    # Load metrics
    print(f"\nLoading metrics from: {metrics_path}")
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)

    print(f"✓ Loaded metrics for {metrics['n_molecules']} molecules")

    # Create report
    report = f"""# Baseline Evaluation Report (Small-Scale)

**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Model**: DiffSBDD Baseline (CrossDocked full-atom conditional)
**Checkpoint**: `checkpoints/crossdocked_fullatom_cond.ckpt`
**Test Set**: 10-pocket subset

---

## Summary Statistics

| Metric | Value |
|--------|-------|
| Test Pockets | {metrics['n_pockets']} |
| Total Molecules | {metrics['n_molecules']} |
| **Validity** | **{metrics['validity']:.1%}** |
| **Uniqueness** | **{metrics['uniqueness']:.1%}** ({metrics.get('n_unique', 'N/A')}/{metrics['n_molecules']}) |

---

## Molecular Properties

### Drug-likeness (QED)
"""

    if 'qed_mean' in metrics:
        report += f"""
- **Mean**: {metrics['qed_mean']:.3f} ± {metrics['qed_std']:.3f}
- **Range**: [{metrics['qed_min']:.3f}, {metrics['qed_max']:.3f}]
- **Interpretation**: 0-1 scale, higher is better. >0.5 is good drug-likeness.
"""
    else:
        report += "\n*Not available*\n"

    report += "\n### Synthetic Accessibility (SA Score)\n"

    if 'sa_mean' in metrics:
        report += f"""
- **Mean**: {metrics['sa_mean']:.3f} ± {metrics['sa_std']:.3f}
- **Range**: [{metrics['sa_min']:.3f}, {metrics['sa_max']:.3f}]
- **Interpretation**: 1-10 scale, lower is better. <3 is good synthetic accessibility.
"""
    else:
        report += "\n*Not available* (install rdkit.Contrib.SA_Score)\n"

    report += "\n### Lipophilicity (LogP)\n"

    if 'logp_mean' in metrics:
        report += f"""
- **Mean**: {metrics['logp_mean']:.3f} ± {metrics['logp_std']:.3f}
- **Range**: [{metrics['logp_min']:.3f}, {metrics['logp_max']:.3f}]
- **Interpretation**: Optimal range 0-3 for drug-likeness.
"""
    else:
        report += "\n*Not available*\n"

    report += "\n### Molecular Weight\n"

    if 'mw_mean' in metrics:
        report += f"""
- **Mean**: {metrics['mw_mean']:.1f} ± {metrics['mw_std']:.1f} Da
- **Range**: [{metrics['mw_min']:.1f}, {metrics['mw_max']:.1f}] Da
- **Interpretation**: Optimal range 160-500 Da for drug-likeness (Lipinski's Rule).
"""
    else:
        report += "\n*Not available*\n"

    report += "\n### Structure Statistics\n"

    if 'n_atoms_mean' in metrics:
        report += f"""
- **Atoms**: {metrics['n_atoms_mean']:.1f} ± {metrics['n_atoms_std']:.1f} (range: {metrics['n_atoms_min']}-{metrics['n_atoms_max']})
- **Bonds**: {metrics.get('n_bonds_mean', 'N/A'):.1f} ± {metrics.get('n_bonds_std', 0):.1f} (range: {metrics.get('n_bonds_min', 'N/A')}-{metrics.get('n_bonds_max', 'N/A')})
"""
    else:
        report += "\n*Not available*\n"

    report += """
---

## Interpretation

### Overall Quality
"""

    # Interpret results
    qed_good = metrics.get('qed_mean', 0) > 0.4
    sa_good = metrics.get('sa_mean', 10) < 3.5
    validity_good = metrics.get('validity', 0) > 0.6
    uniqueness_good = metrics.get('uniqueness', 0) > 0.8

    quality_score = sum([qed_good, sa_good, validity_good, uniqueness_good])

    if quality_score >= 3:
        quality = "**GOOD** ✅"
    elif quality_score >= 2:
        quality = "**ACCEPTABLE** ⚠️"
    else:
        quality = "**NEEDS IMPROVEMENT** ❌"

    report += f"\n{quality}\n\n"

    report += f"""
- Validity ({metrics['validity']:.1%}): {'✅ Good' if validity_good else '⚠️ Low'}
- Uniqueness ({metrics['uniqueness']:.1%}): {'✅ Good' if uniqueness_good else '⚠️ Low'}
- QED ({metrics.get('qed_mean', 'N/A'):.3f}): {'✅ Good' if qed_good else '⚠️ Low'}
- SA Score ({metrics.get('sa_mean', 'N/A'):.3f}): {'✅ Good' if sa_good else '⚠️ High (harder to synthesize)'}

### Comparison to Literature

Expected baseline performance (from paper):
- Validity: ~72%
- QED: ~0.45
- SA Score: ~3.2

**This small-scale test {'matches' if abs(metrics.get('qed_mean', 0) - 0.45) < 0.1 else 'differs from'} expected baseline.**

---

## Next Steps

### If Results Look Good (✅)

1. **Scale up** to full test set (100+ pockets)
   - Run full baseline evaluation
   - Generate more molecules per pocket (100 instead of 20)

2. **Proceed with ESM-C integration**
   - Data re-processing (add residue IDs)
   - ESM-C embedding pre-computation
   - Model training

3. **Compare** ESM-C model to these baseline numbers
   - Target: >5% improvement in validity or QED
   - Statistical significance: p < 0.05

### If Results Look Poor (❌)

**Check these issues**:

1. **Wrong checkpoint**
   - Verify it's the conditional model
   - Check it's trained on CrossDocked

2. **Data mismatch**
   - Verify test set is correct
   - Check preprocessing matches training

3. **Generation parameters**
   - Try increasing temperature
   - Check sampling steps

---

## Files

- **Molecules**: `baseline/results/molecules/*.sdf`
- **Metrics**: `baseline/results/metrics.json`
- **Report**: `baseline/results/REPORT.md`
- **Summary**: `baseline/results/generation_summary.json`

---

## Detailed Metrics (JSON)

For programmatic access, see: `{metrics_path}`

Key metrics available:
- `validity`: Fraction of chemically valid molecules
- `uniqueness`: Fraction of unique SMILES
- `qed_mean`, `qed_std`: Drug-likeness statistics
- `sa_mean`, `sa_std`: Synthetic accessibility statistics
- `logp_mean`, `logp_std`: Lipophilicity statistics
- `mw_mean`, `mw_std`: Molecular weight statistics
- `detailed_scores`: Per-molecule scores for analysis

---

*Report generated by `baseline/scripts/4_create_report.py`*
"""

    # Save report
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        f.write(report)

    print(f"\n✓ Report saved to: {output_path}")

    # Also print summary to console
    print("\n" + "="*80)
    print("BASELINE METRICS SUMMARY")
    print("="*80)
    print(f"Validity:    {metrics['validity']:.1%}")
    print(f"Uniqueness:  {metrics['uniqueness']:.1%}")
    if 'qed_mean' in metrics:
        print(f"QED:         {metrics['qed_mean']:.3f} ± {metrics['qed_std']:.3f}")
    if 'sa_mean' in metrics:
        print(f"SA Score:    {metrics['sa_mean']:.3f} ± {metrics['sa_std']:.3f}")
    if 'logp_mean' in metrics:
        print(f"LogP:        {metrics['logp_mean']:.3f} ± {metrics['logp_std']:.3f}")
    if 'mw_mean' in metrics:
        print(f"Mol Weight:  {metrics['mw_mean']:.1f} ± {metrics['mw_std']:.1f} Da")
    print("="*80)

    return True


if __name__ == '__main__':
    success = create_report()

    if success:
        print("\n" + "="*80)
        print("STEP 4 COMPLETE - BASELINE EVALUATION FINISHED!")
        print("="*80)
        print("\nAll results saved in: baseline/results/")
        print("\nNext steps:")
        print("1. Review the report: baseline/results/REPORT.md")
        print("2. If results look good, scale up to full test set")
        print("3. Proceed with ESM-C integration")
    else:
        print("\n" + "="*80)
        print("STEP 4 FAILED")
        print("="*80)
        sys.exit(1)
