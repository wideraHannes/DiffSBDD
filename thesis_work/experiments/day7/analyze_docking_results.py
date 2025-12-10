"""
Docking Experiment B: ESM Steering Analysis

Analyzes whether λ=2 ESM steering improves ligand quality vs baseline by:
1. Distribution analysis (histograms, KDE, violin/box plots)
2. Summary statistics (mean, median, std, top-1, top-5, hit rates)
3. Statistical testing (Mann-Whitney U, KS test)
4. Property drift analysis (MW, LogP, QED, SA, num_atoms)
5. Stratified analysis by MW bins
6. Chemical diversity (scaffolds, fingerprint similarity)
7. Results summary with plots and tables

Usage:
    uv run python analyze_docking_results.py
    uv run python analyze_docking_results.py --data-dir baseline_comparison_v3
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# RDKit imports (optional)
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors
    from rdkit.Chem.Scaffolds import MurckoScaffold
    from rdkit import DataStructs
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("Warning: RDKit not available. Diversity analysis will be skipped.")


def load_data(data_dir):
    """Load baseline and lambda2 detailed CSV files."""
    baseline_path = Path(data_dir) / 'baseline' / 'analysis_summary.detailed.csv'
    lambda2_path = Path(data_dir) / 'output_lambda2' / 'analysis_summary.detailed.csv'

    print(f"\nLoading data from:")
    print(f"  Baseline: {baseline_path}")
    print(f"  Lambda=2: {lambda2_path}")

    baseline_df = pd.read_csv(baseline_path)
    lambda2_df = pd.read_csv(lambda2_path)

    # Add condition labels
    baseline_df['condition'] = 'baseline'
    lambda2_df['condition'] = 'lambda2'

    # Combine
    df_combined = pd.concat([baseline_df, lambda2_df], ignore_index=True)

    print(f"\nLoaded {len(baseline_df)} baseline molecules and {len(lambda2_df)} lambda=2 molecules")
    print(f"Total: {len(df_combined)} molecules\n")

    return df_combined, baseline_df, lambda2_df


def plot_vina_distributions(df_combined, output_dir):
    """Generate distribution plots for Vina scores."""
    print("Generating Vina score distribution plots...")

    # 2.1: Histogram + KDE + Violin
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram with KDE overlay
    for condition, color in [('baseline', '#1f77b4'), ('lambda2', '#ff7f0e')]:
        data = df_combined[df_combined['condition'] == condition]['vina_score'].dropna()
        axes[0].hist(data, bins=30, alpha=0.5, label=condition, color=color, density=True)

        # KDE overlay
        if len(data) > 0:
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(data)
            x_range = np.linspace(data.min(), data.max(), 200)
            axes[0].plot(x_range, kde(x_range), color=color, linewidth=2)

    axes[0].set_xlabel('Vina Score (kcal/mol)')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Vina Score Distribution')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Violin plot
    baseline_vina = df_combined[df_combined['condition'] == 'baseline']['vina_score'].dropna()
    lambda2_vina = df_combined[df_combined['condition'] == 'lambda2']['vina_score'].dropna()

    parts = axes[1].violinplot([baseline_vina, lambda2_vina],
                               positions=[0, 1], showmeans=True, showmedians=True)
    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels(['Baseline', 'Lambda=2'])
    axes[1].set_ylabel('Vina Score (kcal/mol)')
    axes[1].set_title('Vina Score Violin Plot')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    dist_path = Path(output_dir) / 'vina_score_distributions.png'
    plt.savefig(dist_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {dist_path}")

    # 2.2: Box Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    df_combined.boxplot(column='vina_score', by='condition', ax=ax)
    ax.set_xlabel('Condition')
    ax.set_ylabel('Vina Score (kcal/mol)')
    ax.set_title('Vina Score Box Plot Comparison')
    plt.suptitle('')  # Remove default title

    box_path = Path(output_dir) / 'vina_score_boxplot.png'
    plt.savefig(box_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {box_path}")


def compute_summary_stats(df, condition_name):
    """Compute comprehensive summary statistics."""
    vina_scores = df[df['condition'] == condition_name]['vina_score'].dropna()

    # Basic stats
    mean_score = vina_scores.mean()
    median_score = vina_scores.median()
    std_score = vina_scores.std()

    # Top performers
    top1_score = vina_scores.min()  # Best score (most negative)
    top5_scores = vina_scores.nsmallest(min(5, len(vina_scores)))
    top5_mean = top5_scores.mean()

    # Hit rate (better than -7.0 threshold)
    hit_rate_7 = (vina_scores < -7.0).sum() / len(vina_scores) if len(vina_scores) > 0 else 0

    # Hit rate (better than baseline median)
    baseline_median = df[df['condition'] == 'baseline']['vina_score'].dropna().median()
    hit_rate_baseline = (vina_scores < baseline_median).sum() / len(vina_scores) if len(vina_scores) > 0 else 0

    return {
        'mean': mean_score,
        'median': median_score,
        'std': std_score,
        'top1': top1_score,
        'top5_mean': top5_mean,
        'hit_rate_-7.0': hit_rate_7,
        f'hit_rate_<baseline_median_{baseline_median:.2f}': hit_rate_baseline,
        'count': len(vina_scores)
    }


def print_summary_statistics(df_combined):
    """Print comprehensive summary statistics."""
    print("\n" + "="*80)
    print("VINA SCORE SUMMARY STATISTICS")
    print("="*80 + "\n")

    baseline_stats = compute_summary_stats(df_combined, 'baseline')
    lambda2_stats = compute_summary_stats(df_combined, 'lambda2')

    print(f"{'Metric':<35} {'Baseline':<15} {'Lambda=2':<15} {'Delta':<15}")
    print("-" * 80)

    for key in baseline_stats.keys():
        if key == 'count':
            print(f"{key:<35} {baseline_stats[key]:<15} {lambda2_stats[key]:<15}")
        else:
            delta = lambda2_stats[key] - baseline_stats[key]
            print(f"{key:<35} {baseline_stats[key]:<15.4f} {lambda2_stats[key]:<15.4f} {delta:<+15.4f}")

    return baseline_stats, lambda2_stats


def statistical_tests(df_combined):
    """Perform statistical tests comparing conditions."""
    print("\n" + "="*80)
    print("STATISTICAL TESTING")
    print("="*80 + "\n")

    baseline_vina = df_combined[df_combined['condition'] == 'baseline']['vina_score'].dropna()
    lambda2_vina = df_combined[df_combined['condition'] == 'lambda2']['vina_score'].dropna()

    # Mann-Whitney U test
    statistic_mw, p_value_mw = stats.mannwhitneyu(baseline_vina, lambda2_vina, alternative='two-sided')

    print("Mann-Whitney U Test:")
    print(f"  Statistic: {statistic_mw:.4f}")
    print(f"  P-value: {p_value_mw:.6f}")
    print(f"  Significant (α=0.05): {'YES' if p_value_mw < 0.05 else 'NO'}")
    print(f"  Interpretation: {'Lambda=2 significantly different from baseline' if p_value_mw < 0.05 else 'No significant difference'}")

    # Kolmogorov-Smirnov test
    statistic_ks, p_value_ks = stats.ks_2samp(baseline_vina, lambda2_vina)
    print(f"\nKolmogorov-Smirnov Test:")
    print(f"  Statistic: {statistic_ks:.4f}")
    print(f"  P-value: {p_value_ks:.6f}")
    print(f"  Significant (α=0.05): {'YES' if p_value_ks < 0.05 else 'NO'}")

    print(f"\n**LIMITATION**: No pocket/target IDs available for paired analysis.")
    print(f"Using global Mann-Whitney U and KS tests instead of paired Wilcoxon test.")

    return p_value_mw, p_value_ks


def property_drift_analysis(df_combined, output_dir):
    """Analyze property drift between conditions."""
    print("\n" + "="*80)
    print("PROPERTY DRIFT ANALYSIS")
    print("="*80 + "\n")

    properties = ['molecular_weight', 'logp', 'qed', 'sa_score', 'num_atoms']

    # Plot distributions
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for idx, prop in enumerate(properties):
        ax = axes[idx]
        for condition in ['baseline', 'lambda2']:
            data = df_combined[df_combined['condition'] == condition][prop].dropna()
            ax.hist(data, bins=20, alpha=0.5, label=condition, density=True)

        ax.set_xlabel(prop.replace('_', ' ').title())
        ax.set_ylabel('Density')
        ax.set_title(f'{prop.replace("_", " ").title()} Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Remove extra subplot
    axes[-1].axis('off')

    plt.tight_layout()
    prop_path = Path(output_dir) / 'property_distributions.png'
    plt.savefig(prop_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved property distributions: {prop_path}\n")

    # Statistical tests
    print(f"{'Property':<20} {'Baseline Mean':<15} {'Lambda=2 Mean':<15} {'P-value':<12} {'Significant?':<12}")
    print("-" * 80)

    property_drifts = {}
    for prop in properties:
        baseline_prop = df_combined[df_combined['condition'] == 'baseline'][prop].dropna()
        lambda2_prop = df_combined[df_combined['condition'] == 'lambda2'][prop].dropna()

        _, p_val = stats.mannwhitneyu(baseline_prop, lambda2_prop, alternative='two-sided')
        property_drifts[prop] = p_val

        print(f"{prop:<20} {baseline_prop.mean():<15.3f} {lambda2_prop.mean():<15.3f} {p_val:<12.6f} {'YES' if p_val < 0.05 else 'NO':<12}")

    # Correlation analysis
    print("\n" + "="*80)
    print("CORRELATION ANALYSIS (Vina vs MW/num_atoms)")
    print("="*80 + "\n")

    for condition in ['baseline', 'lambda2']:
        df_cond = df_combined[df_combined['condition'] == condition]

        # Filter out NaN values for correlation
        vina_vals = df_cond['vina_score'].dropna()
        mw_vals = df_cond.loc[vina_vals.index, 'molecular_weight'].dropna()
        atoms_vals = df_cond.loc[vina_vals.index, 'num_atoms'].dropna()

        # Align indices
        common_mw_idx = vina_vals.index.intersection(mw_vals.index)
        common_atoms_idx = vina_vals.index.intersection(atoms_vals.index)

        if len(common_mw_idx) > 0:
            corr_mw, p_mw = stats.pearsonr(
                vina_vals.loc[common_mw_idx],
                df_cond.loc[common_mw_idx, 'molecular_weight']
            )
        else:
            corr_mw, p_mw = np.nan, np.nan

        if len(common_atoms_idx) > 0:
            corr_atoms, p_atoms = stats.pearsonr(
                vina_vals.loc[common_atoms_idx],
                df_cond.loc[common_atoms_idx, 'num_atoms']
            )
        else:
            corr_atoms, p_atoms = np.nan, np.nan

        print(f"{condition.upper()}:")
        print(f"  Vina vs MW:        r={corr_mw:.3f}, p={p_mw:.6f}")
        print(f"  Vina vs num_atoms: r={corr_atoms:.3f}, p={p_atoms:.6f}\n")

    return property_drifts


def stratified_analysis(df_combined):
    """Perform stratified analysis by molecular weight bins."""
    print("\n" + "="*80)
    print("STRATIFIED ANALYSIS BY MOLECULAR WEIGHT")
    print("="*80 + "\n")

    def assign_mw_bin(mw):
        if mw < 300:
            return '<300'
        elif mw < 450:
            return '300-450'
        else:
            return '>450'

    df_combined['mw_bin'] = df_combined['molecular_weight'].apply(assign_mw_bin)

    print(f"{'MW Bin':<15} {'Condition':<12} {'Count':<8} {'Vina Mean':<12} {'Vina Median':<12}")
    print("-" * 70)

    for mw_bin in ['<300', '300-450', '>450']:
        for condition in ['baseline', 'lambda2']:
            df_subset = df_combined[(df_combined['mw_bin'] == mw_bin) & (df_combined['condition'] == condition)]
            vina_subset = df_subset['vina_score'].dropna()

            if len(vina_subset) > 0:
                print(f"{mw_bin:<15} {condition:<12} {len(vina_subset):<8} {vina_subset.mean():<12.4f} {vina_subset.median():<12.4f}")

        # Test difference within bin
        baseline_bin = df_combined[(df_combined['mw_bin'] == mw_bin) & (df_combined['condition'] == 'baseline')]['vina_score'].dropna()
        lambda2_bin = df_combined[(df_combined['mw_bin'] == mw_bin) & (df_combined['condition'] == 'lambda2')]['vina_score'].dropna()

        if len(baseline_bin) > 0 and len(lambda2_bin) > 0:
            _, p_val = stats.mannwhitneyu(baseline_bin, lambda2_bin, alternative='two-sided')
            print(f"  → MW bin {mw_bin} p-value: {p_val:.6f} ({'Significant' if p_val < 0.05 else 'Not significant'})\n")


def diversity_analysis(df_combined):
    """Analyze chemical diversity using scaffolds and fingerprints."""
    if not RDKIT_AVAILABLE:
        print("\n[SKIPPED] Chemical diversity analysis (RDKit not available)")
        return

    print("\n" + "="*80)
    print("CHEMICAL DIVERSITY ANALYSIS")
    print("="*80 + "\n")

    def compute_morgan_fingerprints(smiles_list):
        """Compute Morgan fingerprints for molecules."""
        fps = []
        valid_indices = []
        for idx, smiles in enumerate(smiles_list):
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
                    fps.append(fp)
                    valid_indices.append(idx)
            except:
                continue
        return fps, valid_indices

    def compute_scaffold_diversity(smiles_list):
        """Compute scaffold diversity using Bemis-Murcko scaffolds."""
        scaffolds = set()
        for smiles in smiles_list:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
                    scaffolds.add(Chem.MolToSmiles(scaffold))
            except:
                continue

        return len(scaffolds), len(scaffolds) / len(smiles_list) if len(smiles_list) > 0 else 0

    for condition in ['baseline', 'lambda2']:
        smiles_list = df_combined[df_combined['condition'] == condition]['smiles'].dropna().tolist()

        # Scaffold diversity
        n_scaffolds, scaffold_ratio = compute_scaffold_diversity(smiles_list)

        # Fingerprint-based internal similarity
        fps, valid_indices = compute_morgan_fingerprints(smiles_list)

        if len(fps) > 1:
            # Compute pairwise Tanimoto similarities
            similarities = []
            for i in range(len(fps)):
                for j in range(i+1, len(fps)):
                    sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
                    similarities.append(sim)

            avg_similarity = np.mean(similarities)

            print(f"{condition.upper()}:")
            print(f"  Total molecules: {len(smiles_list)}")
            print(f"  Unique scaffolds: {n_scaffolds}")
            print(f"  Scaffold diversity ratio: {scaffold_ratio:.3f}")
            print(f"  Avg internal Tanimoto similarity: {avg_similarity:.3f}")
            print(f"  Diversity score (1 - similarity): {1 - avg_similarity:.3f}\n")


def generate_summary_table(df_combined, output_dir):
    """Generate and save summary table."""
    print("\nGenerating summary table...")

    summary_data = []

    for condition in ['baseline', 'lambda2']:
        df_cond = df_combined[df_combined['condition'] == condition]
        vina = df_cond['vina_score'].dropna()

        row = {
            'Condition': condition,
            'N': len(vina),
            'Vina Mean': vina.mean(),
            'Vina Median': vina.median(),
            'Vina Std': vina.std(),
            'Vina Best': vina.min(),
            'Vina Top5 Mean': vina.nsmallest(5).mean(),
            'Hit Rate (<-7.0)': (vina < -7.0).sum() / len(vina),
            'MW Mean': df_cond['molecular_weight'].mean(),
            'LogP Mean': df_cond['logp'].mean(),
            'QED Mean': df_cond['qed'].mean(),
            'SA Mean': df_cond['sa_score'].mean(),
        }
        summary_data.append(row)

    summary_df = pd.DataFrame(summary_data)
    table_path = Path(output_dir) / 'analysis_summary_table.csv'
    summary_df.to_csv(table_path, index=False)
    print(f"  Saved: {table_path}")


def generate_results_paragraph(baseline_stats, lambda2_stats, p_value_mw, property_drifts, output_dir):
    """Generate concise results paragraph."""
    print("\n" + "="*80)
    print("GENERATING RESULTS SUMMARY")
    print("="*80 + "\n")

    vina_improvement = lambda2_stats['median'] - baseline_stats['median']
    improvement_pct = (vina_improvement / abs(baseline_stats['median'])) * 100

    # Check significance
    is_significant = p_value_mw < 0.05

    # Check property drift
    significant_drifts = [prop for prop, p_val in property_drifts.items() if p_val < 0.05]

    paragraph = f"""
RESULTS SUMMARY: Docking Experiment B (λ=2 ESM Steering vs Baseline)
{'='*80}

Dataset:
  - Baseline molecules: {baseline_stats['count']}
  - Lambda=2 molecules: {lambda2_stats['count']}

Vina Score Performance:
  - Baseline: median {baseline_stats['median']:.2f} kcal/mol (mean {baseline_stats['mean']:.2f} ± {baseline_stats['std']:.2f})
  - Lambda=2: median {lambda2_stats['median']:.2f} kcal/mol (mean {lambda2_stats['mean']:.2f} ± {lambda2_stats['std']:.2f})
  - Improvement: {vina_improvement:+.3f} kcal/mol ({improvement_pct:+.1f}%)
  - Statistical significance: {'SIGNIFICANT' if is_significant else 'NOT SIGNIFICANT'} (Mann-Whitney p={p_value_mw:.6f})

Top Performer Analysis:
  - Best score:      Baseline {baseline_stats['top1']:.2f} kcal/mol vs Lambda=2 {lambda2_stats['top1']:.2f} kcal/mol
  - Top-5 mean:      Baseline {baseline_stats['top5_mean']:.2f} kcal/mol vs Lambda=2 {lambda2_stats['top5_mean']:.2f} kcal/mol
  - Hit rate (<-7.0): Baseline {baseline_stats['hit_rate_-7.0']:.1%} vs Lambda=2 {lambda2_stats['hit_rate_-7.0']:.1%}

Property Drift Assessment:
  - Significant property changes detected in: {', '.join(significant_drifts) if significant_drifts else 'None'}
  - {'⚠ WARNING: Vina improvements may be confounded by property drift.' if significant_drifts else '✓ No major property drift detected.'}

{'='*80}

CONCLUSION:
  {'✓ Lambda=2 shows meaningful enrichment over baseline.' if is_significant and not significant_drifts else '✗ Lambda=2 does NOT show robust improvement over baseline.'}
  {'The effect appears genuine and not driven by trivial property inflation.' if is_significant and not significant_drifts else 'The effect is either not significant or likely confounded by property changes (MW, num_atoms, etc.).'}

RECOMMENDATION:
  {'→ Proceed with λ=2 for further docking validation.' if is_significant and not significant_drifts else '→ Consider alternative conditioning strategies or focus on learned FiLM adaptation.'}

{'='*80}

Notes:
  - **Limitation**: No pocket/target IDs available, so paired per-pocket analysis not possible.
  - Used global Mann-Whitney U test instead of paired Wilcoxon signed-rank test.
  - Sample size (~{baseline_stats['count']} molecules per condition) provides moderate statistical power.
"""

    print(paragraph)

    # Save to file
    results_path = Path(output_dir) / 'results_summary.txt'
    with open(results_path, 'w') as f:
        f.write(paragraph)
    print(f"\nSaved results summary: {results_path}")

    return paragraph


def main():
    parser = argparse.ArgumentParser(description='Analyze docking results comparing baseline vs λ=2')
    parser.add_argument('--data-dir', type=str,
                       default='thesis_work/experiments/day5_film_finetuning/outputs/baseline_comparison_v3',
                       help='Directory containing baseline/ and output_lambda2/ subdirectories')

    args = parser.parse_args()

    print("\n" + "="*80)
    print("DOCKING EXPERIMENT B: ESM STEERING ANALYSIS")
    print("="*80)

    # Load data
    df_combined, baseline_df, lambda2_df = load_data(args.data_dir)

    # Generate plots
    plot_vina_distributions(df_combined, args.data_dir)

    # Summary statistics
    baseline_stats, lambda2_stats = print_summary_statistics(df_combined)

    # Statistical tests
    p_value_mw, p_value_ks = statistical_tests(df_combined)

    # Property drift
    property_drifts = property_drift_analysis(df_combined, args.data_dir)

    # Stratified analysis
    stratified_analysis(df_combined)

    # Diversity analysis
    diversity_analysis(df_combined)

    # Summary table
    generate_summary_table(df_combined, args.data_dir)

    # Results paragraph
    generate_results_paragraph(baseline_stats, lambda2_stats, p_value_mw, property_drifts, args.data_dir)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nOutput files saved in: {args.data_dir}/")
    print("  - vina_score_distributions.png")
    print("  - vina_score_boxplot.png")
    print("  - property_distributions.png")
    print("  - analysis_summary_table.csv")
    print("  - results_summary.txt")


if __name__ == '__main__':
    main()
