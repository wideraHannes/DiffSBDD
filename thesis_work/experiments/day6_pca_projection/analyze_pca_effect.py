"""
PCA Effect Distribution Analysis for h_residues Modulation

This script analyzes how PCA-projected ESM-C embeddings affect h_residues
during the diffusion process. It provides visibility into:
- Original ESM-C embedding distributions (960D)
- PCA transformation (960D → 32D)
- Scaled contribution (λ * z_esm_pca)
- Relative effect on h_residues

Usage:
    python analyze_pca_effect.py
    python analyze_pca_effect.py --data-path data/small_dataset_1000/train_esmc.npz
    python analyze_pca_effect.py --n-samples 500 --lambda-scale 0.2
    python analyze_pca_effect.py --no-plot  # Skip visualization
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
from scipy import stats
import argparse


def print_header(text):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(text)
    print("=" * 80 + "\n")


def print_success(text):
    """Print a success message."""
    print(f"✓ {text}")


def load_esmc_embeddings(data_path, n_samples=None):
    """Load ESM-C embeddings from npz file."""
    print_header("LOADING DATA")

    print(f"Loading embeddings from: {data_path}")
    data = np.load(data_path, allow_pickle=True)

    embeddings = data["embeddings"]
    names = data["names"]

    # Handle object dtype
    if embeddings.dtype == object:
        embeddings = np.stack([e for e in embeddings]).astype(np.float32)

    # Subsample if requested
    if n_samples is not None and n_samples < len(embeddings):
        embeddings = embeddings[:n_samples]
        names = names[:n_samples]

    print_success(f"Loaded {len(embeddings)} ESM-C embeddings from {data_path}")
    print_success(f"Embeddings shape: {embeddings.shape}")

    return embeddings, names


def load_pca_model(pca_path):
    """Load pre-trained PCA model."""
    print_header("LOADING PCA MODEL")

    print(f"Loading PCA model from: {pca_path}")

    with open(pca_path, "rb") as f:
        pca_model = pickle.load(f)

    print_success(f"Loaded PCA model: {pca_path}")
    print_success(f"PCA: 960D → {pca_model.n_components_}D")
    print_success(f"Variance explained: {pca_model.explained_variance_ratio_.sum() * 100:.2f}%")

    return pca_model


def analyze_stage1_esmc(embeddings):
    """Analyze original ESM-C embeddings (960D)."""
    print_header("STAGE 1: Original ESM-C Embeddings (960D)")

    # Overall statistics
    print("Distribution Statistics:")
    print(f"  Mean:        {embeddings.mean():.4f}")
    print(f"  Std:         {embeddings.std():.4f}")
    print(f"  Min:         {embeddings.min():.4f}")
    print(f"  Max:         {embeddings.max():.4f}")

    # L2 norm per sample
    l2_norms = np.linalg.norm(embeddings, axis=1)
    print(f"  L2 norm (mean): {l2_norms.mean():.4f}")
    print(f"  L2 norm (std):  {l2_norms.std():.4f}")

    # Per-dimension variance
    dim_variances = embeddings.var(axis=0)
    print(f"\nPer-Dimension Variance:")
    print(f"  Min variance: {dim_variances.min():.6f}")
    print(f"  Max variance: {dim_variances.max():.6f}")
    print(f"  Mean variance: {dim_variances.mean():.6f}")

    # Percentiles
    print(f"\nPercentiles:")
    for p in [5, 25, 50, 75, 95]:
        val = np.percentile(embeddings, p)
        print(f"  {p:2d}th percentile: {val:.4f}")

    # Distribution shape
    flattened = embeddings.flatten()
    skewness = stats.skew(flattened)
    kurtosis = stats.kurtosis(flattened)
    print(f"\nDistribution Shape:")
    print(f"  Skewness: {skewness:.4f} (0 = symmetric)")
    print(f"  Kurtosis: {kurtosis:.4f} (0 = Gaussian)")

    return {
        "embeddings": embeddings,
        "mean": embeddings.mean(),
        "std": embeddings.std(),
        "l2_norms": l2_norms,
        "dim_variances": dim_variances,
        "skewness": skewness,
        "kurtosis": kurtosis,
    }


def analyze_stage2_pca(embeddings, pca_model):
    """Analyze PCA transformation (960D → 32D)."""
    print_header("STAGE 2: PCA Transformation (960D → 32D)")

    # Transform using PCA (exactly as in dynamics.py)
    z_esm_pca = pca_model.transform(embeddings)
    n_components = z_esm_pca.shape[1]

    print(f"Transformed Embeddings (z_esm_pca):")
    print(f"  Shape:       {z_esm_pca.shape}")
    print(f"  Mean:        {z_esm_pca.mean():.4f}")
    print(f"  Std:         {z_esm_pca.std():.4f}")
    print(f"  Min:         {z_esm_pca.min():.4f}")
    print(f"  Max:         {z_esm_pca.max():.4f}")

    # Per-component analysis
    print(f"\nComponent Analysis (first 10 of {n_components}):")
    print(f"{'Component':>12} {'Mean':>8} {'Std':>8} {'Var%':>8}")
    print("-" * 40)

    for i in range(min(10, n_components)):
        comp_mean = z_esm_pca[:, i].mean()
        comp_std = z_esm_pca[:, i].std()
        var_pct = pca_model.explained_variance_ratio_[i] * 100
        print(f"  {i+1:10d} {comp_mean:8.3f} {comp_std:8.3f} {var_pct:7.2f}%")

    if n_components > 10:
        print(f"  ... ({n_components - 10} more components)")

    # Cumulative variance
    cumsum_var = pca_model.explained_variance_ratio_.cumsum()
    print(f"\nCumulative Variance Explained:")
    for n in [5, 10, 20, 32] if n_components >= 32 else [5, 10, n_components]:
        if n <= n_components:
            print(f"  First {n:2d} components: {cumsum_var[n-1] * 100:.2f}%")

    # L2 norm per sample
    l2_norms_pca = np.linalg.norm(z_esm_pca, axis=1)
    print(f"\nL2 Norms:")
    print(f"  Mean: {l2_norms_pca.mean():.4f}")
    print(f"  Std:  {l2_norms_pca.std():.4f}")

    return {
        "z_esm_pca": z_esm_pca,
        "mean": z_esm_pca.mean(),
        "std": z_esm_pca.std(),
        "l2_norms": l2_norms_pca,
        "component_means": z_esm_pca.mean(axis=0),
        "component_stds": z_esm_pca.std(axis=0),
        "variance_explained": pca_model.explained_variance_ratio_,
    }


def analyze_stage3_scaled(z_esm_pca, lambda_scale):
    """Analyze scaled PCA contribution (λ * z_esm_pca)."""
    print_header("STAGE 3: Scaled Contribution (λ=%.2f)" % lambda_scale)

    # Apply scaling (exactly as in dynamics.py line 181)
    scaled_contribution = lambda_scale * z_esm_pca

    print(f"Scaled PCA (λ * z_esm_pca):")
    print(f"  λ (lambda):  {lambda_scale}")
    print(f"  Mean:        {scaled_contribution.mean():.4f}")
    print(f"  Std:         {scaled_contribution.std():.4f}")
    print(f"  Min:         {scaled_contribution.min():.4f}")
    print(f"  Max:         {scaled_contribution.max():.4f}")

    print(f"\nThis is what gets ADDED to h_residues at each timestep!")

    # L2 norm
    l2_norms_scaled = np.linalg.norm(scaled_contribution, axis=1)
    print(f"\nL2 Norms:")
    print(f"  Mean: {l2_norms_scaled.mean():.4f}")
    print(f"  Std:  {l2_norms_scaled.std():.4f}")

    return {
        "scaled_contribution": scaled_contribution,
        "mean": scaled_contribution.mean(),
        "std": scaled_contribution.std(),
        "l2_norms": l2_norms_scaled,
    }


def analyze_stage4_effect(scaled_contribution):
    """Analyze effect magnitude relative to h_residues."""
    print_header("STAGE 4: Effect Magnitude Analysis")

    # Simulate typical h_residues values
    # Based on typical EGNN feature scales (zero mean, unit-ish variance)
    n_samples, n_features = scaled_contribution.shape
    z_pocket_simulated = np.random.randn(n_samples, n_features) * 0.5

    print("Simulated h_residues (z_pocket):")
    print(f"  Mean:        {z_pocket_simulated.mean():.4f} (should be ~0)")
    print(f"  Std:         {z_pocket_simulated.std():.4f} (typical EGNN scale)")

    # Compute relative contribution
    ratio = scaled_contribution.std() / z_pocket_simulated.std()
    print(f"\nRelative Contribution:")
    print(f"  PCA std / h_residues std: {ratio:.1%}")

    # Interpretation
    print(f"\n  Interpretation:")
    if ratio < 0.1:
        print(f"    - PCA contribution is WEAK (< 10%) - may be negligible")
    elif ratio < 0.3:
        print(f"    - PCA contribution is MODERATE (~{ratio:.0%}) - balanced")
    elif ratio < 0.5:
        print(f"    - PCA contribution is STRONG (~{ratio:.0%}) - meaningful signal")
    else:
        print(f"    - PCA contribution is VERY STRONG (> 50%) - may overwhelm pretrained features")

    # Per-component contribution
    print(f"\nPer-Component Statistics:")
    comp_means = scaled_contribution.mean(axis=0)
    comp_stds = scaled_contribution.std(axis=0)

    print(f"{'Component':>12} {'Mean':>8} {'Std':>8} {'|Mean/Std|':>12}")
    print("-" * 45)
    for i in range(min(10, len(comp_means))):
        ratio_i = abs(comp_means[i]) / (comp_stds[i] + 1e-8)
        print(f"  {i+1:10d} {comp_means[i]:8.4f} {comp_stds[i]:8.4f} {ratio_i:12.4f}")

    if len(comp_means) > 10:
        print(f"  ... ({len(comp_means) - 10} more components)")

    return {
        "z_pocket_simulated": z_pocket_simulated,
        "relative_ratio": ratio,
    }


def create_visualizations(stage1, stage2, stage3, stage4, output_path, pca_model):
    """Create comprehensive multi-panel visualization."""
    print_header("VISUALIZATION")

    print("Generating plots...")

    fig = plt.figure(figsize=(18, 12))

    # Panel 1: ESM-C Embedding Distribution (960D)
    ax1 = plt.subplot(3, 3, 1)
    embeddings_flat = stage1["embeddings"].flatten()
    ax1.hist(embeddings_flat, bins=100, alpha=0.7, edgecolor='black')
    ax1.set_xlabel("Value")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Panel 1: ESM-C Embeddings (960D, flattened)")
    ax1.axvline(stage1["mean"], color='r', linestyle='--', label=f'Mean: {stage1["mean"]:.3f}')
    ax1.legend()

    # Panel 2: Per-dimension variance (first 100 dims)
    ax2 = plt.subplot(3, 3, 2)
    ax2.bar(range(min(100, len(stage1["dim_variances"]))),
            stage1["dim_variances"][:100], alpha=0.7)
    ax2.set_xlabel("Dimension")
    ax2.set_ylabel("Variance")
    ax2.set_title("Panel 2: ESM-C Per-Dimension Variance (first 100)")

    # Panel 3: PCA Component Distribution (box plot)
    ax3 = plt.subplot(3, 3, 3)
    z_esm_pca = stage2["z_esm_pca"]
    n_comp = min(32, z_esm_pca.shape[1])
    ax3.boxplot([z_esm_pca[:, i] for i in range(n_comp)], showfliers=False)
    ax3.set_xlabel("Component")
    ax3.set_ylabel("Value")
    ax3.set_title(f"Panel 3: PCA Components (960D → {z_esm_pca.shape[1]}D)")
    ax3.set_xticklabels([str(i+1) if (i+1) % 5 == 0 or i == 0 else ''
                         for i in range(n_comp)], rotation=0)

    # Panel 4: Scaled Contribution Distribution
    ax4 = plt.subplot(3, 3, 4)
    scaled_flat = stage3["scaled_contribution"].flatten()
    n, bins, patches = ax4.hist(scaled_flat, bins=100, alpha=0.7,
                                  edgecolor='black', density=True, label='Data')
    # Overlay Gaussian
    mu, sigma = scaled_flat.mean(), scaled_flat.std()
    x = np.linspace(scaled_flat.min(), scaled_flat.max(), 100)
    ax4.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2,
             label=f'Gaussian (μ={mu:.3f}, σ={sigma:.3f})')
    ax4.set_xlabel("Value")
    ax4.set_ylabel("Density")
    ax4.set_title(f"Panel 4: Scaled PCA Contribution (λ={stage3['scaled_contribution'].mean() / z_esm_pca.mean():.2f})")
    ax4.legend()

    # Panel 5: Effect Magnitude Comparison
    ax5 = plt.subplot(3, 3, 5)
    z_pocket = stage4["z_pocket_simulated"]
    scaled = stage3["scaled_contribution"]

    # Per-component std comparison
    pocket_stds = z_pocket.std(axis=0)
    scaled_stds = scaled.std(axis=0)

    x_pos = np.arange(min(20, len(pocket_stds)))
    width = 0.35
    ax5.bar(x_pos - width/2, pocket_stds[:len(x_pos)], width,
            label='h_residues (simulated)', alpha=0.7)
    ax5.bar(x_pos + width/2, scaled_stds[:len(x_pos)], width,
            label='PCA contribution', alpha=0.7)
    ax5.set_xlabel("Component")
    ax5.set_ylabel("Standard Deviation")
    ax5.set_title("Panel 5: Magnitude Comparison (first 20 components)")
    ax5.legend()
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels([str(i+1) if (i+1) % 5 == 0 or i == 0 else ''
                         for i in range(len(x_pos))])

    # Panel 6: Correlation Heatmap
    ax6 = plt.subplot(3, 3, 6)
    corr_matrix = np.corrcoef(z_esm_pca.T)
    im = ax6.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
    ax6.set_xlabel("Component")
    ax6.set_ylabel("Component")
    ax6.set_title("Panel 6: PCA Component Correlations")
    plt.colorbar(im, ax=ax6)

    # Panel 7: Cumulative Variance Explained
    ax7 = plt.subplot(3, 3, 7)
    cumsum_var = pca_model.explained_variance_ratio_.cumsum() * 100
    ax7.plot(range(1, len(cumsum_var) + 1), cumsum_var, 'b-', linewidth=2)
    ax7.axhline(87.5, color='r', linestyle='--', label='87.5% (target)')
    ax7.set_xlabel("Number of Components")
    ax7.set_ylabel("Cumulative Variance Explained (%)")
    ax7.set_title("Panel 7: Variance Retention")
    ax7.grid(True, alpha=0.3)
    ax7.legend()

    # Panel 8: L2 Norm Distributions
    ax8 = plt.subplot(3, 3, 8)
    ax8.hist(stage1["l2_norms"], bins=50, alpha=0.5, label='ESM-C (960D)', edgecolor='black')
    ax8.hist(stage2["l2_norms"], bins=50, alpha=0.5, label='PCA (32D)', edgecolor='black')
    ax8.hist(stage3["l2_norms"], bins=50, alpha=0.5, label='Scaled', edgecolor='black')
    ax8.set_xlabel("L2 Norm")
    ax8.set_ylabel("Frequency")
    ax8.set_title("Panel 8: L2 Norm Distributions")
    ax8.legend()

    # Panel 9: Per-Component Variance Explained
    ax9 = plt.subplot(3, 3, 9)
    var_explained = stage2["variance_explained"] * 100
    ax9.bar(range(1, len(var_explained) + 1), var_explained, alpha=0.7)
    ax9.set_xlabel("Component")
    ax9.set_ylabel("Variance Explained (%)")
    ax9.set_title("Panel 9: Per-Component Variance")
    ax9.set_xlim(0, min(40, len(var_explained) + 1))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print_success(f"Saved figure: {output_path}")

    return fig


def print_summary(stage1, stage2, stage3, stage4, lambda_scale):
    """Print final summary."""
    print_header("SUMMARY")

    print("Key Findings:")
    print(f"1. PCA effectively compresses 960D → {stage2['z_esm_pca'].shape[1]}D " +
          f"({stage2['variance_explained'].sum() * 100:.1f}% variance retained)")
    print(f"2. Scaled contribution (λ={lambda_scale}) adds " +
          f"~{stage4['relative_ratio']:.1%} relative to h_residues magnitude")
    print(f"3. Distribution skewness: {stage1['skewness']:.3f} " +
          f"(0 = symmetric, Gaussian-like)")
    print(f"4. Components are {'decorrelated' if np.abs(np.corrcoef(stage2['z_esm_pca'].T)).mean() < 0.1 else 'partially correlated'} " +
          f"(PCA ensures orthogonality)")

    # Recommendation
    print(f"\nRecommendation:")
    ratio = stage4['relative_ratio']
    if 0.15 <= ratio <= 0.35:
        print(f"✓ Current setup ({stage2['z_esm_pca'].shape[1]}D, λ={lambda_scale}) provides BALANCED conditioning")
        print(f"  - Strong enough to influence generation (~{ratio:.0%})")
        print(f"  - Weak enough to preserve pretrained spatial chemistry")
    elif ratio < 0.15:
        print(f"⚠ Current setup may provide WEAK conditioning (~{ratio:.0%})")
        print(f"  - Consider increasing λ or using more PCA components")
    else:
        print(f"⚠ Current setup may provide STRONG conditioning (~{ratio:.0%})")
        print(f"  - Consider decreasing λ or using fewer PCA components")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze PCA effect on h_residues in diffusion model"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/real_testing_dataset_10_tests/train_esmc.npz",
        help="Path to ESM-C embeddings npz file",
    )
    parser.add_argument(
        "--pca-model",
        type=str,
        default="thesis_work/experiments/day6_pca_projection/pca_model_32d.pkl",
        help="Path to PCA model pickle file",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=1000,
        help="Number of samples to analyze (None = all)",
    )
    parser.add_argument(
        "--lambda-scale",
        type=float,
        default=0.1,
        help="PCA lambda scaling factor (from dynamics.py)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="thesis_work/experiments/day6_pca_projection",
        help="Output directory for plots",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip matplotlib visualizations",
    )

    args = parser.parse_args()

    # Print analysis header
    print_header("PCA EFFECT DISTRIBUTION ANALYSIS")

    # Load data
    embeddings, names = load_esmc_embeddings(args.data_path, args.n_samples)
    pca_model = load_pca_model(args.pca_model)

    # Run analysis stages
    stage1 = analyze_stage1_esmc(embeddings)
    stage2 = analyze_stage2_pca(embeddings, pca_model)
    stage3 = analyze_stage3_scaled(stage2["z_esm_pca"], args.lambda_scale)
    stage4 = analyze_stage4_effect(stage3["scaled_contribution"])

    # Create visualizations
    if not args.no_plot:
        output_path = Path(args.output_dir) / "pca_effect_analysis.png"
        create_visualizations(stage1, stage2, stage3, stage4, output_path, pca_model)

    # Print summary
    print_summary(stage1, stage2, stage3, stage4, args.lambda_scale)

    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
