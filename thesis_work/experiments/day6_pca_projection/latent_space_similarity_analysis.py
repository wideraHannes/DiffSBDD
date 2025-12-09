"""
Rigorous Latent Space Similarity Analysis

This script performs the key analysis to determine if ESM-C conditioning is adding
signal or noise to DiffSBDD's residue encoder latent space.

KEY QUESTION:
    Does ESM_PCA_Dim_0 correlate with DiffSBDD_Feat_Dim_0?
    If YES: Steering will work beautifully (aligned spaces)
    If NO: Just adding noise (orthogonal spaces)

Analysis Methods:
    1. Dimension-wise Pearson correlation
    2. Dimension-wise Spearman correlation
    3. Dimension-wise cosine similarity
    4. Global CKA (Centered Kernel Alignment)
    5. Hybrid variant comparison (h_residues + λ * z_esm_pca)

Usage:
    python latent_space_similarity_analysis.py
"""

import torch
import numpy as np
import pickle
import argparse
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.spatial.distance import cosine
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parents[3]))

from lightning_modules import LigandPocketDDPM
from dataset import ProcessedLigandPocketDataset


def print_header(text):
    """Print formatted section header."""
    print("\n" + "=" * 80)
    print(text)
    print("=" * 80 + "\n")


def print_success(text):
    """Print success message."""
    print(f"✓ {text}")


def extract_latent_representations(model, dataloader, pca_model, device="cpu", max_batches=100):
    """
    Extract h_residues (DiffSBDD encoder) and z_esm_pca (ESM-C PCA).

    Returns:
        h_residues: [N, joint_nf] - DiffSBDD encoded residue features
        z_esm_pca: [N, joint_nf] - PCA-projected ESM-C embeddings
    """
    print_header("EXTRACTING LATENT REPRESENTATIONS")

    model.to(device)
    model.eval()

    h_residues_list = []
    z_esm_pca_list = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Extracting")):
            if batch_idx >= max_batches:
                break

            try:
                # Get pocket data
                pocket_coords = batch["pocket_coords"].to(device).float()
                pocket_one_hot = batch["pocket_one_hot"].to(device).float()
                pocket_mask = batch["pocket_mask"].to(device).long()
                pocket_emb = batch.get("pocket_emb", None)

                if pocket_emb is None:
                    print("Warning: No pocket_emb in batch")
                    continue

                pocket_emb = pocket_emb.to(device).float()

                # 1. Extract h_residues (DiffSBDD encoder baseline)
                xh_residues = torch.cat([pocket_coords, pocket_one_hot], dim=1)
                h_residues_raw = xh_residues[:, 3:]  # Skip coords

                # Encode through residue_encoder
                h_residues_encoded = model.ddpm.dynamics.residue_encoder(h_residues_raw)
                h_residues_list.append(h_residues_encoded.cpu().numpy())

                # 2. Extract PCA-projected ESM-C embeddings
                z_esm_pca = pca_model.transform(pocket_emb.cpu().numpy())
                z_esm_pca = torch.from_numpy(z_esm_pca).float().to(device)

                # Expand to per-residue using mask
                z_esm_pca_expanded = z_esm_pca[pocket_mask]
                z_esm_pca_list.append(z_esm_pca_expanded.cpu().numpy())

            except Exception as e:
                print(f"Warning: Error processing batch {batch_idx}: {e}")
                continue

    if not h_residues_list:
        raise ValueError("No data extracted!")

    h_residues = np.concatenate(h_residues_list, axis=0)
    z_esm_pca = np.concatenate(z_esm_pca_list, axis=0)

    print_success(f"Extracted {len(h_residues)} residue samples")
    print(f"  h_residues shape: {h_residues.shape}")
    print(f"  z_esm_pca shape: {z_esm_pca.shape}")

    return h_residues, z_esm_pca


def compute_dimension_wise_correlations(h_residues, z_esm_pca):
    """
    Compute dimension-wise correlations between h_residues and z_esm_pca.

    KEY METRIC: Are corresponding dimensions correlated?
    - High correlation → aligned spaces → steering works
    - Low correlation → orthogonal spaces → adding noise
    """
    print_header("DIMENSION-WISE CORRELATION ANALYSIS")

    n_dims = h_residues.shape[1]

    pearson_corrs = []
    spearman_corrs = []
    cosine_sims = []

    print(f"{'Dim':>4} {'Pearson r':>12} {'p-value':>10} {'Spearman ρ':>13} {'Cosine Sim':>12}")
    print("-" * 65)

    for dim in range(n_dims):
        # Pearson correlation (linear relationship)
        pearson_r, pearson_p = stats.pearsonr(h_residues[:, dim], z_esm_pca[:, dim])
        pearson_corrs.append(pearson_r)

        # Spearman correlation (monotonic relationship)
        spearman_r, _ = stats.spearmanr(h_residues[:, dim], z_esm_pca[:, dim])
        spearman_corrs.append(spearman_r)

        # Cosine similarity (directional alignment)
        # Compute 1 - cosine_distance to get cosine similarity
        cos_sim = 1 - cosine(h_residues[:, dim], z_esm_pca[:, dim])
        cosine_sims.append(cos_sim)

        # Print significant correlations
        if abs(pearson_r) > 0.1 or abs(spearman_r) > 0.1:
            sig = "**" if pearson_p < 0.01 else "*" if pearson_p < 0.05 else ""
            print(f"{dim:4d} {pearson_r:12.4f}{sig:2s} {pearson_p:10.2e} {spearman_r:13.4f} {cos_sim:12.4f}")

    pearson_corrs = np.array(pearson_corrs)
    spearman_corrs = np.array(spearman_corrs)
    cosine_sims = np.array(cosine_sims)

    # Summary statistics
    print("\n" + "=" * 65)
    print("SUMMARY STATISTICS:")
    print(f"  Pearson correlation:")
    print(f"    Mean:   {pearson_corrs.mean():8.4f}")
    print(f"    Median: {np.median(pearson_corrs):8.4f}")
    print(f"    Std:    {pearson_corrs.std():8.4f}")
    print(f"    Max:    {pearson_corrs.max():8.4f}")
    print(f"    Min:    {pearson_corrs.min():8.4f}")
    print(f"    |r| > 0.1: {(np.abs(pearson_corrs) > 0.1).sum()}/{n_dims} dims")
    print(f"    |r| > 0.3: {(np.abs(pearson_corrs) > 0.3).sum()}/{n_dims} dims")
    print(f"    |r| > 0.5: {(np.abs(pearson_corrs) > 0.5).sum()}/{n_dims} dims")

    print(f"\n  Spearman correlation:")
    print(f"    Mean:   {spearman_corrs.mean():8.4f}")
    print(f"    Median: {np.median(spearman_corrs):8.4f}")
    print(f"    |ρ| > 0.1: {(np.abs(spearman_corrs) > 0.1).sum()}/{n_dims} dims")

    print(f"\n  Cosine similarity:")
    print(f"    Mean:   {cosine_sims.mean():8.4f}")
    print(f"    Median: {np.median(cosine_sims):8.4f}")

    # VERDICT
    print("\n" + "=" * 65)
    mean_abs_corr = np.abs(pearson_corrs).mean()

    if mean_abs_corr > 0.3:
        verdict = "✓ STRONG ALIGNMENT - ESM-C steering will work well!"
    elif mean_abs_corr > 0.15:
        verdict = "○ MODERATE ALIGNMENT - ESM-C steering may help"
    elif mean_abs_corr > 0.05:
        verdict = "⚠ WEAK ALIGNMENT - ESM-C steering adds weak signal"
    else:
        verdict = "✗ NO ALIGNMENT - ESM-C is adding NOISE, not signal!"

    print(f"VERDICT: {verdict}")
    print("=" * 65)

    return {
        'pearson': pearson_corrs,
        'spearman': spearman_corrs,
        'cosine': cosine_sims,
        'verdict': verdict
    }


def compute_cka(X, Y):
    """
    Compute Centered Kernel Alignment (CKA) between two representation matrices.

    CKA measures similarity between entire representation spaces (not just dimensions).
    Range: [0, 1], where 1 = identical representations
    """
    # Center the Gram matrices
    def center_gram(K):
        n = K.shape[0]
        H = np.eye(n) - np.ones((n, n)) / n
        return H @ K @ H

    # Linear kernel (Gram matrix)
    K_X = X @ X.T
    K_Y = Y @ Y.T

    # Center
    K_X = center_gram(K_X)
    K_Y = center_gram(K_Y)

    # CKA formula
    numerator = np.trace(K_X @ K_Y)
    denominator = np.sqrt(np.trace(K_X @ K_X) * np.trace(K_Y @ K_Y))

    cka = numerator / (denominator + 1e-10)
    return cka


def compute_global_similarity(h_residues, z_esm_pca):
    """Compute global similarity metrics between representation spaces."""
    print_header("GLOBAL SIMILARITY METRICS")

    # Subsample for computational efficiency
    n_samples = min(5000, len(h_residues))
    indices = np.random.choice(len(h_residues), n_samples, replace=False)
    h_sub = h_residues[indices]
    z_sub = z_esm_pca[indices]

    # 1. CKA (Centered Kernel Alignment)
    print("Computing CKA...")
    cka_score = compute_cka(h_sub, z_sub)
    print(f"  CKA: {cka_score:.4f}")

    # 2. Full correlation matrix (not dimension-wise)
    print("Computing cross-correlation matrix...")
    # Normalize each dimension
    h_norm = (h_sub - h_sub.mean(axis=0)) / (h_sub.std(axis=0) + 1e-8)
    z_norm = (z_sub - z_sub.mean(axis=0)) / (z_sub.std(axis=0) + 1e-8)

    # Cross-correlation: [n_dims_h, n_dims_z]
    cross_corr = (h_norm.T @ z_norm) / len(h_norm)

    print(f"  Cross-correlation shape: {cross_corr.shape}")
    print(f"  Max abs correlation: {np.abs(cross_corr).max():.4f}")
    print(f"  Mean abs correlation: {np.abs(cross_corr).mean():.4f}")
    print(f"  Diagonal mean: {np.diag(cross_corr).mean():.4f}")

    # 3. Subspace overlap
    print("Computing subspace overlap...")
    # Use SVD to find principal subspaces
    _, s_h, _ = np.linalg.svd(h_sub, full_matrices=False)
    _, s_z, _ = np.linalg.svd(z_sub, full_matrices=False)

    # Explained variance
    var_h = (s_h**2).cumsum() / (s_h**2).sum()
    var_z = (s_z**2).cumsum() / (s_z**2).sum()

    print(f"  h_residues: 90% variance in {(var_h < 0.9).sum()} dims")
    print(f"  z_esm_pca:  90% variance in {(var_z < 0.9).sum()} dims")

    return {
        'cka': cka_score,
        'cross_corr': cross_corr,
        'var_h': var_h,
        'var_z': var_z
    }


def compare_hybrid_variants(h_residues, z_esm_pca, lambda_values):
    """
    Compare hybrid representations: h_hybrid = h_residues + λ * z_esm_pca

    Analyzes how different lambda values affect the combined representation.
    """
    print_header("HYBRID VARIANT ANALYSIS")

    results = []

    # Baseline stats
    h_l2 = np.linalg.norm(h_residues, axis=1).mean()
    z_l2 = np.linalg.norm(z_esm_pca, axis=1).mean()

    print(f"Baseline L2 norms:")
    print(f"  h_residues: {h_l2:.4f}")
    print(f"  z_esm_pca:  {z_l2:.4f}")
    print()

    print(f"{'Lambda':>8} {'Hybrid L2':>12} {'Δ from h':>12} {'Rel. Change':>13} {'h vs z_pca corr':>18}")
    print("-" * 75)

    for lam in lambda_values:
        # Create hybrid
        h_hybrid = h_residues + lam * z_esm_pca

        # Statistics
        hybrid_l2 = np.linalg.norm(h_hybrid, axis=1).mean()
        delta_l2 = hybrid_l2 - h_l2
        rel_change = delta_l2 / h_l2

        # Correlation between h_residues and scaled z_esm_pca in hybrid
        # (measures how much they're aligned vs orthogonal)
        h_flat = h_residues.flatten()
        z_scaled_flat = (lam * z_esm_pca).flatten()
        corr, _ = stats.pearsonr(h_flat, z_scaled_flat)

        print(f"{lam:8.4f} {hybrid_l2:12.4f} {delta_l2:12.4f} {rel_change:12.1%} {corr:18.4f}")

        results.append({
            'lambda': lam,
            'hybrid_l2': hybrid_l2,
            'delta_l2': delta_l2,
            'rel_change': rel_change,
            'correlation': corr
        })

    return results


def create_visualizations(h_residues, z_esm_pca, dim_corrs, global_metrics, hybrid_results, output_dir):
    """Create comprehensive visualization plots."""
    print_header("CREATING VISUALIZATIONS")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ============================================================================
    # FIGURE 1: Dimension-wise Correlation Analysis
    # ============================================================================
    fig1 = plt.figure(figsize=(20, 12))

    # Panel 1: Pearson correlation per dimension
    ax1 = plt.subplot(3, 3, 1)
    dims = np.arange(len(dim_corrs['pearson']))
    ax1.bar(dims, dim_corrs['pearson'], alpha=0.7, edgecolor='black')
    ax1.axhline(0, color='k', linestyle='-', linewidth=0.5)
    ax1.axhline(0.3, color='g', linestyle='--', alpha=0.5, label='Strong (±0.3)')
    ax1.axhline(-0.3, color='g', linestyle='--', alpha=0.5)
    ax1.axhline(0.15, color='orange', linestyle='--', alpha=0.5, label='Moderate (±0.15)')
    ax1.axhline(-0.15, color='orange', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Dimension')
    ax1.set_ylabel('Pearson r')
    ax1.set_title('Dimension-wise Pearson Correlation\n(h_residues vs z_esm_pca)')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Spearman correlation per dimension
    ax2 = plt.subplot(3, 3, 2)
    ax2.bar(dims, dim_corrs['spearman'], alpha=0.7, edgecolor='black', color='orange')
    ax2.axhline(0, color='k', linestyle='-', linewidth=0.5)
    ax2.axhline(0.3, color='g', linestyle='--', alpha=0.5)
    ax2.axhline(-0.3, color='g', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Dimension')
    ax2.set_ylabel('Spearman ρ')
    ax2.set_title('Dimension-wise Spearman Correlation\n(Monotonic relationship)')
    ax2.grid(True, alpha=0.3)

    # Panel 3: Cosine similarity per dimension
    ax3 = plt.subplot(3, 3, 3)
    ax3.bar(dims, dim_corrs['cosine'], alpha=0.7, edgecolor='black', color='green')
    ax3.axhline(0, color='k', linestyle='-', linewidth=0.5)
    ax3.set_xlabel('Dimension')
    ax3.set_ylabel('Cosine Similarity')
    ax3.set_title('Dimension-wise Cosine Similarity\n(Directional alignment)')
    ax3.grid(True, alpha=0.3)

    # Panel 4: Histogram of Pearson correlations
    ax4 = plt.subplot(3, 3, 4)
    ax4.hist(dim_corrs['pearson'], bins=30, alpha=0.7, edgecolor='black')
    ax4.axvline(0, color='r', linestyle='--', alpha=0.5)
    ax4.axvline(dim_corrs['pearson'].mean(), color='g', linestyle='--',
                linewidth=2, label=f"Mean: {dim_corrs['pearson'].mean():.3f}")
    ax4.set_xlabel('Pearson r')
    ax4.set_ylabel('Count')
    ax4.set_title('Distribution of Dimension-wise Correlations')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Panel 5: Scatter plot of correlations
    ax5 = plt.subplot(3, 3, 5)
    ax5.scatter(dim_corrs['pearson'], dim_corrs['spearman'], alpha=0.6)
    ax5.plot([-1, 1], [-1, 1], 'r--', alpha=0.5, label='y=x')
    ax5.set_xlabel('Pearson r')
    ax5.set_ylabel('Spearman ρ')
    ax5.set_title('Pearson vs Spearman Correlation')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.axis('equal')

    # Panel 6: Cumulative distribution
    ax6 = plt.subplot(3, 3, 6)
    sorted_corr = np.sort(np.abs(dim_corrs['pearson']))
    cumsum = np.arange(1, len(sorted_corr) + 1) / len(sorted_corr)
    ax6.plot(sorted_corr, cumsum, linewidth=2)
    ax6.axvline(0.1, color='orange', linestyle='--', alpha=0.5, label='|r| = 0.1')
    ax6.axvline(0.3, color='g', linestyle='--', alpha=0.5, label='|r| = 0.3')
    ax6.set_xlabel('|Pearson r|')
    ax6.set_ylabel('Cumulative Fraction of Dimensions')
    ax6.set_title('CDF of Absolute Correlations')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    # Panel 7: Cross-correlation heatmap (sample)
    ax7 = plt.subplot(3, 3, 7)
    cross_corr = global_metrics['cross_corr']
    im = ax7.imshow(cross_corr, cmap='RdBu_r', vmin=-0.5, vmax=0.5, aspect='auto')
    ax7.set_xlabel('z_esm_pca dimensions')
    ax7.set_ylabel('h_residues dimensions')
    ax7.set_title('Cross-Correlation Matrix\n(h_residues × z_esm_pca)')
    plt.colorbar(im, ax=ax7)

    # Panel 8: Singular value spectrum
    ax8 = plt.subplot(3, 3, 8)
    ax8.plot(global_metrics['var_h'], label='h_residues', linewidth=2)
    ax8.plot(global_metrics['var_z'], label='z_esm_pca', linewidth=2)
    ax8.axhline(0.9, color='k', linestyle='--', alpha=0.5, label='90% variance')
    ax8.set_xlabel('Number of Components')
    ax8.set_ylabel('Cumulative Variance Explained')
    ax8.set_title('Principal Subspace Analysis')
    ax8.legend()
    ax8.grid(True, alpha=0.3)

    # Panel 9: Summary statistics
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    summary_text = f"""
SIMILARITY ANALYSIS SUMMARY

Dimension-wise Correlations:
  Mean |Pearson r|: {np.abs(dim_corrs['pearson']).mean():.4f}
  Median Pearson r: {np.median(dim_corrs['pearson']):.4f}
  Dims with |r| > 0.1: {(np.abs(dim_corrs['pearson']) > 0.1).sum()}/{len(dims)}
  Dims with |r| > 0.3: {(np.abs(dim_corrs['pearson']) > 0.3).sum()}/{len(dims)}

Global Metrics:
  CKA score: {global_metrics['cka']:.4f}
  Max cross-correlation: {np.abs(global_metrics['cross_corr']).max():.4f}

VERDICT:
{dim_corrs['verdict']}
    """
    ax9.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    fig1_path = output_dir / 'dimension_wise_correlation_analysis.png'
    plt.savefig(fig1_path, dpi=150, bbox_inches='tight')
    print_success(f"Saved: {fig1_path}")
    plt.close()

    # ============================================================================
    # FIGURE 2: Hybrid Variant Comparison
    # ============================================================================
    fig2 = plt.figure(figsize=(16, 10))

    lambdas = [r['lambda'] for r in hybrid_results]
    hybrid_l2s = [r['hybrid_l2'] for r in hybrid_results]
    rel_changes = [r['rel_change'] for r in hybrid_results]
    correlations = [r['correlation'] for r in hybrid_results]

    # Panel 1: L2 norm vs lambda
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(lambdas, hybrid_l2s, 'bo-', linewidth=2, markersize=8)
    ax1.axhline(np.linalg.norm(h_residues, axis=1).mean(), color='r',
                linestyle='--', label='Baseline (h_residues only)')
    ax1.set_xlabel('Lambda (λ)')
    ax1.set_ylabel('Mean L2 Norm')
    ax1.set_title('Hybrid Representation Magnitude')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')

    # Panel 2: Relative change vs lambda
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(lambdas, [r*100 for r in rel_changes], 'go-', linewidth=2, markersize=8)
    ax2.axhline(0, color='k', linestyle='-', linewidth=0.5)
    ax2.axhline(20, color='orange', linestyle='--', alpha=0.5, label='20% change')
    ax2.axhline(50, color='r', linestyle='--', alpha=0.5, label='50% change')
    ax2.set_xlabel('Lambda (λ)')
    ax2.set_ylabel('Relative Change (%)')
    ax2.set_title('Impact on Representation Magnitude')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')

    # Panel 3: Correlation vs lambda
    ax3 = plt.subplot(2, 3, 3)
    ax3.plot(lambdas, correlations, 'ro-', linewidth=2, markersize=8)
    ax3.axhline(0, color='k', linestyle='-', linewidth=0.5)
    ax3.set_xlabel('Lambda (λ)')
    ax3.set_ylabel('Correlation (h_residues vs λ·z_esm_pca)')
    ax3.set_title('Component Alignment in Hybrid')
    ax3.grid(True, alpha=0.3)
    ax3.set_xscale('log')

    # Panels 4-6: Distribution comparisons for selected lambdas
    selected_lambdas = [0.1, 0.5, 1.0]
    for idx, lam in enumerate(selected_lambdas):
        ax = plt.subplot(2, 3, 4 + idx)

        # Create hybrid
        h_hybrid = h_residues + lam * z_esm_pca

        # Sample for plotting
        h_sample = np.random.choice(h_residues.flatten(), 10000, replace=False)
        hybrid_sample = np.random.choice(h_hybrid.flatten(), 10000, replace=False)

        ax.hist(h_sample, bins=50, alpha=0.5, label='h_residues', density=True, edgecolor='black')
        ax.hist(hybrid_sample, bins=50, alpha=0.5, label=f'h + {lam}·z_pca',
                density=True, edgecolor='black', color='orange')
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.set_title(f'λ = {lam}: Distribution Comparison')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig2_path = output_dir / 'hybrid_variant_comparison.png'
    plt.savefig(fig2_path, dpi=150, bbox_inches='tight')
    print_success(f"Saved: {fig2_path}")
    plt.close()

    return fig1_path, fig2_path


def main():
    parser = argparse.ArgumentParser(
        description='Rigorous latent space similarity analysis'
    )
    parser.add_argument('--checkpoint', type=str,
                        default='checkpoints/crossdocked_fullatom_cond.ckpt',
                        help='Path to model checkpoint')
    parser.add_argument('--test_npz', type=str,
                        default='data/real_testing_dataset_10_tests/test.npz',
                        help='Path to test dataset')
    parser.add_argument('--test_esmc', type=str,
                        default='data/real_testing_dataset_10_tests/test_esmc.npz',
                        help='Path to test ESM-C embeddings')
    parser.add_argument('--pca_model', type=str,
                        default='thesis_work/experiments/day6_pca_projection/pca_model_32d.pkl',
                        help='Path to PCA model')
    parser.add_argument('--lambda_values', type=float, nargs='+',
                        default=[0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0],
                        help='Lambda values for hybrid comparison')
    parser.add_argument('--output_dir', type=str,
                        default='thesis_work/experiments/day6_pca_projection',
                        help='Output directory')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device (cpu/cuda)')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Batch size')
    parser.add_argument('--max_batches', type=int, default=100,
                        help='Max batches to process')

    args = parser.parse_args()

    print_header("LATENT SPACE SIMILARITY ANALYSIS")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Test npz: {args.test_npz}")
    print(f"Test ESM-C: {args.test_esmc}")
    print(f"PCA model: {args.pca_model}")

    # Load model
    print_header("LOADING MODEL")
    model = LigandPocketDDPM.load_from_checkpoint(
        args.checkpoint, map_location=args.device
    )
    model.eval()
    print_success(f"Loaded checkpoint")

    # Load PCA
    print_header("LOADING PCA MODEL")
    with open(args.pca_model, 'rb') as f:
        pca_model = pickle.load(f)
    print_success(f"Loaded PCA: 960D → {pca_model.n_components_}D")
    print_success(f"Variance explained: {pca_model.explained_variance_ratio_.sum()*100:.2f}%")

    # Load dataset
    print_header("LOADING DATASET")
    dataset = ProcessedLigandPocketDataset(args.test_npz, esmc_path=args.test_esmc)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=dataset.collate_fn
    )
    print_success(f"Loaded {len(dataset)} test samples")

    # Extract latent representations
    h_residues, z_esm_pca = extract_latent_representations(
        model, dataloader, pca_model, device=args.device, max_batches=args.max_batches
    )

    # 1. Dimension-wise correlation analysis
    dim_corrs = compute_dimension_wise_correlations(h_residues, z_esm_pca)

    # 2. Global similarity metrics
    global_metrics = compute_global_similarity(h_residues, z_esm_pca)

    # 3. Hybrid variant comparison
    hybrid_results = compare_hybrid_variants(h_residues, z_esm_pca, args.lambda_values)

    # 4. Create visualizations
    fig1_path, fig2_path = create_visualizations(
        h_residues, z_esm_pca, dim_corrs, global_metrics, hybrid_results, args.output_dir
    )

    # Save results
    results_path = Path(args.output_dir) / 'latent_space_similarity_results.pkl'
    with open(results_path, 'wb') as f:
        pickle.dump({
            'dim_correlations': dim_corrs,
            'global_metrics': global_metrics,
            'hybrid_results': hybrid_results,
            'h_residues_shape': h_residues.shape,
            'z_esm_pca_shape': z_esm_pca.shape,
        }, f)
    print_success(f"Saved results: {results_path}")

    # Final summary
    print_header("FINAL SUMMARY")
    print("\nKEY QUESTION: Does ESM_PCA_Dim_i correlate with DiffSBDD_Feat_Dim_i?")
    print(f"\nANSWER: {dim_corrs['verdict']}\n")
    print(f"Mean |Pearson r|: {np.abs(dim_corrs['pearson']).mean():.4f}")
    print(f"CKA score: {global_metrics['cka']:.4f}")
    print(f"\nDimensions with significant correlation (|r| > 0.1): "
          f"{(np.abs(dim_corrs['pearson']) > 0.1).sum()}/{len(dim_corrs['pearson'])}")

    print("\n" + "="*80)
    print("Analysis complete! Check the output plots:")
    print(f"  1. {fig1_path}")
    print(f"  2. {fig2_path}")
    print("="*80)


if __name__ == '__main__':
    main()
