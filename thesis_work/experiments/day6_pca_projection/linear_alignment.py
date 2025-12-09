"""
Linear Alignment of ESM-C to DiffSBDD Feature Space

PROBLEM DIAGNOSIS:
    ESM (z_esm): Dynamic Context - "I am a conserved Valine in a binding pocket"
    DiffSBDD (h_residues): Static Identity - "I am Valine"

    Direct addition (h + z) overwrites identity → model sees garbage

SOLUTION:
    Learn a linear "translator" W such that:
        W · z_esm ≈ h_residues (identity component)

    Then inject:
        h_new = h_diffsbdd + λ · (W · z_esm)

    W aligns the identity component, context comes as useful perturbation.

THESIS OUTCOMES:
    Scenario A: R² > 0.3  → Success! ESM can simulate DiffSBDD features
    Scenario B: R² < 0.1  → Semantic gap confirmed, pivot to global conditioning

Usage:
    python linear_alignment.py
    python linear_alignment.py --extract_fresh  # Re-extract features
"""

import numpy as np
import pickle
import argparse
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parents[3]))

from torch.utils.data import DataLoader
from lightning_modules import LigandPocketDDPM
from dataset import ProcessedLigandPocketDataset
import torch
from tqdm import tqdm


def print_header(text):
    """Print formatted section header."""
    print("\n" + "=" * 80)
    print(text)
    print("=" * 80 + "\n")


def print_success(text):
    """Print success message."""
    print(f"✓ {text}")


def extract_features(model, dataloader, pca_model, device="cpu", max_batches=100):
    """Extract h_residues and z_esm_pca features."""
    print_header("EXTRACTING FEATURES")

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
                    continue

                pocket_emb = pocket_emb.to(device).float()

                # 1. Extract h_residues (DiffSBDD encoder)
                xh_residues = torch.cat([pocket_coords, pocket_one_hot], dim=1)
                h_residues_raw = xh_residues[:, 3:]  # Skip coords
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

    h_residues = np.concatenate(h_residues_list, axis=0)
    z_esm_pca = np.concatenate(z_esm_pca_list, axis=0)

    print_success(f"Extracted {len(h_residues)} residue samples")
    print(f"  h_residues shape: {h_residues.shape}")
    print(f"  z_esm_pca shape: {z_esm_pca.shape}")

    return h_residues, z_esm_pca


def train_linear_translator(X_esm, y_diffsbdd, alpha_values=[0.01, 0.1, 1.0, 10.0], test_size=0.2):
    """
    Train a linear translator from ESM space to DiffSBDD space.

    Args:
        X_esm: [N, D] ESM-C PCA features (source)
        y_diffsbdd: [N, D] DiffSBDD residue features (target)
        alpha_values: Ridge regularization parameters to try
        test_size: Fraction of data for validation

    Returns:
        best_translator: Trained Ridge model
        results: Training metrics
    """
    print_header("TRAINING LINEAR TRANSLATOR")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_esm, y_diffsbdd, test_size=test_size, random_state=42
    )

    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")

    # Try different regularization strengths
    best_alpha = None
    best_score = -np.inf
    best_model = None

    print(f"\n{'Alpha':>10} {'Train R²':>12} {'Test R²':>12} {'Test MSE':>12}")
    print("-" * 50)

    results = []
    for alpha in alpha_values:
        # Train translator
        translator = Ridge(alpha=alpha, random_state=42)
        translator.fit(X_train, y_train)

        # Evaluate
        train_r2 = translator.score(X_train, y_train)
        test_r2 = translator.score(X_test, y_test)

        y_pred = translator.predict(X_test)
        test_mse = mean_squared_error(y_test, y_pred)

        print(f"{alpha:10.3f} {train_r2:12.4f} {test_r2:12.4f} {test_mse:12.4f}")

        results.append({
            'alpha': alpha,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'test_mse': test_mse
        })

        # Track best
        if test_r2 > best_score:
            best_score = test_r2
            best_alpha = alpha
            best_model = translator

    print("\n" + "=" * 50)
    print(f"Best alpha: {best_alpha:.3f}")
    print(f"Best test R²: {best_score:.4f}")
    print("=" * 50)

    return best_model, results, (X_train, X_test, y_train, y_test)


def evaluate_alignment(translator, X_esm, y_diffsbdd):
    """
    Evaluate how well the translator aligns ESM to DiffSBDD space.

    Returns detailed metrics including:
        - Overall R² score
        - Per-dimension R² scores
        - Residual analysis
    """
    print_header("EVALUATING ALIGNMENT QUALITY")

    # Get predictions
    y_pred = translator.predict(X_esm)

    # Overall metrics
    overall_r2 = r2_score(y_diffsbdd, y_pred)
    overall_mse = mean_squared_error(y_diffsbdd, y_pred)

    print(f"Overall Alignment Metrics:")
    print(f"  R² score:  {overall_r2:.4f}")
    print(f"  MSE:       {overall_mse:.4f}")
    print(f"  RMSE:      {np.sqrt(overall_mse):.4f}")

    # Per-dimension R² scores
    print(f"\nPer-Dimension Alignment:")
    n_dims = y_diffsbdd.shape[1]
    dim_r2_scores = []

    print(f"{'Dim':>4} {'R²':>10} {'MSE':>10} {'Pearson r':>12}")
    print("-" * 40)

    for dim in range(n_dims):
        dim_r2 = r2_score(y_diffsbdd[:, dim], y_pred[:, dim])
        dim_mse = mean_squared_error(y_diffsbdd[:, dim], y_pred[:, dim])
        pearson_r, _ = stats.pearsonr(y_diffsbdd[:, dim], y_pred[:, dim])

        dim_r2_scores.append(dim_r2)

        if dim_r2 > 0.1 or dim < 5:  # Show first 5 dims + significant ones
            print(f"{dim:4d} {dim_r2:10.4f} {dim_mse:10.4f} {pearson_r:12.4f}")

    dim_r2_scores = np.array(dim_r2_scores)

    print(f"\nPer-Dimension R² Statistics:")
    print(f"  Mean:   {dim_r2_scores.mean():.4f}")
    print(f"  Median: {np.median(dim_r2_scores):.4f}")
    print(f"  Std:    {dim_r2_scores.std():.4f}")
    print(f"  Min:    {dim_r2_scores.min():.4f}")
    print(f"  Max:    {dim_r2_scores.max():.4f}")
    print(f"  Dims with R² > 0.1: {(dim_r2_scores > 0.1).sum()}/{n_dims}")
    print(f"  Dims with R² > 0.3: {(dim_r2_scores > 0.3).sum()}/{n_dims}")

    # Residual analysis
    residuals = y_diffsbdd - y_pred
    print(f"\nResidual Analysis:")
    print(f"  Mean residual:     {residuals.mean():.6f}")
    print(f"  Std residual:      {residuals.std():.4f}")
    print(f"  Max abs residual:  {np.abs(residuals).max():.4f}")

    # VERDICT
    print("\n" + "=" * 80)
    if overall_r2 > 0.3:
        verdict = "✓ SUCCESS! Strong alignment achieved."
        recommendation = "ESM can simulate DiffSBDD features. Use W·z_esm for conditioning."
    elif overall_r2 > 0.15:
        verdict = "○ MODERATE alignment achieved."
        recommendation = "Partial success. Consider λ tuning or hybrid approach."
    elif overall_r2 > 0.05:
        verdict = "⚠ WEAK alignment."
        recommendation = "Limited alignment. May need non-linear mapping or global conditioning."
    else:
        verdict = "✗ NO ALIGNMENT - Semantic gap confirmed."
        recommendation = "Pivot: Do not inject into residue features. Use global conditioning instead."

    print(f"VERDICT: {verdict}")
    print(f"RECOMMENDATION: {recommendation}")
    print("=" * 80)

    return {
        'overall_r2': overall_r2,
        'overall_mse': overall_mse,
        'dim_r2_scores': dim_r2_scores,
        'y_pred': y_pred,
        'residuals': residuals,
        'verdict': verdict,
        'recommendation': recommendation
    }


def create_visualizations(translator, X_test, y_test, eval_results, output_dir):
    """Create comprehensive alignment visualization."""
    print_header("CREATING VISUALIZATIONS")

    output_dir = Path(output_dir)
    y_pred = eval_results['y_pred']
    residuals = eval_results['residuals']
    dim_r2_scores = eval_results['dim_r2_scores']

    # ============================================================================
    # FIGURE 1: Alignment Quality
    # ============================================================================
    fig = plt.figure(figsize=(20, 12))

    # Panel 1: First dimension scatter (example)
    ax1 = plt.subplot(3, 3, 1)
    ax1.scatter(y_test[:, 0], y_pred[:, 0], alpha=0.2, s=1)
    ax1.plot([y_test[:, 0].min(), y_test[:, 0].max()],
             [y_test[:, 0].min(), y_test[:, 0].max()],
             'r--', linewidth=2, label='Perfect alignment')
    ax1.set_xlabel('DiffSBDD Feature (Dim 0)')
    ax1.set_ylabel('Projected ESM Feature (Dim 0)')
    ax1.set_title(f'Dimension 0 Alignment (R²={dim_r2_scores[0]:.3f})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Panel 2: Second dimension scatter
    ax2 = plt.subplot(3, 3, 2)
    ax2.scatter(y_test[:, 1], y_pred[:, 1], alpha=0.2, s=1)
    ax2.plot([y_test[:, 1].min(), y_test[:, 1].max()],
             [y_test[:, 1].min(), y_test[:, 1].max()],
             'r--', linewidth=2)
    ax2.set_xlabel('DiffSBDD Feature (Dim 1)')
    ax2.set_ylabel('Projected ESM Feature (Dim 1)')
    ax2.set_title(f'Dimension 1 Alignment (R²={dim_r2_scores[1]:.3f})')
    ax2.grid(True, alpha=0.3)

    # Panel 3: Third dimension scatter
    ax3 = plt.subplot(3, 3, 3)
    ax3.scatter(y_test[:, 2], y_pred[:, 2], alpha=0.2, s=1)
    ax3.plot([y_test[:, 2].min(), y_test[:, 2].max()],
             [y_test[:, 2].min(), y_test[:, 2].max()],
             'r--', linewidth=2)
    ax3.set_xlabel('DiffSBDD Feature (Dim 2)')
    ax3.set_ylabel('Projected ESM Feature (Dim 2)')
    ax3.set_title(f'Dimension 2 Alignment (R²={dim_r2_scores[2]:.3f})')
    ax3.grid(True, alpha=0.3)

    # Panel 4: Per-dimension R² scores
    ax4 = plt.subplot(3, 3, 4)
    dims = np.arange(len(dim_r2_scores))
    ax4.bar(dims, dim_r2_scores, alpha=0.7, edgecolor='black')
    ax4.axhline(0.3, color='g', linestyle='--', alpha=0.5, label='Strong (R²=0.3)')
    ax4.axhline(0.15, color='orange', linestyle='--', alpha=0.5, label='Moderate (R²=0.15)')
    ax4.axhline(0, color='k', linestyle='-', linewidth=0.5)
    ax4.set_xlabel('Dimension')
    ax4.set_ylabel('R² Score')
    ax4.set_title('Per-Dimension Alignment Quality')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    # Panel 5: R² distribution
    ax5 = plt.subplot(3, 3, 5)
    ax5.hist(dim_r2_scores, bins=30, alpha=0.7, edgecolor='black')
    ax5.axvline(dim_r2_scores.mean(), color='r', linestyle='--',
                linewidth=2, label=f'Mean: {dim_r2_scores.mean():.3f}')
    ax5.axvline(np.median(dim_r2_scores), color='g', linestyle='--',
                linewidth=2, label=f'Median: {np.median(dim_r2_scores):.3f}')
    ax5.set_xlabel('R² Score')
    ax5.set_ylabel('Count')
    ax5.set_title('Distribution of Per-Dimension R² Scores')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # Panel 6: Cumulative R²
    ax6 = plt.subplot(3, 3, 6)
    sorted_r2 = np.sort(dim_r2_scores)
    cumsum = np.arange(1, len(sorted_r2) + 1) / len(sorted_r2)
    ax6.plot(sorted_r2, cumsum, linewidth=2)
    ax6.axvline(0.1, color='orange', linestyle='--', alpha=0.5, label='R² = 0.1')
    ax6.axvline(0.3, color='g', linestyle='--', alpha=0.5, label='R² = 0.3')
    ax6.set_xlabel('R² Score')
    ax6.set_ylabel('Cumulative Fraction of Dimensions')
    ax6.set_title('CDF of Per-Dimension Alignment')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    # Panel 7: Residual distribution
    ax7 = plt.subplot(3, 3, 7)
    ax7.hist(residuals.flatten(), bins=100, alpha=0.7, edgecolor='black', density=True)
    ax7.axvline(0, color='r', linestyle='--', linewidth=2, label='Zero residual')
    ax7.set_xlabel('Residual (True - Predicted)')
    ax7.set_ylabel('Density')
    ax7.set_title('Residual Distribution')
    ax7.legend()
    ax7.grid(True, alpha=0.3)

    # Panel 8: Residual vs predicted (heteroscedasticity check)
    ax8 = plt.subplot(3, 3, 8)
    ax8.scatter(y_pred.flatten(), residuals.flatten(), alpha=0.1, s=1)
    ax8.axhline(0, color='r', linestyle='--', linewidth=2)
    ax8.set_xlabel('Predicted Value')
    ax8.set_ylabel('Residual')
    ax8.set_title('Residuals vs Predicted (Homoscedasticity Check)')
    ax8.grid(True, alpha=0.3)

    # Panel 9: Summary statistics
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    summary_text = f"""
LINEAR ALIGNMENT RESULTS

Overall Metrics:
  R² score:  {eval_results['overall_r2']:.4f}
  RMSE:      {np.sqrt(eval_results['overall_mse']):.4f}

Per-Dimension R²:
  Mean:      {dim_r2_scores.mean():.4f}
  Median:    {np.median(dim_r2_scores):.4f}
  R² > 0.1:  {(dim_r2_scores > 0.1).sum()}/{len(dim_r2_scores)}
  R² > 0.3:  {(dim_r2_scores > 0.3).sum()}/{len(dim_r2_scores)}

VERDICT:
{eval_results['verdict']}

RECOMMENDATION:
{eval_results['recommendation'][:40]}...
    """
    ax9.text(0.1, 0.5, summary_text, fontsize=9, family='monospace',
             verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    fig_path = output_dir / 'linear_alignment_results.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print_success(f"Saved: {fig_path}")
    plt.close()

    # ============================================================================
    # FIGURE 2: Translator Weight Analysis
    # ============================================================================
    fig2 = plt.figure(figsize=(16, 10))

    # Get weight matrix
    W = translator.coef_  # [D_out, D_in]
    b = translator.intercept_  # [D_out]

    # Panel 1: Weight matrix heatmap
    ax1 = plt.subplot(2, 3, 1)
    im = ax1.imshow(W, cmap='RdBu_r', aspect='auto',
                    vmin=-np.abs(W).max(), vmax=np.abs(W).max())
    ax1.set_xlabel('ESM Input Dimension')
    ax1.set_ylabel('DiffSBDD Output Dimension')
    ax1.set_title('Translator Weight Matrix W')
    plt.colorbar(im, ax=ax1)

    # Panel 2: Weight distribution
    ax2 = plt.subplot(2, 3, 2)
    ax2.hist(W.flatten(), bins=100, alpha=0.7, edgecolor='black')
    ax2.axvline(0, color='r', linestyle='--', linewidth=2)
    ax2.set_xlabel('Weight Value')
    ax2.set_ylabel('Count')
    ax2.set_title(f'Weight Distribution (μ={W.mean():.4f}, σ={W.std():.4f})')
    ax2.grid(True, alpha=0.3)

    # Panel 3: Bias distribution
    ax3 = plt.subplot(2, 3, 3)
    ax3.bar(np.arange(len(b)), b, alpha=0.7, edgecolor='black')
    ax3.axhline(0, color='r', linestyle='--', linewidth=1)
    ax3.set_xlabel('Output Dimension')
    ax3.set_ylabel('Bias Value')
    ax3.set_title(f'Translator Bias Terms (μ={b.mean():.4f})')
    ax3.grid(True, alpha=0.3)

    # Panel 4: Weight L2 norms per output dimension
    ax4 = plt.subplot(2, 3, 4)
    weight_norms = np.linalg.norm(W, axis=1)
    ax4.bar(np.arange(len(weight_norms)), weight_norms, alpha=0.7, edgecolor='black')
    ax4.set_xlabel('Output Dimension')
    ax4.set_ylabel('L2 Norm of Weights')
    ax4.set_title('Weight Magnitude per Output Dimension')
    ax4.grid(True, alpha=0.3)

    # Panel 5: Correlation between weight norm and R²
    ax5 = plt.subplot(2, 3, 5)
    ax5.scatter(weight_norms, dim_r2_scores, alpha=0.6)
    ax5.set_xlabel('Weight L2 Norm')
    ax5.set_ylabel('Dimension R² Score')
    ax5.set_title('Weight Magnitude vs Alignment Quality')
    ax5.grid(True, alpha=0.3)

    # Panel 6: Top contributing input dimensions
    ax6 = plt.subplot(2, 3, 6)
    input_importance = np.abs(W).mean(axis=0)
    top_k = 15
    top_dims = np.argsort(input_importance)[-top_k:][::-1]
    ax6.barh(np.arange(len(top_dims)), input_importance[top_dims], alpha=0.7, edgecolor='black')
    ax6.set_yticks(np.arange(len(top_dims)))
    ax6.set_yticklabels([f'Dim {d}' for d in top_dims])
    ax6.set_xlabel('Mean |Weight|')
    ax6.set_title(f'Top {top_k} Most Important ESM Dimensions')
    ax6.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    fig2_path = output_dir / 'translator_weight_analysis.png'
    plt.savefig(fig2_path, dpi=150, bbox_inches='tight')
    print_success(f"Saved: {fig2_path}")
    plt.close()

    return fig_path, fig2_path


def main():
    parser = argparse.ArgumentParser(
        description='Linear alignment of ESM-C to DiffSBDD feature space'
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
    parser.add_argument('--previous_results', type=str,
                        default='thesis_work/experiments/day6_pca_projection/latent_space_similarity_results.pkl',
                        help='Path to previous similarity analysis results (optional)')
    parser.add_argument('--extract_fresh', action='store_true',
                        help='Extract features fresh instead of using cached results')
    parser.add_argument('--alpha_values', type=float, nargs='+',
                        default=[0.01, 0.1, 1.0, 10.0, 100.0],
                        help='Ridge alpha values to try')
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

    print_header("LINEAR ALIGNMENT: ESM-C → DiffSBDD")

    # Try to load cached features first
    h_residues = None
    z_esm_pca = None

    if not args.extract_fresh and Path(args.previous_results).exists():
        try:
            print_header("LOADING CACHED FEATURES")
            # Try to load from a separate cache file
            cache_path = Path(args.output_dir) / 'extracted_features.npz'
            if cache_path.exists():
                print(f"Loading from: {cache_path}")
                data = np.load(cache_path)
                h_residues = data['h_residues']
                z_esm_pca = data['z_esm_pca']
                print_success(f"Loaded cached features")
                print(f"  h_residues: {h_residues.shape}")
                print(f"  z_esm_pca: {z_esm_pca.shape}")
        except Exception as e:
            print(f"Warning: Could not load cached features: {e}")

    # Extract fresh if needed
    if h_residues is None or z_esm_pca is None:
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

        # Extract features
        h_residues, z_esm_pca = extract_features(
            model, dataloader, pca_model, device=args.device, max_batches=args.max_batches
        )

        # Cache for future use
        cache_path = Path(args.output_dir) / 'extracted_features.npz'
        np.savez(cache_path, h_residues=h_residues, z_esm_pca=z_esm_pca)
        print_success(f"Cached features to: {cache_path}")

    # Train translator
    translator, training_results, (X_train, X_test, y_train, y_test) = train_linear_translator(
        z_esm_pca, h_residues, alpha_values=args.alpha_values
    )

    # Evaluate alignment
    eval_results = evaluate_alignment(translator, X_test, y_test)

    # Create visualizations
    fig1_path, fig2_path = create_visualizations(
        translator, X_test, y_test, eval_results, args.output_dir
    )

    # Save translator model
    translator_path = Path(args.output_dir) / 'esmc_to_diffsbdd_translator.pkl'
    with open(translator_path, 'wb') as f:
        pickle.dump(translator, f)
    print_success(f"Saved translator: {translator_path}")

    # Save full results
    results_path = Path(args.output_dir) / 'linear_alignment_results.pkl'
    with open(results_path, 'wb') as f:
        pickle.dump({
            'translator': translator,
            'training_results': training_results,
            'eval_results': eval_results,
            'args': vars(args)
        }, f)
    print_success(f"Saved results: {results_path}")

    # Final summary
    print_header("FINAL SUMMARY")
    print(f"\nAlignment R² Score: {eval_results['overall_r2']:.4f}")
    print(f"\n{eval_results['verdict']}")
    print(f"\n{eval_results['recommendation']}")

    print("\n" + "="*80)
    print("Analysis complete! Check the output:")
    print(f"  1. Alignment quality: {fig1_path}")
    print(f"  2. Weight analysis: {fig2_path}")
    print(f"  3. Translator model: {translator_path}")
    print("="*80)

    # Next steps guidance
    print("\nNEXT STEPS:")
    if eval_results['overall_r2'] > 0.3:
        print("  ✓ Strong alignment achieved!")
        print("  → Use the translator in inference:")
        print("     h_new = h_diffsbdd + λ · translator.predict(z_esm_pca)")
        print("  → Try λ values: 0.1, 0.3, 0.5, 1.0")
    elif eval_results['overall_r2'] > 0.1:
        print("  ○ Moderate alignment achieved")
        print("  → Test with small λ values: 0.05, 0.1, 0.2")
        print("  → Consider non-linear mapping (MLP)")
    else:
        print("  ✗ Weak alignment - semantic gap confirmed")
        print("  → Pivot to global conditioning (inject into y_global)")
        print("  → Or try FiLM layers (γ, β modulation)")
    print("="*80)


if __name__ == '__main__':
    main()
