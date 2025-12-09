"""
Extract Real Latent Representations from Checkpoint Model

This script extracts actual h_residues from a trained checkpoint and compares
them with PCA-projected ESM-C embeddings across different lambda values to
determine the optimal conditioning strength.

Key Analysis:
1. Load checkpoint model
2. Hook into EGNN forward pass to capture real h_residues
3. Extract PCA-projected embeddings (z_esm_pca)
4. Test multiple lambda values [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
5. Compare distributions and compute optimal balance metrics

Usage:
    python extract_real_latent_representations.py \
        --checkpoint checkpoints/crossdocked_fullatom_cond.ckpt \
        --test_dir data/real_testing_dataset_10_tests/test \
        --pca_model thesis_work/experiments/day6_pca_projection/pca_model_32d.pkl
"""

import torch
import numpy as np
import pickle
import argparse
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy import stats
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


class LatentCapture:
    """Hook to capture intermediate latent representations."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset captured data."""
        self.h_residues_before = []  # Before PCA addition
        self.h_residues_after = []   # After PCA addition
        self.z_esm_pca = []          # PCA-projected embeddings
        self.timestamps = []         # Diffusion timesteps

    def get_statistics(self):
        """Compute statistics from captured data."""
        if not self.h_residues_before:
            return None

        h_before = np.concatenate(self.h_residues_before, axis=0)
        h_after = np.concatenate(self.h_residues_after, axis=0)
        z_pca = np.concatenate(self.z_esm_pca, axis=0)

        return {
            'h_residues_before': {
                'mean': h_before.mean(),
                'std': h_before.std(),
                'min': h_before.min(),
                'max': h_before.max(),
                'l2_norm_mean': np.linalg.norm(h_before, axis=1).mean(),
                'l2_norm_std': np.linalg.norm(h_before, axis=1).std(),
                'data': h_before,
            },
            'h_residues_after': {
                'mean': h_after.mean(),
                'std': h_after.std(),
                'min': h_after.min(),
                'max': h_after.max(),
                'l2_norm_mean': np.linalg.norm(h_after, axis=1).mean(),
                'l2_norm_std': np.linalg.norm(h_after, axis=1).std(),
                'data': h_after,
            },
            'z_esm_pca': {
                'mean': z_pca.mean(),
                'std': z_pca.std(),
                'min': z_pca.min(),
                'max': z_pca.max(),
                'l2_norm_mean': np.linalg.norm(z_pca, axis=1).mean(),
                'l2_norm_std': np.linalg.norm(z_pca, axis=1).std(),
                'data': z_pca,
            },
        }


def create_forward_hook(capture, lambda_value):
    """Create a forward hook to capture h_residues and z_esm_pca."""

    def hook(module, inputs, outputs):
        """
        Hook into EGNN forward pass.

        Inputs: (xh_atoms, xh_residues, t, mask_atoms, mask_residues, pocket_emb)
        """
        xh_atoms, xh_residues, t, mask_atoms, mask_residues, pocket_emb = inputs

        with torch.no_grad():
            # Extract h_residues (before encoding)
            h_residues_raw = xh_residues[:, module.n_dims:].clone()

            # Encode residues
            h_residues_encoded = module.residue_encoder(h_residues_raw)

            # Capture BEFORE PCA addition
            capture.h_residues_before.append(
                h_residues_encoded.cpu().numpy()
            )

            # Apply PCA transformation (if available)
            if module.use_pca and pocket_emb is not None and module.pca_model is not None:
                # Transform ESM-C embeddings using PCA
                z_esm_pca = module.pca_model.transform(pocket_emb.cpu().numpy())
                z_esm_pca = torch.from_numpy(z_esm_pca).float().to(pocket_emb.device)

                # Expand to per-residue
                z_esm_pca_expanded = z_esm_pca[mask_residues.long()]

                # Capture PCA embeddings
                capture.z_esm_pca.append(
                    z_esm_pca_expanded.cpu().numpy()
                )

                # Apply scaled PCA (simulating different lambda)
                h_residues_after_pca = h_residues_encoded + lambda_value * z_esm_pca_expanded

                # Capture AFTER PCA addition
                capture.h_residues_after.append(
                    h_residues_after_pca.cpu().numpy()
                )
            else:
                # No PCA applied
                capture.h_residues_after.append(
                    h_residues_encoded.cpu().numpy()
                )
                capture.z_esm_pca.append(
                    np.zeros_like(h_residues_encoded.cpu().numpy())
                )

            # Capture timestep
            if hasattr(t, 'item'):
                capture.timestamps.append(t.item())
            else:
                capture.timestamps.append(t[0].item() if len(t) > 0 else 0)

    return hook


def load_checkpoint_with_pca(checkpoint_path, pca_model_path):
    """Load checkpoint and inject PCA model."""
    print_header("LOADING CHECKPOINT")

    print(f"Loading checkpoint: {checkpoint_path}")
    model = LigandPocketDDPM.load_from_checkpoint(
        checkpoint_path,
        map_location='cpu'
    )
    model.eval()

    print_success(f"Loaded model from {checkpoint_path}")

    # Load PCA model
    if pca_model_path:
        print(f"Loading PCA model: {pca_model_path}")
        with open(pca_model_path, 'rb') as f:
            pca_model = pickle.load(f)

        # Inject PCA into dynamics
        model.ddpm.dynamics.pca_model = pca_model
        model.ddpm.dynamics.use_pca = True

        print_success(f"Loaded PCA model: 960D → {pca_model.n_components_}D")
        print_success(f"Variance explained: {pca_model.explained_variance_ratio_.sum() * 100:.2f}%")

    return model


def extract_latents_with_lambda(model, dataloader, lambda_value, device='cpu'):
    """Extract latent representations for a specific lambda value."""
    print(f"\n→ Testing λ = {lambda_value:.4f}")

    capture = LatentCapture()

    # Register hook
    hook_handle = model.ddpm.dynamics.register_forward_hook(
        create_forward_hook(capture, lambda_value)
    )

    model.to(device)
    model.eval()

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"λ={lambda_value:.3f}")):
            if batch_idx >= 50:  # Limit to 50 batches for speed
                break

            # Extract batch data
            lig_coords = batch['lig_coords'].to(device)
            lig_one_hot = batch['lig_one_hot'].to(device)
            pocket_coords = batch['pocket_coords'].to(device)
            pocket_one_hot = batch['pocket_one_hot'].to(device)
            lig_mask = batch['lig_mask'].to(device)
            pocket_mask = batch['pocket_mask'].to(device)
            pocket_emb = batch.get('pocket_emb', None)
            if pocket_emb is not None:
                pocket_emb = pocket_emb.to(device)

            # Create timestep
            batch_size = pocket_mask.max().long().item() + 1
            t = torch.zeros(batch_size, device=device)

            # Combine coords and features (xh format)
            xh_atoms = torch.cat([lig_coords, lig_one_hot], dim=1)
            xh_residues = torch.cat([pocket_coords, pocket_one_hot], dim=1)

            # Forward pass (hook will capture latents)
            try:
                _ = model.ddpm.dynamics(
                    xh_atoms, xh_residues, t,
                    lig_mask, pocket_mask,
                    pocket_emb=pocket_emb
                )
            except Exception as e:
                print(f"Warning: Error in forward pass: {e}")
                continue

    # Remove hook
    hook_handle.remove()

    # Get statistics
    stats = capture.get_statistics()

    if stats is not None:
        print(f"  h_residues (before): mean={stats['h_residues_before']['mean']:.4f}, "
              f"std={stats['h_residues_before']['std']:.4f}")
        print(f"  z_esm_pca:           mean={stats['z_esm_pca']['mean']:.4f}, "
              f"std={stats['z_esm_pca']['std']:.4f}")
        print(f"  h_residues (after):  mean={stats['h_residues_after']['mean']:.4f}, "
              f"std={stats['h_residues_after']['std']:.4f}")

    return stats


def compute_balance_metrics(stats, lambda_value):
    """
    Compute metrics to evaluate the balance between h_residues and PCA contribution.

    Key metrics:
    - Relative contribution: std(λ * z_pca) / std(h_residues)
    - Signal-to-noise ratio
    - Distribution overlap (KL divergence, Wasserstein distance)
    """
    if stats is None:
        return None

    h_before = stats['h_residues_before']
    z_pca = stats['z_esm_pca']
    h_after = stats['h_residues_after']

    # Scaled PCA contribution
    scaled_pca_std = lambda_value * z_pca['std']

    # Relative contribution (key metric)
    relative_contribution = scaled_pca_std / h_before['std']

    # Change in h_residues due to PCA
    delta_h_std = abs(h_after['std'] - h_before['std'])
    percent_change = (delta_h_std / h_before['std']) * 100

    # L2 norm comparison
    l2_ratio = (lambda_value * z_pca['l2_norm_mean']) / h_before['l2_norm_mean']

    # Distribution similarity (Wasserstein distance)
    h_before_flat = h_before['data'].flatten()
    z_pca_flat = z_pca['data'].flatten()

    # Subsample for efficiency
    subsample_size = min(10000, len(h_before_flat), len(z_pca_flat))
    h_before_sample = np.random.choice(h_before_flat, subsample_size, replace=False)
    z_pca_sample = np.random.choice(z_pca_flat, subsample_size, replace=False)

    wasserstein_dist = stats.wasserstein_distance(h_before_sample, z_pca_sample)

    return {
        'lambda': lambda_value,
        'relative_contribution': relative_contribution,
        'percent_change_std': percent_change,
        'l2_norm_ratio': l2_ratio,
        'wasserstein_distance': wasserstein_dist,
        'h_before_std': h_before['std'],
        'z_pca_std': z_pca['std'],
        'scaled_pca_std': scaled_pca_std,
        'h_after_std': h_after['std'],
    }


def find_optimal_lambda(results):
    """
    Determine optimal lambda based on multiple criteria.

    Optimal lambda should:
    1. Provide meaningful contribution (15-35% relative strength)
    2. Not overwhelm pretrained features (< 50%)
    3. Balance signal-to-noise
    """
    print_header("OPTIMAL LAMBDA ANALYSIS")

    print(f"{'Lambda':>8} {'Rel. Contrib':>12} {'% Change':>10} {'L2 Ratio':>10} {'Wasserstein':>12} {'Rating':>8}")
    print("-" * 70)

    best_lambda = None
    best_score = float('inf')

    for result in results:
        if result is None:
            continue

        rel = result['relative_contribution']
        pct = result['percent_change_std']
        l2 = result['l2_norm_ratio']
        wd = result['wasserstein_distance']

        # Score based on how close to ideal range (0.15 - 0.35)
        ideal_min, ideal_max = 0.15, 0.35
        if rel < ideal_min:
            score = (ideal_min - rel) * 2  # Penalize being too weak
        elif rel > ideal_max:
            score = (rel - ideal_max) * 3  # Heavily penalize being too strong
        else:
            score = 0  # Perfect range

        # Rating
        if 0.15 <= rel <= 0.35:
            rating = "✓ OPTIMAL"
        elif 0.10 <= rel <= 0.50:
            rating = "○ GOOD"
        elif rel < 0.10:
            rating = "✗ WEAK"
        else:
            rating = "✗ STRONG"

        print(f"{result['lambda']:8.4f} {rel:12.1%} {pct:9.2f}% {l2:10.3f} {wd:12.4f} {rating:>8}")

        if score < best_score:
            best_score = score
            best_lambda = result

    print("\n" + "=" * 70)
    if best_lambda:
        print(f"RECOMMENDED LAMBDA: {best_lambda['lambda']:.4f}")
        print(f"  - Relative contribution: {best_lambda['relative_contribution']:.1%}")
        print(f"  - Change in h_residues: {best_lambda['percent_change_std']:.2f}%")
        print(f"  - L2 norm ratio: {best_lambda['l2_norm_ratio']:.3f}")
        print("=" * 70)

    return best_lambda


def create_comparison_visualization(lambda_results, all_stats, output_path):
    """Create comprehensive visualization comparing different lambda values."""
    print_header("CREATING VISUALIZATION")

    fig = plt.figure(figsize=(20, 12))

    # Panel 1: Relative Contribution vs Lambda
    ax1 = plt.subplot(3, 4, 1)
    lambdas = [r['lambda'] for r in lambda_results if r is not None]
    rel_contribs = [r['relative_contribution'] for r in lambda_results if r is not None]
    ax1.plot(lambdas, rel_contribs, 'bo-', linewidth=2, markersize=8)
    ax1.axhline(0.15, color='g', linestyle='--', alpha=0.5, label='Ideal min (15%)')
    ax1.axhline(0.35, color='g', linestyle='--', alpha=0.5, label='Ideal max (35%)')
    ax1.axhline(0.50, color='r', linestyle='--', alpha=0.5, label='Too strong (50%)')
    ax1.set_xlabel('Lambda (λ)')
    ax1.set_ylabel('Relative Contribution')
    ax1.set_title('Panel 1: Relative Contribution vs Lambda')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')

    # Panel 2: L2 Norm Ratio vs Lambda
    ax2 = plt.subplot(3, 4, 2)
    l2_ratios = [r['l2_norm_ratio'] for r in lambda_results if r is not None]
    ax2.plot(lambdas, l2_ratios, 'ro-', linewidth=2, markersize=8)
    ax2.set_xlabel('Lambda (λ)')
    ax2.set_ylabel('L2 Norm Ratio')
    ax2.set_title('Panel 2: L2 Norm Ratio (PCA/h_residues)')
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')

    # Panel 3: Percent Change in Std vs Lambda
    ax3 = plt.subplot(3, 4, 3)
    pct_changes = [r['percent_change_std'] for r in lambda_results if r is not None]
    ax3.plot(lambdas, pct_changes, 'go-', linewidth=2, markersize=8)
    ax3.set_xlabel('Lambda (λ)')
    ax3.set_ylabel('% Change in Std')
    ax3.set_title('Panel 3: Impact on h_residues Distribution')
    ax3.grid(True, alpha=0.3)
    ax3.set_xscale('log')

    # Panel 4: Wasserstein Distance vs Lambda
    ax4 = plt.subplot(3, 4, 4)
    wasserstein_dists = [r['wasserstein_distance'] for r in lambda_results if r is not None]
    ax4.plot(lambdas, wasserstein_dists, 'mo-', linewidth=2, markersize=8)
    ax4.set_xlabel('Lambda (λ)')
    ax4.set_ylabel('Wasserstein Distance')
    ax4.set_title('Panel 4: Distribution Similarity')
    ax4.grid(True, alpha=0.3)
    ax4.set_xscale('log')

    # Panels 5-8: Distribution comparisons for selected lambdas
    selected_lambdas = [0.01, 0.1, 0.2, 0.5]
    for idx, lambda_val in enumerate(selected_lambdas):
        ax = plt.subplot(3, 4, 5 + idx)

        # Find stats for this lambda
        stats = None
        for lam, st in zip(lambdas, all_stats):
            if abs(lam - lambda_val) < 1e-6:
                stats = st
                break

        if stats is not None:
            h_before = stats['h_residues_before']['data'].flatten()
            z_pca = stats['z_esm_pca']['data'].flatten()

            # Subsample for plotting
            sample_size = min(10000, len(h_before))
            h_sample = np.random.choice(h_before, sample_size, replace=False)
            z_pca_sample = np.random.choice(z_pca, sample_size, replace=False) * lambda_val

            ax.hist(h_sample, bins=50, alpha=0.5, label='h_residues', density=True, edgecolor='black')
            ax.hist(z_pca_sample, bins=50, alpha=0.5, label=f'λ·z_pca (λ={lambda_val})', density=True, edgecolor='black')
            ax.set_xlabel('Value')
            ax.set_ylabel('Density')
            ax.set_title(f'Panel {5+idx}: λ={lambda_val}')
            ax.legend(fontsize=8)

    # Panels 9-12: Per-component std comparison for selected lambdas
    for idx, lambda_val in enumerate(selected_lambdas):
        ax = plt.subplot(3, 4, 9 + idx)

        # Find stats for this lambda
        stats = None
        for lam, st in zip(lambdas, all_stats):
            if abs(lam - lambda_val) < 1e-6:
                stats = st
                break

        if stats is not None:
            h_before_data = stats['h_residues_before']['data']
            z_pca_data = stats['z_esm_pca']['data']

            # Per-component std
            h_stds = h_before_data.std(axis=0)
            z_stds = (lambda_val * z_pca_data).std(axis=0)

            n_comp = min(20, len(h_stds))
            x_pos = np.arange(n_comp)
            width = 0.35

            ax.bar(x_pos - width/2, h_stds[:n_comp], width, alpha=0.7, label='h_residues')
            ax.bar(x_pos + width/2, z_stds[:n_comp], width, alpha=0.7, label=f'λ·z_pca')
            ax.set_xlabel('Component')
            ax.set_ylabel('Std Dev')
            ax.set_title(f'Panel {9+idx}: Component-wise (λ={lambda_val})')
            ax.legend(fontsize=8)
            ax.set_xticks([0, 5, 10, 15, 19])

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print_success(f"Saved visualization: {output_path}")

    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Extract real latent representations and find optimal lambda"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/crossdocked_fullatom_cond.ckpt",
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--test_npz",
        type=str,
        default="data/real_testing_dataset_10_tests/test.npz",
        help="Path to test dataset npz file"
    )
    parser.add_argument(
        "--test_esmc",
        type=str,
        default="data/real_testing_dataset_10_tests/test_esmc.npz",
        help="Path to test ESM-C embeddings"
    )
    parser.add_argument(
        "--pca_model",
        type=str,
        default="thesis_work/experiments/day6_pca_projection/pca_model_32d.pkl",
        help="Path to PCA model"
    )
    parser.add_argument(
        "--lambda_values",
        type=float,
        nargs='+',
        default=[0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0],
        help="Lambda values to test"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="thesis_work/experiments/day6_pca_projection",
        help="Output directory"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use (cpu/cuda)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for dataloader"
    )

    args = parser.parse_args()

    print_header("REAL LATENT REPRESENTATION EXTRACTION")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Test npz: {args.test_npz}")
    print(f"Test ESM-C: {args.test_esmc}")
    print(f"PCA model: {args.pca_model}")
    print(f"Lambda values: {args.lambda_values}")
    print(f"Device: {args.device}")

    # Load model
    model = load_checkpoint_with_pca(args.checkpoint, args.pca_model)

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

    # Extract latents for each lambda value
    print_header("EXTRACTING LATENTS FOR DIFFERENT LAMBDA VALUES")

    all_stats = []
    lambda_results = []

    for lambda_val in args.lambda_values:
        stats = extract_latents_with_lambda(
            model, dataloader, lambda_val, device=args.device
        )
        all_stats.append(stats)

        # Compute balance metrics
        metrics = compute_balance_metrics(stats, lambda_val)
        lambda_results.append(metrics)

    # Find optimal lambda
    optimal = find_optimal_lambda(lambda_results)

    # Create visualization
    output_path = Path(args.output_dir) / "optimal_lambda_analysis.png"
    create_comparison_visualization(lambda_results, all_stats, output_path)

    # Save results
    results_path = Path(args.output_dir) / "lambda_analysis_results.pkl"
    with open(results_path, 'wb') as f:
        pickle.dump({
            'lambda_results': lambda_results,
            'optimal_lambda': optimal,
            'all_stats': all_stats,
        }, f)
    print_success(f"Saved results: {results_path}")

    print_header("ANALYSIS COMPLETE")
    if optimal:
        print(f"\n🎯 RECOMMENDED LAMBDA: {optimal['lambda']:.4f}\n")
        print(f"Update your config with: pca_lambda: {optimal['lambda']:.4f}")

    return optimal


if __name__ == "__main__":
    main()
