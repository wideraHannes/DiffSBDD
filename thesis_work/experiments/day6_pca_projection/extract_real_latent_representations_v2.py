"""
Extract Real Latent Representations from Checkpoint Model - Simple Version

This script extracts h_residues and compares with PCA embeddings to find optimal lambda.

Usage:
    python extract_real_latent_representations_v2.py
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


def extract_h_residues_and_pca(model, dataloader, pca_model, device='cpu'):
    """
    Extract real h_residues and PCA embeddings from the model.

    Returns h_residues BEFORE any PCA addition (the baseline features),
    and the PCA-projected ESM-C embeddings.
    """
    print_header("EXTRACTING LATENT REPRESENTATIONS")

    model.to(device)
    model.eval()

    h_residues_list = []
    z_esm_pca_list = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Extracting")):
            if batch_idx >= 20:  # Limit for speed
                break

            try:
                # Get pocket data
                pocket_coords = batch['pocket_coords'].to(device).float()
                pocket_one_hot = batch['pocket_one_hot'].to(device).float()
                pocket_mask = batch['pocket_mask'].to(device).long()
                pocket_emb = batch.get('pocket_emb', None)

                if pocket_emb is None:
                    print("Warning: No pocket_emb in batch")
                    continue

                pocket_emb = pocket_emb.to(device).float()

                # 1. Extract h_residues (encoded residue features BEFORE PCA)
                xh_residues = torch.cat([pocket_coords, pocket_one_hot], dim=1)
                h_residues_raw = xh_residues[:, 3:]  # Skip first 3 dims (coords)

                # Encode through residue_encoder
                h_residues_encoded = model.ddpm.dynamics.residue_encoder(h_residues_raw)
                h_residues_list.append(h_residues_encoded.cpu().numpy())

                # 2. Extract PCA-projected embeddings
                z_esm_pca = pca_model.transform(pocket_emb.cpu().numpy())
                z_esm_pca = torch.from_numpy(z_esm_pca).float().to(device)

                # Expand to per-residue using mask
                z_esm_pca_expanded = z_esm_pca[pocket_mask]
                z_esm_pca_list.append(z_esm_pca_expanded.cpu().numpy())

            except Exception as e:
                print(f"Warning: Error processing batch {batch_idx}: {e}")
                continue

    if not h_residues_list:
        print("ERROR: No data extracted!")
        return None, None

    h_residues_all = np.concatenate(h_residues_list, axis=0)
    z_esm_pca_all = np.concatenate(z_esm_pca_list, axis=0)

    print_success(f"Extracted {len(h_residues_all)} h_residues samples")
    print_success(f"Extracted {len(z_esm_pca_all)} z_esm_pca samples")
    print(f"  h_residues shape: {h_residues_all.shape}")
    print(f"  z_esm_pca shape: {z_esm_pca_all.shape}")

    return h_residues_all, z_esm_pca_all


def analyze_lambda_values(h_residues, z_esm_pca, lambda_values):
    """
    Analyze different lambda values to find optimal conditioning strength.

    Key metric: Relative contribution = std(λ * z_pca) / std(h_residues)
    Target: 15-35% for balanced conditioning
    """
    print_header("ANALYZING LAMBDA VALUES")

    results = []

    # Baseline stats
    h_std = h_residues.std()
    z_std = z_esm_pca.std()
    h_l2_mean = np.linalg.norm(h_residues, axis=1).mean()
    z_l2_mean = np.linalg.norm(z_esm_pca, axis=1).mean()

    print(f"Baseline Statistics:")
    print(f"  h_residues: mean={h_residues.mean():.4f}, std={h_std:.4f}, L2={h_l2_mean:.4f}")
    print(f"  z_esm_pca:  mean={z_esm_pca.mean():.4f}, std={z_std:.4f}, L2={z_l2_mean:.4f}")
    print()

    print(f"{'Lambda':>8} {'Scaled PCA Std':>15} {'Rel. Contrib':>13} {'L2 Ratio':>10} {'Rating':>10}")
    print("-" * 70)

    best_lambda = None
    best_score = float('inf')

    for lam in lambda_values:
        # Scaled PCA contribution
        scaled_pca_std = lam * z_std
        scaled_pca_l2 = lam * z_l2_mean

        # Relative contribution (key metric)
        rel_contrib = scaled_pca_std / h_std
        l2_ratio = scaled_pca_l2 / h_l2_mean

        # Score (how far from ideal 15-35% range)
        if 0.15 <= rel_contrib <= 0.35:
            score = 0.0  # Perfect!
            rating = "✓ OPTIMAL"
        elif 0.10 <= rel_contrib <= 0.50:
            score = min(abs(rel_contrib - 0.15), abs(rel_contrib - 0.35))
            rating = "○ GOOD"
        elif rel_contrib < 0.10:
            score = (0.15 - rel_contrib) * 2
            rating = "✗ WEAK"
        else:
            score = (rel_contrib - 0.35) * 3
            rating = "✗ STRONG"

        print(f"{lam:8.4f} {scaled_pca_std:15.4f} {rel_contrib:12.1%} {l2_ratio:10.3f} {rating:>10}")

        results.append({
            'lambda': lam,
            'scaled_pca_std': scaled_pca_std,
            'rel_contrib': rel_contrib,
            'l2_ratio': l2_ratio,
            'rating': rating,
            'score': score,
        })

        if score < best_score:
            best_score = score
            best_lambda = results[-1]

    print("\n" + "=" * 70)
    if best_lambda:
        print(f"🎯 RECOMMENDED LAMBDA: {best_lambda['lambda']:.4f}")
        print(f"   Relative contribution: {best_lambda['rel_contrib']:.1%}")
        print(f"   L2 ratio: {best_lambda['l2_ratio']:.3f}")
        print(f"   Rating: {best_lambda['rating']}")
    print("=" * 70)

    return results, best_lambda


def create_visualization(h_residues, z_esm_pca, results, output_path):
    """Create comprehensive visualization of lambda analysis."""
    print_header("CREATING VISUALIZATION")

    fig = plt.figure(figsize=(18, 10))

    # Panel 1: h_residues distribution
    ax1 = plt.subplot(2, 4, 1)
    h_flat = h_residues.flatten()
    h_sample = np.random.choice(h_flat, min(10000, len(h_flat)), replace=False)
    ax1.hist(h_sample, bins=50, alpha=0.7, edgecolor='black', density=True)
    ax1.set_xlabel('Value')
    ax1.set_ylabel('Density')
    ax1.set_title(f'h_residues Distribution\nMean={h_residues.mean():.3f}, Std={h_residues.std():.3f}')
    ax1.axvline(0, color='r', linestyle='--', alpha=0.5)

    # Panel 2: z_esm_pca distribution
    ax2 = plt.subplot(2, 4, 2)
    z_flat = z_esm_pca.flatten()
    z_sample = np.random.choice(z_flat, min(10000, len(z_flat)), replace=False)
    ax2.hist(z_sample, bins=50, alpha=0.7, edgecolor='black', density=True, color='orange')
    ax2.set_xlabel('Value')
    ax2.set_ylabel('Density')
    ax2.set_title(f'z_esm_pca Distribution\nMean={z_esm_pca.mean():.3f}, Std={z_esm_pca.std():.3f}')
    ax2.axvline(0, color='r', linestyle='--', alpha=0.5)

    # Panel 3: Relative Contribution vs Lambda
    ax3 = plt.subplot(2, 4, 3)
    lambdas = [r['lambda'] for r in results]
    rel_contribs = [r['rel_contrib'] for r in results]
    ax3.plot(lambdas, rel_contribs, 'bo-', linewidth=2, markersize=8)
    ax3.axhline(0.15, color='g', linestyle='--', alpha=0.5, label='Ideal min (15%)')
    ax3.axhline(0.35, color='g', linestyle='--', alpha=0.5, label='Ideal max (35%)')
    ax3.axhline(0.50, color='r', linestyle='--', alpha=0.5, label='Too strong (50%)')
    ax3.fill_between(lambdas, 0.15, 0.35, alpha=0.2, color='green', label='Optimal range')
    ax3.set_xlabel('Lambda (λ)')
    ax3.set_ylabel('Relative Contribution')
    ax3.set_title('Relative Contribution vs Lambda')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # Panel 4: L2 Norm Ratio vs Lambda
    ax4 = plt.subplot(2, 4, 4)
    l2_ratios = [r['l2_ratio'] for r in results]
    ax4.plot(lambdas, l2_ratios, 'ro-', linewidth=2, markersize=8)
    ax4.set_xlabel('Lambda (λ)')
    ax4.set_ylabel('L2 Norm Ratio')
    ax4.set_title('L2 Norm Ratio (PCA/h_residues)')
    ax4.grid(True, alpha=0.3)

    # Panels 5-8: Distribution comparisons for selected lambdas
    selected_lambdas = [0.05, 0.1, 0.2, 0.5]
    for idx, lam in enumerate(selected_lambdas):
        ax = plt.subplot(2, 4, 5 + idx)

        # Scale PCA by lambda
        scaled_pca = lam * z_esm_pca

        # Sample for plotting
        h_sample = np.random.choice(h_flat, min(5000, len(h_flat)), replace=False)
        z_sample = np.random.choice(scaled_pca.flatten(), min(5000, len(scaled_pca.flatten())), replace=False)

        ax.hist(h_sample, bins=40, alpha=0.5, label='h_residues', density=True, edgecolor='black')
        ax.hist(z_sample, bins=40, alpha=0.5, label=f'λ·z_pca (λ={lam})', density=True, edgecolor='black', color='orange')
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')

        # Find the result for this lambda
        result = next((r for r in results if r['lambda'] == lam), None)
        if result:
            ax.set_title(f'λ={lam}: {result["rel_contrib"]:.1%} contrib ({result["rating"].replace("✓ ", "").replace("○ ", "").replace("✗ ", "")})')
        else:
            ax.set_title(f'λ={lam}')
        ax.legend(fontsize=8)

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
        default=2,
        help="Batch size for dataloader"
    )

    args = parser.parse_args()

    print_header("REAL LATENT REPRESENTATION EXTRACTION - V2")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Test npz: {args.test_npz}")
    print(f"Test ESM-C: {args.test_esmc}")
    print(f"PCA model: {args.pca_model}")
    print(f"Lambda values: {args.lambda_values}")

    # Load model
    print_header("LOADING CHECKPOINT")
    model = LigandPocketDDPM.load_from_checkpoint(args.checkpoint, map_location=args.device)
    model.eval()
    print_success(f"Loaded checkpoint: {args.checkpoint}")

    # Load PCA model
    print_header("LOADING PCA MODEL")
    with open(args.pca_model, 'rb') as f:
        pca_model = pickle.load(f)
    print_success(f"Loaded PCA: 960D → {pca_model.n_components_}D")
    print_success(f"Variance explained: {pca_model.explained_variance_ratio_.sum() * 100:.2f}%")

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
    h_residues, z_esm_pca = extract_h_residues_and_pca(model, dataloader, pca_model, device=args.device)

    if h_residues is None or z_esm_pca is None:
        print("ERROR: Failed to extract latent representations!")
        return None

    # Analyze lambda values
    results, best_lambda = analyze_lambda_values(h_residues, z_esm_pca, args.lambda_values)

    # Create visualization
    output_path = Path(args.output_dir) / "optimal_lambda_analysis_v2.png"
    create_visualization(h_residues, z_esm_pca, results, output_path)

    # Save results
    results_path = Path(args.output_dir) / "lambda_analysis_results_v2.pkl"
    with open(results_path, 'wb') as f:
        pickle.dump({
            'results': results,
            'best_lambda': best_lambda,
            'h_residues_stats': {
                'mean': h_residues.mean(),
                'std': h_residues.std(),
                'shape': h_residues.shape,
            },
            'z_esm_pca_stats': {
                'mean': z_esm_pca.mean(),
                'std': z_esm_pca.std(),
                'shape': z_esm_pca.shape,
            },
        }, f)
    print_success(f"Saved results: {results_path}")

    print_header("ANALYSIS COMPLETE")
    if best_lambda:
        print(f"\n🎯 RECOMMENDED LAMBDA: {best_lambda['lambda']:.4f}\n")
        print(f"Add this to your config:")
        print(f"  pca_lambda: {best_lambda['lambda']:.4f}")
        print()

    return best_lambda


if __name__ == "__main__":
    main()
