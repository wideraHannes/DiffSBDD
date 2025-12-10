"""
Experiment A: EGNN Sensitivity Test

Goal: Test whether ESM-PCA steering actually changes EGNN's immediate ligand
      update behavior without running full diffusion sampling.

Approach:
- Hold input pocket, noisy ligand state, and diffusion time fixed
- Compare predicted ligand velocities and feature updates for λ=0 vs λ>0
- Test true ESM embeddings vs matched random embeddings

Outputs:
- CSV: batch_idx, t, lambda, condition (true/random), n_lig_atoms,
       delta_v_mean, delta_v_median, delta_h_mean, delta_h_median
- Optional: Visualization comparing true vs random across λ values

Usage:
    python expA_egnn_sensitivity.py
    python expA_egnn_sensitivity.py --checkpoint path/to/model.ckpt
    python expA_egnn_sensitivity.py --max-batches 20 --no-plot
"""

import torch
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader
import argparse
import sys
from tqdm import tqdm

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


def run_dyn(dyn, xh_atoms, xh_residues, t, mask_atoms, mask_residues,
            pocket_emb, lam, pocket_emb_override=None):
    """
    Run dynamics forward with specified lambda and optional embedding override.

    Args:
        dyn: EGNNDynamics module
        xh_atoms: [n_atoms, 3+atom_nf] ligand coordinates + features
        xh_residues: [n_residues, 3+residue_nf] pocket coordinates + features
        t: scalar tensor, diffusion timestep
        mask_atoms: [n_atoms] batch indices for ligand atoms
        mask_residues: [n_residues] batch indices for residues
        pocket_emb: [batch_size, 960] true ESM-C embeddings
        lam: float, PCA lambda scaling factor
        pocket_emb_override: optional [batch_size, 960] replacement embedding

    Returns:
        vel_atoms: [n_atoms, 3] - predicted velocity
        hat_h_atoms: [n_atoms, atom_nf] - predicted feature updates
    """
    # Set lambda
    original_lambda = dyn.pca_lambda
    dyn.pca_lambda = lam

    # Use override if provided
    emb = pocket_emb_override if pocket_emb_override is not None else pocket_emb

    # Forward pass (dynamics.py:144-287)
    # Returns: (ligand_output, pocket_output) where each is [coords+features]
    with torch.no_grad():
        out_atoms, _ = dyn(xh_atoms, xh_residues, t, mask_atoms, mask_residues,
                          pocket_emb=emb)

    # Split output (dynamics.py:285-287)
    vel_atoms = out_atoms[:, :3]  # First 3 dims are velocity
    hat_h_atoms = out_atoms[:, 3:]  # Remaining dims are feature updates

    # Restore lambda
    dyn.pca_lambda = original_lambda

    return vel_atoms, hat_h_atoms


def create_visualization(df, output_path):
    """
    Create bar chart comparing delta_v_mean for true vs random across λ values.

    Expected result: True ESM embeddings should show λ-dependent changes
                    (increasing delta_v with λ) while random embeddings
                    should show minimal or inconsistent changes.
    """
    print_header("CREATING VISUALIZATION")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, t_val in enumerate([0.1, 0.5, 0.9]):
        ax = axes[idx]
        df_t = df[df['t'] == t_val]

        # Group by lambda and condition
        true_data = df_t[df_t['condition'] == 'true'].groupby('lambda')['delta_v_mean'].mean()
        rand_data = df_t[df_t['condition'] == 'random'].groupby('lambda')['delta_v_mean'].mean()

        # Bar plot
        x = np.arange(len(true_data))
        width = 0.35
        ax.bar(x - width/2, true_data, width, label='True ESM', alpha=0.8, color='#1f77b4')
        ax.bar(x + width/2, rand_data, width, label='Random', alpha=0.8, color='#ff7f0e')

        ax.set_xlabel('Lambda (λ)', fontsize=11)
        ax.set_ylabel('Mean Δvelocity', fontsize=11)
        ax.set_title(f't={t_val}', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'{lam:.1f}' for lam in true_data.index])
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')

    plt.suptitle('EGNN Sensitivity Test: True ESM vs Random Embeddings',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print_success(f"Saved visualization to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Experiment A: EGNN Sensitivity Test"
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='thesis_work/experiments/day5_film_finetuning/outputs/film-v15/checkpoints/last.ckpt',
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--test-npz',
        type=str,
        default='data/real_testing_dataset_10_tests/test.npz',
        help='Path to test dataset NPZ file'
    )
    parser.add_argument(
        '--test-esmc',
        type=str,
        default='data/real_testing_dataset_10_tests/test_esmc.npz',
        help='Path to test ESM-C embeddings NPZ file'
    )
    parser.add_argument(
        '--pca-model',
        type=str,
        default='thesis_work/experiments/day6_pca_projection/pca_model_32d.pkl',
        help='Path to PCA model pickle file'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        help='Device to run on (cpu or cuda)'
    )
    parser.add_argument(
        '--max-batches',
        type=int,
        default=10,
        help='Maximum number of batches to process'
    )
    parser.add_argument(
        '--no-plot',
        action='store_true',
        help='Skip visualization generation'
    )

    args = parser.parse_args()

    # Print experiment header
    print_header("EXPERIMENT A: EGNN SENSITIVITY TEST")
    print("Testing whether ESM-PCA steering mechanistically affects EGNN predictions")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Test data: {args.test_npz}")
    print(f"ESM-C data: {args.test_esmc}")
    print(f"PCA model: {args.pca_model}")
    print(f"Device: {args.device}")
    print(f"Max batches: {args.max_batches}")

    # ============================================================================
    # SETUP: Load model, enable PCA, load dataset
    # ============================================================================
    print_header("LOADING MODEL AND DATA")

    # Load checkpoint (strict=False to ignore FiLM network weights if present)
    print("Loading checkpoint...")
    model = LigandPocketDDPM.load_from_checkpoint(
        args.checkpoint,
        map_location=args.device,
        strict=False
    )
    model.eval()
    model.to(args.device)
    print_success(f"Loaded checkpoint: {args.checkpoint}")

    # Access dynamics
    dyn = model.ddpm.dynamics
    print_success("Accessed dynamics module")

    # Enable PCA mode
    dyn.use_pca = True
    print_success("Enabled PCA mode")

    # Load PCA model
    print(f"Loading PCA model from: {args.pca_model}")
    with open(args.pca_model, 'rb') as f:
        dyn.pca_model = pickle.load(f)
    print_success(f"Loaded PCA model: {dyn.pca_model.n_components_}D projection")

    # Load dataset
    print("Loading test dataset...")
    test_dataset = ProcessedLigandPocketDataset(
        args.test_npz,
        esmc_path=args.test_esmc,
        center=True
    )
    dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=ProcessedLigandPocketDataset.collate_fn
    )
    print_success(f"Loaded test dataset: {len(test_dataset)} samples")

    # ============================================================================
    # MAIN LOOP: Test sensitivity across λ values and embedding conditions
    # ============================================================================
    print_header("RUNNING SENSITIVITY TESTS")

    t_values = [0.1, 0.5, 0.9]  # Fixed diffusion timesteps
    lambda_values = [1.0, 2.0, 4.0]  # λ=0 is baseline (computed separately)

    results = []
    first_batch = True

    print(f"Testing {args.max_batches} batches × {len(t_values)} timesteps × {len(lambda_values)} λ values × 2 conditions")
    print(f"Expected total measurements: {args.max_batches * len(t_values) * len(lambda_values) * 2}\n")

    for batch_idx, batch in enumerate(tqdm(dataloader, total=args.max_batches, desc="Processing batches")):
        if batch_idx >= args.max_batches:
            break

        # Debug: Print batch keys on first iteration
        if first_batch:
            print(f"\nBatch keys: {list(batch.keys())}")
            first_batch = False

        try:
            # Extract pocket data
            pocket_coords = batch["pocket_coords"].to(args.device).float()
            pocket_one_hot = batch["pocket_one_hot"].to(args.device).float()
            xh_residues = torch.cat([pocket_coords, pocket_one_hot], dim=1)
            mask_residues = batch["pocket_mask"].to(args.device).long()

            # Extract ligand data
            ligand_coords = batch["lig_coords"].to(args.device).float()
            ligand_one_hot = batch["lig_one_hot"].to(args.device).float()
            xh_atoms = torch.cat([ligand_coords, ligand_one_hot], dim=1)
            mask_atoms = batch["lig_mask"].to(args.device).long()

            # Extract ESM-C embeddings
            pocket_emb = batch["pocket_emb"].to(args.device).float()

            # Count atoms
            n_lig_atoms = len(mask_atoms)

            # Create matched random embedding
            # Match global statistics of true embedding across all dimensions
            mu = pocket_emb.mean()  # scalar
            sigma = pocket_emb.std() + 1e-6  # scalar
            rand_emb = mu + sigma * torch.randn_like(pocket_emb)

            # Test each timestep
            for t_val in t_values:
                # Convert to tensor on device
                t = torch.tensor([t_val], device=args.device, dtype=torch.float32)

                # ----------------------------------------------------------------
                # TRUE EMBEDDING TESTS
                # ----------------------------------------------------------------

                # Baseline (λ=0)
                v0, h0 = run_dyn(dyn, xh_atoms, xh_residues, t, mask_atoms,
                                mask_residues, pocket_emb, lam=0.0)

                # Test each λ with true embedding
                for lam in lambda_values:
                    v_lam, h_lam = run_dyn(dyn, xh_atoms, xh_residues, t, mask_atoms,
                                          mask_residues, pocket_emb, lam=lam)

                    # Per-atom deltas
                    delta_v = torch.norm(v_lam - v0, dim=1)  # [n_atoms]
                    delta_h = torch.norm(h_lam - h0, dim=1)  # [n_atoms]

                    results.append({
                        'batch_idx': batch_idx,
                        't': t_val,
                        'lambda': lam,
                        'condition': 'true',
                        'n_lig_atoms': n_lig_atoms,
                        'delta_v_mean': delta_v.mean().item(),
                        'delta_v_median': delta_v.median().item(),
                        'delta_h_mean': delta_h.mean().item(),
                        'delta_h_median': delta_h.median().item(),
                    })

                # ----------------------------------------------------------------
                # RANDOM EMBEDDING TESTS (control)
                # ----------------------------------------------------------------

                # Baseline with random embedding (λ=0)
                v0_rand, h0_rand = run_dyn(dyn, xh_atoms, xh_residues, t, mask_atoms,
                                          mask_residues, pocket_emb, lam=0.0,
                                          pocket_emb_override=rand_emb)

                # Test each λ with random embedding
                for lam in lambda_values:
                    v_lam_rand, h_lam_rand = run_dyn(dyn, xh_atoms, xh_residues, t,
                                                    mask_atoms, mask_residues, pocket_emb,
                                                    lam=lam, pocket_emb_override=rand_emb)

                    delta_v = torch.norm(v_lam_rand - v0_rand, dim=1)
                    delta_h = torch.norm(h_lam_rand - h0_rand, dim=1)

                    results.append({
                        'batch_idx': batch_idx,
                        't': t_val,
                        'lambda': lam,
                        'condition': 'random',
                        'n_lig_atoms': n_lig_atoms,
                        'delta_v_mean': delta_v.mean().item(),
                        'delta_v_median': delta_v.median().item(),
                        'delta_h_mean': delta_h.mean().item(),
                        'delta_h_median': delta_h.median().item(),
                    })

        except Exception as e:
            print(f"\nWarning: Error processing batch {batch_idx}: {e}")
            import traceback
            traceback.print_exc()
            continue

    if not results:
        print("\nERROR: No results collected!")
        return

    print(f"\n✓ Collected {len(results)} measurements")

    # ============================================================================
    # SAVE RESULTS
    # ============================================================================
    print_header("SAVING RESULTS")

    # Create output directory
    output_dir = Path(__file__).parent
    output_dir.mkdir(exist_ok=True)

    # Save CSV
    df = pd.DataFrame(results)
    csv_path = output_dir / "expA_results.csv"
    df.to_csv(csv_path, index=False)
    print_success(f"Saved results to {csv_path}")

    # Print summary statistics
    print("\nSummary Statistics:")
    print(df.groupby(['condition', 'lambda'])[['delta_v_mean', 'delta_h_mean']].mean().round(4))

    # Create visualization
    if not args.no_plot:
        plot_path = output_dir / "expA_sensitivity_plot.png"
        create_visualization(df, plot_path)

    # ============================================================================
    # INTERPRETATION
    # ============================================================================
    print_header("INTERPRETATION")

    print("Expected Results:")
    print("  ✓ True ESM embeddings: Δv and Δh should increase monotonically with λ")
    print("  ✓ Random embeddings: Changes should be smaller and/or inconsistent")
    print("\nIf ESM-PCA steering is mechanistically active:")
    print("  - True embeddings show λ-dependent signal")
    print("  - Random embeddings show weaker/random signal")
    print("\nAnalysis:")

    # Compare true vs random for each lambda
    for lam in lambda_values:
        true_mean = df[(df['condition'] == 'true') & (df['lambda'] == lam)]['delta_v_mean'].mean()
        rand_mean = df[(df['condition'] == 'random') & (df['lambda'] == lam)]['delta_v_mean'].mean()
        ratio = true_mean / (rand_mean + 1e-8)

        print(f"  λ={lam:.1f}: True={true_mean:.4f}, Random={rand_mean:.4f}, Ratio={ratio:.2f}x")

    print("\n" + "=" * 80)
    print("Experiment complete!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
