"""
Experiment A2: Denoising Accuracy Test

Goal: Measure whether global ESM-PCA conditioning improves denoising accuracy
      on real test complexes.

Approach:
- Create noised ligand states x_t from ground truth x_0 using forward diffusion
- Run denoising with λ=[0, 1, 2] using true vs shuffled ESM embeddings
- Compute ε-MSE (epsilon mean squared error) between predicted and true noise
- Compare: Does true ESM + λ reduce error vs λ=0 and vs shuffled ESM?

Outputs:
- CSV: batch_idx, t, lambda, condition (true/shuffled), n_lig_atoms,
       eps_mse_mean, eps_mse_median
- Paired summaries showing improvement or degradation
- Decision: Proceed to docking pilot OR conclude ESM needs learned adapter

Usage:
    python expA2_denoising_accuracy.py
    python expA2_denoising_accuracy.py --checkpoint path/to/model.ckpt
    python expA2_denoising_accuracy.py --max-batches 5
"""

import torch
import numpy as np
import pandas as pd
import pickle
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


def create_noised_ligand(model, xh0_lig, xh0_pocket, lig_mask, pocket_mask, t_val):
    """
    Create noised ligand state using forward diffusion.

    Formula: z_t = alpha_t * x_0 + sigma_t * eps

    Args:
        model: LigandPocketDDPM model
        xh0_lig: ground truth ligand [n_atoms, 3+atom_nf]
        xh0_pocket: ground truth pocket [n_residues, 3+residue_nf]
        lig_mask: ligand batch indices [n_atoms]
        pocket_mask: pocket batch indices [n_residues]
        t_val: timestep value in [0, 1]

    Returns:
        z_t_lig: noised ligand [n_atoms, 3+atom_nf]
        xh_pocket: processed pocket (COM-removed) [n_residues, 3+residue_nf]
        eps_lig: sampled noise [n_atoms, 3+atom_nf]
        t: normalized timestep tensor
    """
    device = xh0_lig.device

    # Normalize timestep to [0, 1]
    t = torch.tensor([[t_val]], device=device, dtype=torch.float32)

    # Compute gamma_t (noise schedule)
    gamma_t = model.ddpm.gamma(t)
    gamma_t = model.ddpm.inflate_batch_array(gamma_t, xh0_lig)

    # Use model's noised_representation method (from conditional_model.py:183-205)
    z_t_lig, xh_pocket, eps_lig = model.ddpm.noised_representation(
        xh0_lig, xh0_pocket, lig_mask, pocket_mask, gamma_t
    )

    return z_t_lig, xh_pocket, eps_lig, t


def compute_denoising_error(dyn, z_t_lig, xh_pocket, t, lig_mask, pocket_mask,
                           eps_true, pocket_emb, lam):
    """
    Run denoising and compute ε-MSE.

    Args:
        dyn: EGNNDynamics module
        z_t_lig: noised ligand state [n_atoms, 3+atom_nf]
        xh_pocket: processed pocket [n_residues, 3+residue_nf]
        t: normalized timestep tensor
        lig_mask: ligand batch indices [n_atoms]
        pocket_mask: pocket batch indices [n_residues]
        eps_true: true noise that was added [n_atoms, 3+atom_nf]
        pocket_emb: ESM-C embedding [batch_size, 960]
        lam: PCA lambda value

    Returns:
        eps_mse_mean: mean per-atom ε-MSE
        eps_mse_median: median per-atom ε-MSE
    """
    # Set lambda
    original_lambda = dyn.pca_lambda
    dyn.pca_lambda = lam

    # Run dynamics to predict noise
    with torch.no_grad():
        net_out_lig, _ = dyn(z_t_lig, xh_pocket, t, lig_mask, pocket_mask,
                            pocket_emb=pocket_emb)

    # Restore lambda
    dyn.pca_lambda = original_lambda

    # Compute per-atom squared error
    # net_out_lig is the predicted epsilon (noise)
    squared_error = (eps_true - net_out_lig) ** 2  # [n_atoms, 3+atom_nf]

    # Sum over dimensions (coords + features) to get per-atom error
    per_atom_mse = squared_error.sum(dim=1)  # [n_atoms]

    return per_atom_mse.mean().item(), per_atom_mse.median().item()


def main():
    parser = argparse.ArgumentParser(
        description="Experiment A2: Denoising Accuracy Test"
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

    args = parser.parse_args()

    # Print experiment header
    print_header("EXPERIMENT A2: DENOISING ACCURACY TEST")
    print("Testing whether ESM-PCA conditioning improves denoising on real test data")
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
        center=True  # Will be re-centered later
    )
    dataloader = DataLoader(
        test_dataset,
        batch_size=1,  # Must be 1 for proper shuffling
        shuffle=False,
        collate_fn=ProcessedLigandPocketDataset.collate_fn
    )
    print_success(f"Loaded test dataset: {len(test_dataset)} samples")

    # ============================================================================
    # PHASE 1: Collect all embeddings and create shuffled version
    # ============================================================================
    print_header("PHASE 1: COLLECTING EMBEDDINGS")

    all_pocket_embs = []
    all_batches = []

    print(f"Collecting {args.max_batches} batches...")
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= args.max_batches:
            break
        all_batches.append(batch)
        all_pocket_embs.append(batch["pocket_emb"])

    print_success(f"Collected {len(all_batches)} batches")

    # Create shuffled embeddings
    shuffled_indices = torch.randperm(len(all_pocket_embs))
    shuffled_embs = [all_pocket_embs[i] for i in shuffled_indices]
    print_success(f"Created shuffled embeddings (permutation: {shuffled_indices[:5].tolist()}...)")

    # ============================================================================
    # PHASE 2: Run denoising tests
    # ============================================================================
    print_header("PHASE 2: RUNNING DENOISING TESTS")

    t_values = [0.1, 0.5, 0.9]  # Fixed diffusion timesteps
    lambda_values = [0, 1, 2]  # Test λ=0 baseline vs λ=[1,2]

    print(f"Testing {len(all_batches)} batches × {len(t_values)} timesteps × {len(lambda_values)} λ values × 2 conditions")
    print(f"Expected total measurements: {len(all_batches) * len(t_values) * len(lambda_values) * 2}\n")

    results = []

    for batch_idx, batch in enumerate(tqdm(all_batches, desc="Processing batches")):
        try:
            # Extract ground truth ligand and pocket
            lig_coords = batch["lig_coords"].to(args.device).float()
            lig_one_hot = batch["lig_one_hot"].to(args.device).float()
            xh0_lig = torch.cat([lig_coords, lig_one_hot], dim=1)
            lig_mask = batch["lig_mask"].to(args.device).long()

            pocket_coords = batch["pocket_coords"].to(args.device).float()
            pocket_one_hot = batch["pocket_one_hot"].to(args.device).float()
            xh0_pocket = torch.cat([pocket_coords, pocket_one_hot], dim=1)
            pocket_mask = batch["pocket_mask"].to(args.device).long()

            # Get embeddings
            true_emb = batch["pocket_emb"].to(args.device).float()
            shuffled_emb = shuffled_embs[batch_idx].to(args.device).float()

            # Center data (remove mean) - critical for equivariance
            xh0_lig[:, :3], xh0_pocket[:, :3] = model.ddpm.remove_mean_batch(
                xh0_lig[:, :3], xh0_pocket[:, :3], lig_mask, pocket_mask
            )

            n_lig_atoms = len(lig_mask)

            for t_val in t_values:
                # Create noised ligand state
                z_t_lig, xh_pocket, eps_lig, t = create_noised_ligand(
                    model, xh0_lig, xh0_pocket, lig_mask, pocket_mask, t_val
                )

                # ----------------------------------------------------------------
                # Test with TRUE embedding
                # ----------------------------------------------------------------
                for lam in lambda_values:
                    eps_mse_mean, eps_mse_median = compute_denoising_error(
                        dyn, z_t_lig, xh_pocket, t, lig_mask, pocket_mask,
                        eps_lig, true_emb, lam
                    )

                    results.append({
                        'batch_idx': batch_idx,
                        't': t_val,
                        'lambda': lam,
                        'condition': 'true',
                        'n_lig_atoms': n_lig_atoms,
                        'eps_mse_mean': eps_mse_mean,
                        'eps_mse_median': eps_mse_median,
                    })

                # ----------------------------------------------------------------
                # Test with SHUFFLED embedding
                # ----------------------------------------------------------------
                for lam in lambda_values:
                    eps_mse_mean, eps_mse_median = compute_denoising_error(
                        dyn, z_t_lig, xh_pocket, t, lig_mask, pocket_mask,
                        eps_lig, shuffled_emb, lam
                    )

                    results.append({
                        'batch_idx': batch_idx,
                        't': t_val,
                        'lambda': lam,
                        'condition': 'shuffled',
                        'n_lig_atoms': n_lig_atoms,
                        'eps_mse_mean': eps_mse_mean,
                        'eps_mse_median': eps_mse_median,
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
    csv_path = output_dir / "expA2_results.csv"
    df.to_csv(csv_path, index=False)
    print_success(f"Saved results to {csv_path}")

    # ============================================================================
    # ANALYSIS AND INTERPRETATION
    # ============================================================================
    print_header("PAIRED COMPARISONS")

    print("\n1. Effect of λ (within each condition):")
    for cond in ['true', 'shuffled']:
        print(f"\n  {cond.upper()}:")
        lam0 = df[(df['condition'] == cond) & (df['lambda'] == 0)]['eps_mse_mean'].mean()
        lam1 = df[(df['condition'] == cond) & (df['lambda'] == 1)]['eps_mse_mean'].mean()
        lam2 = df[(df['condition'] == cond) & (df['lambda'] == 2)]['eps_mse_mean'].mean()

        print(f"    λ=0: {lam0:.6f}")
        print(f"    λ=1: {lam1:.6f} (Δ={lam1-lam0:+.6f}, {(lam1-lam0)/lam0*100:+.2f}%)")
        print(f"    λ=2: {lam2:.6f} (Δ={lam2-lam0:+.6f}, {(lam2-lam0)/lam0*100:+.2f}%)")

    print("\n2. True vs Shuffled (for each λ):")
    for lam in [0, 1, 2]:
        true_mse = df[(df['condition'] == 'true') & (df['lambda'] == lam)]['eps_mse_mean'].mean()
        shuf_mse = df[(df['condition'] == 'shuffled') & (df['lambda'] == lam)]['eps_mse_mean'].mean()
        diff = true_mse - shuf_mse
        pct = (diff / shuf_mse * 100)

        print(f"\n  λ={lam}:")
        print(f"    True:     {true_mse:.6f}")
        print(f"    Shuffled: {shuf_mse:.6f}")
        print(f"    Δ:        {diff:+.6f} ({pct:+.2f}%)")

        if diff < 0:
            print(f"    → True ESM IMPROVES denoising by {abs(pct):.2f}%")
        else:
            print(f"    → True ESM WORSENS denoising by {abs(pct):.2f}%")

    print_header("DECISION")

    # Check if true ESM + λ>0 reduces error vs λ=0 and vs shuffled
    true_lam0 = df[(df['condition'] == 'true') & (df['lambda'] == 0)]['eps_mse_mean'].mean()
    true_lam1 = df[(df['condition'] == 'true') & (df['lambda'] == 1)]['eps_mse_mean'].mean()
    true_lam2 = df[(df['condition'] == 'true') & (df['lambda'] == 2)]['eps_mse_mean'].mean()

    shuf_lam1 = df[(df['condition'] == 'shuffled') & (df['lambda'] == 1)]['eps_mse_mean'].mean()
    shuf_lam2 = df[(df['condition'] == 'shuffled') & (df['lambda'] == 2)]['eps_mse_mean'].mean()

    print("Criteria:")
    print(f"  1. True λ=1 < True λ=0: {true_lam1 < true_lam0} ({true_lam1:.6f} < {true_lam0:.6f})")
    print(f"  2. True λ=1 < Shuffled λ=1: {true_lam1 < shuf_lam1} ({true_lam1:.6f} < {shuf_lam1:.6f})")
    print(f"  3. True λ=2 < True λ=0: {true_lam2 < true_lam0} ({true_lam2:.6f} < {true_lam0:.6f})")
    print(f"  4. True λ=2 < Shuffled λ=2: {true_lam2 < shuf_lam2} ({true_lam2:.6f} < {shuf_lam2:.6f})")

    print("\nFinal Decision:")
    if (true_lam1 < true_lam0) and (true_lam1 < shuf_lam1):
        improvement = ((true_lam0 - true_lam1) / true_lam0 * 100)
        print(f"  ✓ TRUE ESM + λ=1 consistently improves denoising by {improvement:.2f}%")
        print(f"  → Proceed to small docking pilot with λ=1")
    elif (true_lam2 < true_lam0) and (true_lam2 < shuf_lam2):
        improvement = ((true_lam0 - true_lam2) / true_lam0 * 100)
        print(f"  ✓ TRUE ESM + λ=2 consistently improves denoising by {improvement:.2f}%")
        print(f"  → Proceed to small docking pilot with λ=2")
    else:
        print(f"  ✗ ESM-PCA does NOT improve denoising accuracy")
        print(f"  → Evidence that inference-time ESM adds limited usable signal")
        print(f"  → May need learned adapter (FiLM network) instead of PCA")

    print("\n" + "=" * 80)
    print("Experiment complete!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
