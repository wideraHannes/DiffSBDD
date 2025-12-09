"""
PCA-based Projection for ESM-C Embeddings

Alternative to FiLM-based learned projection: use PCA to reduce ESM-C embeddings
from 960 dimensions to a lower dimension suitable for the diffusion model.

This script:
1. Loads ESM-C embeddings from the training set
2. Fits PCA with different numbers of components
3. Analyzes projection quality (variance explained, reconstruction error)
4. Visualizes the results
5. Saves the PCA model for use in training/inference

Usage:
    python pca_projection_analysis.py --save-models 32 64 128 256
    python pca_projection_analysis.py --save-models 128 --max-components 512
    python pca_projection_analysis.py --data-path /path/to/data.npz --save-models 64 128
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from pathlib import Path
import pickle
import argparse


def load_esmc_embeddings(data_path):
    """Load ESM-C embeddings from npz file."""
    print(f"Loading embeddings from {data_path}")
    data = np.load(data_path, allow_pickle=True)

    embeddings = data["embeddings"]  # Shape: (N, 960)
    names = data["names"]

    print(f"Loaded {embeddings.shape[0]} embeddings of dimension {embeddings.shape[1]}")
    return embeddings, names


def analyze_pca_components(embeddings, max_components=512):
    """Analyze how many components are needed to explain variance."""
    print(f"\nFitting PCA with up to {max_components} components...")

    # Fit PCA with all components up to max
    pca_full = PCA(
        n_components=min(max_components, embeddings.shape[0], embeddings.shape[1])
    )
    pca_full.fit(embeddings)

    # Get variance explained
    variance_explained = pca_full.explained_variance_ratio_
    cumulative_variance = np.cumsum(variance_explained)

    # Print key statistics
    print("\nVariance Explained by Number of Components:")
    print(f"  (Maximum available: {len(cumulative_variance)} components)")
    for n in [32, 64, 128, 256, 512]:
        if n <= len(cumulative_variance):
            print(f"  {n:3d} components: {cumulative_variance[n - 1] * 100:.2f}%")

    return pca_full, variance_explained, cumulative_variance


def evaluate_reconstruction(embeddings, pca_model, n_components_list):
    """Evaluate reconstruction error for different numbers of components."""
    print("\nEvaluating reconstruction quality...")

    results = {}
    max_available = pca_model.n_components_

    for n_comp in n_components_list:
        # Skip if requesting more components than available
        if n_comp > max_available:
            print(f"\n  Skipping {n_comp} components (only {max_available} available)")
            continue

        # Use the already-fitted PCA model's components
        # Transform using first n_comp components
        transformed = embeddings @ pca_model.components_[:n_comp].T
        # Inverse transform: reconstruct from reduced dimensions
        reconstructed = transformed @ pca_model.components_[:n_comp] + pca_model.mean_

        # Calculate reconstruction error
        mse = np.mean((embeddings - reconstructed) ** 2)
        mae = np.mean(np.abs(embeddings - reconstructed))
        relative_error = np.mean(
            np.abs(embeddings - reconstructed) / (np.abs(embeddings) + 1e-8)
        )

        # Calculate cosine similarity
        embeddings_norm = embeddings / (
            np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
        )
        reconstructed_norm = reconstructed / (
            np.linalg.norm(reconstructed, axis=1, keepdims=True) + 1e-8
        )
        cosine_sim = np.mean(np.sum(embeddings_norm * reconstructed_norm, axis=1))

        results[n_comp] = {
            "mse": mse,
            "mae": mae,
            "relative_error": relative_error,
            "cosine_similarity": cosine_sim,
            "compression_ratio": embeddings.shape[1] / n_comp,
        }

        print(
            f"\n  {n_comp} components (compression {results[n_comp]['compression_ratio']:.1f}x):"
        )
        print(f"    MSE: {mse:.6f}")
        print(f"    MAE: {mae:.6f}")
        print(f"    Relative Error: {relative_error:.4f}")
        print(f"    Cosine Similarity: {cosine_sim:.6f}")

    return results


def plot_analysis(
    variance_explained, cumulative_variance, reconstruction_results, save_dir
):
    """Create visualization plots for PCA analysis."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Skip plotting if no reconstruction results
    if not reconstruction_results:
        print("\nSkipping plots - no reconstruction results available (dataset too small)")
        return

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Plot 1: Variance explained per component
    ax = axes[0, 0]
    n_show = min(100, len(variance_explained))
    ax.plot(range(1, n_show + 1), variance_explained[:n_show] * 100)
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Variance Explained (%)")
    ax.set_title("Variance Explained by Each Component")
    ax.grid(True, alpha=0.3)

    # Plot 2: Cumulative variance
    ax = axes[0, 1]
    ax.plot(range(1, len(cumulative_variance) + 1), cumulative_variance * 100)
    ax.axhline(y=90, color="r", linestyle="--", label="90% variance")
    ax.axhline(y=95, color="g", linestyle="--", label="95% variance")
    ax.axhline(y=99, color="b", linestyle="--", label="99% variance")
    ax.set_xlabel("Number of Components")
    ax.set_ylabel("Cumulative Variance Explained (%)")
    ax.set_title("Cumulative Variance Explained")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Reconstruction error vs components
    ax = axes[1, 0]
    n_comps = sorted(reconstruction_results.keys())
    mses = [reconstruction_results[n]["mse"] for n in n_comps]
    maes = [reconstruction_results[n]["mae"] for n in n_comps]

    ax.plot(n_comps, mses, "o-", label="MSE", linewidth=2, markersize=8)
    ax.plot(n_comps, maes, "s-", label="MAE", linewidth=2, markersize=8)
    ax.set_xlabel("Number of Components")
    ax.set_ylabel("Reconstruction Error")
    ax.set_title("Reconstruction Error vs Number of Components")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")
    ax.set_yscale("log")

    # Plot 4: Cosine similarity vs components
    ax = axes[1, 1]
    cosine_sims = [reconstruction_results[n]["cosine_similarity"] for n in n_comps]

    ax.plot(n_comps, cosine_sims, "o-", linewidth=2, markersize=8, color="purple")
    ax.axhline(y=0.95, color="r", linestyle="--", alpha=0.5, label="0.95")
    ax.axhline(y=0.99, color="g", linestyle="--", alpha=0.5, label="0.99")
    ax.set_xlabel("Number of Components")
    ax.set_ylabel("Cosine Similarity")
    ax.set_title("Reconstruction Quality (Cosine Similarity)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.9, 1.0])

    plt.tight_layout()
    plt.savefig(save_dir / "pca_analysis.png", dpi=300, bbox_inches="tight")
    print(f"\nSaved analysis plot to {save_dir / 'pca_analysis.png'}")
    plt.close()


def save_pca_model(pca_model, n_components, save_path):
    """Save PCA model for later use."""
    pca_to_save = PCA(n_components=n_components)
    pca_to_save.fit_transform = pca_model.transform
    pca_to_save.components_ = pca_model.components_[:n_components]
    pca_to_save.mean_ = pca_model.mean_
    pca_to_save.explained_variance_ = pca_model.explained_variance_[:n_components]
    pca_to_save.explained_variance_ratio_ = pca_model.explained_variance_ratio_[
        :n_components
    ]
    pca_to_save.n_components_ = n_components

    with open(save_path, "wb") as f:
        pickle.dump(pca_to_save, f)

    print(f"\nSaved PCA model ({n_components} components) to {save_path}")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="PCA projection analysis for ESM-C embeddings",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--save-models",
        type=int,
        nargs="+",
        default=[64, 128, 256],
        help="Dimensions to save as .pkl models (e.g., --save-models 32 64 128)",
    )

    parser.add_argument(
        "--data-path",
        type=str,
        default="/Users/hanneswidera/Uni/Master/thesis/DiffSBDD/data/small_dataset_1000/train_esmc.npz",
        help="Path to ESM-C embeddings npz file",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results (default: script directory)",
    )

    parser.add_argument(
        "--max-components",
        type=int,
        default=512,
        help="Maximum components to calculate in full PCA (limited by data dimensions)",
    )

    parser.add_argument(
        "--analysis-dims",
        type=int,
        nargs="+",
        default=[16, 32, 64, 128, 256, 512],
        help="Dimensions to show detailed reconstruction analysis for",
    )

    return parser.parse_args()


def main():
    # Parse arguments
    args = parse_args()

    # Configuration
    data_path = Path(args.data_path)
    output_dir = Path(args.output_dir) if args.output_dir else Path(__file__).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    models_to_save = sorted(args.save_models)
    analysis_dims = sorted(args.analysis_dims)
    max_components = args.max_components

    print("=" * 80)
    print("PCA-Based Projection Analysis for ESM-C Embeddings")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Data path: {data_path}")
    print(f"  Output dir: {output_dir}")
    print(f"  Max components to calculate: {max_components}")
    print(f"  Dimensions for detailed analysis: {analysis_dims}")
    print(f"  Models to save: {models_to_save}")

    # Load embeddings
    embeddings, names = load_esmc_embeddings(data_path)

    # Calculate full PCA with all components (this is done once!)
    print(f"\nCalculating full PCA with up to {max_components} components...")
    pca_full, variance_explained, cumulative_variance = analyze_pca_components(
        embeddings, max_components=max_components
    )

    # Evaluate reconstruction quality for selected dimensions
    # (These use the already-calculated PCA components)
    reconstruction_results = evaluate_reconstruction(
        embeddings, pca_full, analysis_dims
    )

    max_available = pca_full.n_components_

    # Create visualizations
    plot_analysis(
        variance_explained, cumulative_variance, reconstruction_results, output_dir
    )

    # Save PCA models for specified dimensions
    print("\n" + "=" * 80)
    print("Saving PCA Models")
    print("=" * 80)

    saved_any = False
    for n_comp in models_to_save:
        if n_comp <= max_available:
            save_path = output_dir / f"pca_model_{n_comp}d.pkl"
            save_pca_model(pca_full, n_comp, save_path)
            saved_any = True
        else:
            print(
                f"\nSkipping {n_comp} components (only {max_available} available)"
            )

    if not saved_any:
        print(f"\nNote: No models saved (requested dimensions exceed available {max_available} components)")

    # Summary and recommendations
    print("\n" + "=" * 80)
    print("Summary and Recommendations")
    print("=" * 80)

    # Find components needed for 95% and 99% variance
    idx_95 = np.argmax(cumulative_variance >= 0.95) + 1
    idx_99 = np.argmax(cumulative_variance >= 0.99) + 1

    print(f"\nComponents needed:")
    print(f"  95% variance: {idx_95} components")
    print(f"  99% variance: {idx_99} components")

    print("\nAnalysis for selected dimensions:")
    for n_comp in analysis_dims:
        if n_comp <= len(cumulative_variance) and n_comp in reconstruction_results:
            variance = cumulative_variance[n_comp - 1] * 100
            compression = embeddings.shape[1] / n_comp
            cosine_sim = reconstruction_results[n_comp]["cosine_similarity"]
            saved = "✓ saved" if n_comp in models_to_save else ""
            print(
                f"  - {n_comp:3d} components: {compression:5.1f}x compression, "
                f"{variance:5.2f}% variance, cosine sim={cosine_sim:.6f} {saved}"
            )


if __name__ == "__main__":
    main()
