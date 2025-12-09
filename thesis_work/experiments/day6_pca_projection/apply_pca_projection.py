"""
Apply Trained PCA Projection to New Data

This script loads a pre-trained PCA model and applies it to new embeddings
(e.g., test set, validation set).

Usage:
    python apply_pca_projection.py --pca-model pca_model_128d.pkl --data-path test_esmc.npz
    python apply_pca_projection.py --pca-model pca_model_64d.pkl --data-path test_esmc.npz --output projected_test.npz
"""

import numpy as np
import pickle
import argparse
from pathlib import Path


def load_pca_model(model_path):
    """Load a trained PCA model from pickle file."""
    print(f"Loading PCA model from {model_path}")
    with open(model_path, 'rb') as f:
        pca = pickle.load(f)

    print(f"  Model info: {pca.n_components_} components")
    print(f"  Variance explained: {pca.explained_variance_ratio_.sum() * 100:.2f}%")

    return pca


def load_embeddings(data_path):
    """Load embeddings from npz file."""
    print(f"\nLoading embeddings from {data_path}")
    data = np.load(data_path, allow_pickle=True)

    embeddings = data['embeddings']  # Shape: (N, 960)

    # Load additional data if available
    extra_data = {}
    if 'names' in data:
        extra_data['names'] = data['names']
    if 'sequences' in data:
        extra_data['sequences'] = data['sequences']

    print(f"  Loaded {embeddings.shape[0]} embeddings of dimension {embeddings.shape[1]}")

    return embeddings, extra_data


def apply_pca_transform(pca, embeddings):
    """Apply PCA transformation to embeddings."""
    print(f"\nApplying PCA transformation...")

    # Transform: reduce dimensionality
    transformed = pca.transform(embeddings)

    print(f"  Original shape: {embeddings.shape}")
    print(f"  Projected shape: {transformed.shape}")
    print(f"  Compression ratio: {embeddings.shape[1] / transformed.shape[1]:.1f}x")

    return transformed


def evaluate_reconstruction(pca, embeddings, transformed):
    """Evaluate reconstruction quality."""
    print(f"\nEvaluating reconstruction quality...")

    # Reconstruct from projected embeddings
    reconstructed = pca.inverse_transform(transformed)

    # Calculate metrics
    mse = np.mean((embeddings - reconstructed) ** 2)
    mae = np.mean(np.abs(embeddings - reconstructed))

    # Cosine similarity
    embeddings_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
    reconstructed_norm = reconstructed / (np.linalg.norm(reconstructed, axis=1, keepdims=True) + 1e-8)
    cosine_sim = np.mean(np.sum(embeddings_norm * reconstructed_norm, axis=1))

    print(f"  MSE: {mse:.6f}")
    print(f"  MAE: {mae:.6f}")
    print(f"  Cosine Similarity: {cosine_sim:.6f}")

    return {
        'mse': mse,
        'mae': mae,
        'cosine_similarity': cosine_sim,
        'reconstructed': reconstructed
    }


def save_projected_embeddings(output_path, transformed, extra_data):
    """Save projected embeddings to npz file."""
    print(f"\nSaving projected embeddings to {output_path}")

    # Prepare data to save
    save_dict = {'embeddings': transformed}
    save_dict.update(extra_data)

    np.savez(output_path, **save_dict)
    print(f"  Saved {transformed.shape[0]} embeddings of dimension {transformed.shape[1]}")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Apply trained PCA projection to new embeddings',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--pca-model',
        type=str,
        required=True,
        help='Path to trained PCA model (.pkl file)'
    )

    parser.add_argument(
        '--data-path',
        type=str,
        required=True,
        help='Path to embeddings to transform (.npz file)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output path for projected embeddings (default: <input>_projected_<n>d.npz)'
    )

    parser.add_argument(
        '--evaluate',
        action='store_true',
        help='Evaluate reconstruction quality'
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 80)
    print("Apply PCA Projection to Embeddings")
    print("=" * 80)

    # Load PCA model
    pca = load_pca_model(args.pca_model)

    # Load embeddings
    embeddings, extra_data = load_embeddings(args.data_path)

    # Check dimensions match
    if embeddings.shape[1] != pca.components_.shape[1]:
        raise ValueError(
            f"Dimension mismatch: embeddings have {embeddings.shape[1]} dimensions "
            f"but PCA model expects {pca.components_.shape[1]} dimensions"
        )

    # Apply PCA transformation
    transformed = apply_pca_transform(pca, embeddings)

    # Evaluate reconstruction quality if requested
    if args.evaluate:
        evaluation = evaluate_reconstruction(pca, embeddings, transformed)

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        input_path = Path(args.data_path)
        n_components = pca.n_components_
        output_path = input_path.parent / f"{input_path.stem}_projected_{n_components}d.npz"

    # Save projected embeddings
    save_projected_embeddings(output_path, transformed, extra_data)

    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Input: {args.data_path}")
    print(f"PCA Model: {args.pca_model} ({pca.n_components_} components)")
    print(f"Output: {output_path}")
    print(f"Projection: {embeddings.shape[1]}D → {transformed.shape[1]}D")
    print("=" * 80)


if __name__ == "__main__":
    main()
