#!/usr/bin/env python3
"""
Day 2: ESM-C Embedding Signal Analysis

Analyze whether ESM-C embeddings contain meaningful signal for drug design.

Success Criteria:
- Embeddings are not random (show structure)
- Some clustering/patterns visible
- Basic statistics reasonable

Note: Working with 10 test samples (limited by available PDB files)
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.manifold import TSNE
from pathlib import Path

# Setup
sns.set_style("whitegrid")
output_dir = Path("thesis_work/experiments/day2_embeddings")
figures_dir = output_dir / "figures"
figures_dir.mkdir(parents=True, exist_ok=True)

print("="*60)
print("DAY 2: ESM-C EMBEDDING SIGNAL ANALYSIS")
print("="*60)

# Load embeddings
print("\n1. Loading ESM-C embeddings...")
emb_path = "esmc_integration/embeddings_cache/test_esmc_embeddings.npz"
data = np.load(emb_path, allow_pickle=True)

embeddings = data['embeddings']
sequences = data['sequences']
names = data['names']

print(f"   Loaded {len(embeddings)} embeddings")
print(f"   Embedding dimension: {embeddings.shape[1]}")
print(f"   Embedding dtype: {embeddings.dtype}")

# Basic statistics
print("\n2. Embedding Statistics:")
print(f"   Mean: {embeddings.mean():.6f}")
print(f"   Std:  {embeddings.std():.6f}")
print(f"   Min:  {embeddings.min():.6f}")
print(f"   Max:  {embeddings.max():.6f}")

# Check for zero embeddings (failed extractions)
zero_mask = np.all(embeddings == 0, axis=1)
print(f"   Zero embeddings: {zero_mask.sum()}/{len(embeddings)}")

if zero_mask.sum() > 0:
    print("   ⚠️  Some embeddings are all zeros (extraction failures)")
    embeddings = embeddings[~zero_mask]
    sequences = sequences[~zero_mask]
    names = names[~zero_mask]
    print(f"   Working with {len(embeddings)} valid embeddings")

# Sequence statistics
print("\n3. Sequence Statistics:")
seq_lengths = [len(s) for s in sequences]
print(f"   Sequences: {len(sequences)}")
print(f"   Length range: {min(seq_lengths)} - {max(seq_lengths)} residues")
print(f"   Mean length: {np.mean(seq_lengths):.1f} residues")

# Pairwise similarity
print("\n4. Computing Pairwise Similarities...")
cos_sim = cosine_similarity(embeddings)
euc_dist = euclidean_distances(embeddings)

print(f"   Cosine similarity:")
print(f"     Mean: {cos_sim[np.triu_indices_from(cos_sim, k=1)].mean():.4f}")
print(f"     Std:  {cos_sim[np.triu_indices_from(cos_sim, k=1)].std():.4f}")
print(f"     Min:  {cos_sim[np.triu_indices_from(cos_sim, k=1)].min():.4f}")
print(f"     Max:  {cos_sim[np.triu_indices_from(cos_sim, k=1)].max():.4f}")

print(f"   Euclidean distance:")
print(f"     Mean: {euc_dist[np.triu_indices_from(euc_dist, k=1)].mean():.4f}")
print(f"     Std:  {euc_dist[np.triu_indices_from(euc_dist, k=1)].std():.4f}")

# Visualizations
print("\n5. Creating Visualizations...")

# Plot 1: Similarity Matrix
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Cosine similarity
im1 = axes[0].imshow(cos_sim, cmap='RdYlBu_r', vmin=0, vmax=1)
axes[0].set_title(f'Cosine Similarity Matrix ({len(embeddings)} samples)', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Sample Index')
axes[0].set_ylabel('Sample Index')
plt.colorbar(im1, ax=axes[0], label='Similarity')

# Euclidean distance
im2 = axes[1].imshow(euc_dist, cmap='viridis')
axes[1].set_title(f'Euclidean Distance Matrix ({len(embeddings)} samples)', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Sample Index')
axes[1].set_ylabel('Sample Index')
plt.colorbar(im2, ax=axes[1], label='Distance')

plt.tight_layout()
plt.savefig(figures_dir / 'similarity_matrices.png', dpi=300, bbox_inches='tight')
print(f"   ✓ Saved: similarity_matrices.png")
plt.close()

# Plot 2: Embedding Distribution
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Per-dimension statistics
dim_means = embeddings.mean(axis=0)
dim_stds = embeddings.std(axis=0)

axes[0, 0].hist(dim_means, bins=50, edgecolor='black', alpha=0.7)
axes[0, 0].set_title('Distribution of Dimension Means', fontweight='bold')
axes[0, 0].set_xlabel('Mean Value')
axes[0, 0].set_ylabel('Count')
axes[0, 0].axvline(0, color='red', linestyle='--', label='Zero')
axes[0, 0].legend()

axes[0, 1].hist(dim_stds, bins=50, edgecolor='black', alpha=0.7)
axes[0, 1].set_title('Distribution of Dimension Std Devs', fontweight='bold')
axes[0, 1].set_xlabel('Std Dev')
axes[0, 1].set_ylabel('Count')

# Per-sample statistics
sample_means = embeddings.mean(axis=1)
sample_stds = embeddings.std(axis=1)

axes[1, 0].bar(range(len(sample_means)), sample_means, edgecolor='black', alpha=0.7)
axes[1, 0].set_title('Mean Embedding Value per Sample', fontweight='bold')
axes[1, 0].set_xlabel('Sample Index')
axes[1, 0].set_ylabel('Mean Value')
axes[1, 0].axhline(0, color='red', linestyle='--')

axes[1, 1].bar(range(len(sample_stds)), sample_stds, edgecolor='black', alpha=0.7)
axes[1, 1].set_title('Std Dev per Sample', fontweight='bold')
axes[1, 1].set_xlabel('Sample Index')
axes[1, 1].set_ylabel('Std Dev')

plt.tight_layout()
plt.savefig(figures_dir / 'embedding_distributions.png', dpi=300, bbox_inches='tight')
print(f"   ✓ Saved: embedding_distributions.png")
plt.close()

# Plot 3: t-SNE visualization (if enough samples)
if len(embeddings) >= 3:
    print("\n6. Running t-SNE...")
    n_components = min(2, len(embeddings) - 1)
    perplexity = min(5, len(embeddings) - 1)

    if len(embeddings) >= 5:
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        coords_2d = tsne.fit_transform(embeddings)

        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        scatter = ax.scatter(coords_2d[:, 0], coords_2d[:, 1],
                           c=range(len(embeddings)), cmap='tab10',
                           s=200, alpha=0.7, edgecolors='black', linewidth=1.5)

        # Annotate points
        for i, (x, y) in enumerate(coords_2d):
            ax.annotate(f'{i}', (x, y), fontsize=10, ha='center', va='center', fontweight='bold')

        ax.set_title(f't-SNE Visualization of ESM-C Embeddings\n({len(embeddings)} samples, perplexity={perplexity})',
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('t-SNE Component 1', fontsize=12)
        ax.set_ylabel('t-SNE Component 2', fontsize=12)
        ax.grid(True, alpha=0.3)

        plt.colorbar(scatter, ax=ax, label='Sample Index')
        plt.tight_layout()
        plt.savefig(figures_dir / 'tsne_visualization.png', dpi=300, bbox_inches='tight')
        print(f"   ✓ Saved: tsne_visualization.png")
        plt.close()
    else:
        print(f"   ⚠️  Too few samples ({len(embeddings)}) for t-SNE")
else:
    print("\n6. Skipping t-SNE (not enough samples)")

# Summary Report
print("\n" + "="*60)
print("SUMMARY: Day 2 Signal Analysis")
print("="*60)

print(f"\n📊 Dataset:")
print(f"   • Valid embeddings: {len(embeddings)}")
print(f"   • Embedding dimension: 960")
print(f"   • Sequence length range: {min(seq_lengths)}-{max(seq_lengths)} residues")

print(f"\n🔍 Embedding Quality:")
mean_cos_sim = cos_sim[np.triu_indices_from(cos_sim, k=1)].mean()
if embeddings.std() > 0.01:
    print(f"   ✓ Non-degenerate (std={embeddings.std():.4f})")
else:
    print(f"   ⚠️  Low variance (std={embeddings.std():.4f})")

if mean_cos_sim < 0.99:
    print(f"   ✓ Embeddings are diverse (mean cosine sim={mean_cos_sim:.4f})")
else:
    print(f"   ⚠️  Embeddings very similar (mean cosine sim={mean_cos_sim:.4f})")

print(f"\n📈 Key Findings:")
print(f"   • Mean embedding value: {embeddings.mean():.6f} (should be ~0)")
print(f"   • Std dev: {embeddings.std():.4f} (indicates spread)")
print(f"   • Cosine similarity range: {cos_sim[np.triu_indices_from(cos_sim, k=1)].min():.4f} - {cos_sim[np.triu_indices_from(cos_sim, k=1)].max():.4f}")

print(f"\n✅ Decision:")
if embeddings.std() > 0.01 and mean_cos_sim < 0.99:
    print(f"   ✓ ESM-C embeddings show meaningful variation")
    print(f"   ✓ PROCEED to Day 3 (Overfit Test)")
else:
    print(f"   ⚠️  ESM-C embeddings may lack sufficient signal")
    print(f"   ⚠️  REVIEW before proceeding")

print(f"\n📁 Outputs saved to:")
print(f"   {figures_dir}/")

# Save summary statistics
stats = {
    'n_samples': len(embeddings),
    'embedding_dim': embeddings.shape[1],
    'mean': float(embeddings.mean()),
    'std': float(embeddings.std()),
    'min': float(embeddings.min()),
    'max': float(embeddings.max()),
    'cosine_sim_mean': float(mean_cos_sim),
    'cosine_sim_std': float(cos_sim[np.triu_indices_from(cos_sim, k=1)].std()),
    'seq_length_range': [int(min(seq_lengths)), int(max(seq_lengths))],
    'seq_length_mean': float(np.mean(seq_lengths))
}

np.savez(output_dir / 'day2_statistics.npz', **stats)
print(f"   {output_dir}/day2_statistics.npz")

print("\n✓ Day 2 analysis complete!")
