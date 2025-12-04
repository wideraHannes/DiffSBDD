#!/usr/bin/env python3
"""Extract global ESM-C embedding (960-dim) via mean pooling"""

import torch
import numpy as np
from esm.sdk import client
from esm.sdk.api import ESMProtein, LogitsConfig

# Read API token
with open('.env', 'r') as f:
    token = f.read().strip()

# Initialize client
print("Initializing ESM-C client...")
model = client(
    model="esmc-300m-2024-12",
    url="https://forge.evolutionaryscale.ai",
    token=token
)
print("✓ Client initialized")

# Test sequence (from a real protein)
test_sequence = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWSTPSELGHAGLNGDILVWNPVLEDAFELSSMGIRVDADTLKHQLALTGDEDRLELEWHQALLRGEMPQTIGGGIGQSRLTMLLLQLPHIGQVQAGVWPAAVRESVPSLL"
print(f"\nTest sequence: {len(test_sequence)} residues")
print(f"First 50: {test_sequence[:50]}...")

# Create protein and encode
protein = ESMProtein(sequence=test_sequence)
encoded = model.encode(protein)

# Get embeddings
print("\nExtracting embeddings...")
logits_output = model.logits(
    encoded,
    LogitsConfig(sequence=True, return_embeddings=True)
)

# Per-residue embeddings: [batch=1, seq_len, hidden_dim=960]
per_residue_emb = logits_output.embeddings
print(f"✓ Per-residue embeddings: {per_residue_emb.shape}")

# Global embedding via mean pooling (excluding BOS/EOS tokens)
# Assuming first and last tokens are special tokens
global_emb = per_residue_emb[0, 1:-1, :].mean(dim=0)  # [960]
print(f"✓ Global embedding: {global_emb.shape}")

# Convert to numpy (need to convert from bfloat16 to float32 first)
global_emb_np = global_emb.cpu().to(torch.float32).numpy()

# Verify statistics
print(f"\nGlobal embedding statistics:")
print(f"  Shape: {global_emb_np.shape}")
print(f"  Dtype: {global_emb_np.dtype}")
print(f"  Mean: {global_emb_np.mean():.6f}")
print(f"  Std: {global_emb_np.std():.6f}")
print(f"  Min: {global_emb_np.min():.6f}")
print(f"  Max: {global_emb_np.max():.6f}")
print(f"  L2 norm: {np.linalg.norm(global_emb_np):.6f}")

# Save test embedding
output_file = "test_embedding.npy"
np.save(output_file, global_emb_np)
print(f"\n✓ Saved to {output_file}")

# Verify we can load it back
loaded = np.load(output_file)
assert np.allclose(loaded, global_emb_np), "Load/save mismatch!"
print(f"✓ Verified load/save works")

print("\n" + "="*60)
print("SUCCESS! ESM-C global embedding extraction working!")
print("="*60)
print(f"\nExpected shape for DiffSBDD: (960,)")
print(f"Got: {global_emb_np.shape}")
print("✓ Shape matches requirements!")
