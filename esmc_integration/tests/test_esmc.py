#!/usr/bin/env python3
"""
Test ESM-C SDK on a single protein sequence.
This verifies the API connection and embedding extraction works.
"""

from esm.sdk import client

# Read API token
with open('.env', 'r') as f:
    token = f.read().strip()

# Initialize ESM-C client
print("Initializing ESM-C client...")
model = client(
    model="esmc-300m-2024-12",
    url="https://forge.evolutionaryscale.ai",
    token=token
)
print("✓ Client initialized")

# Test with a simple protein sequence
test_sequence = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWSTPSELGHAGLNGDILVWNPVLEDAFELSSMGIRVDADTLKHQLALTGDEDRLELEWHQALLRGEMPQTIGGGIGQSRLTMLLLQLPHIGQVQAGVWPAAVRESVPSLL"
print(f"\nTest sequence length: {len(test_sequence)} residues")
print(f"First 50 residues: {test_sequence[:50]}...")

# Generate embedding
print("\nGenerating embedding...")
embedding = model.embed(sequence=test_sequence)
print(f"✓ Embedding generated")

# Check embedding properties
print(f"\nEmbedding shape: {embedding.shape}")
print(f"Embedding dtype: {embedding.dtype}")
print(f"Embedding stats:")
print(f"  Mean: {embedding.mean():.4f}")
print(f"  Std: {embedding.std():.4f}")
print(f"  Min: {embedding.min():.4f}")
print(f"  Max: {embedding.max():.4f}")

# Verify expected shape (should be 960-dim for global embedding)
expected_dim = 960
if len(embedding.shape) == 1 and embedding.shape[0] == expected_dim:
    print(f"\n✓ SUCCESS: Embedding has expected shape ({expected_dim},)")
elif len(embedding.shape) == 2:
    print(f"\n⚠ WARNING: Got per-residue embeddings {embedding.shape}")
    print(f"  Need to extract global embedding (e.g., mean pooling)")
else:
    print(f"\n✗ ERROR: Unexpected embedding shape: {embedding.shape}")

print("\nTest complete!")
