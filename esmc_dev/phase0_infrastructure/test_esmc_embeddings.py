#!/usr/bin/env python3
"""Extract ESM-C embeddings"""

import torch
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

# Test sequence
test_sequence = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWSTPSELGHAGLNGDILVWNPVLEDAFELSSMGIRVDADTLKHQLALTGDEDRLELEWHQALLRGEMPQTIGGGIGQSRLTMLLLQLPHIGQVQAGVWPAAVRESVPSLL"
print(f"\nTest sequence: {len(test_sequence)} residues")

# Create protein and encode
protein = ESMProtein(sequence=test_sequence)
encoded = model.encode(protein)

# Get logits/embeddings
print("\nGetting logits...")
logits_output = model.logits(
    encoded,
    LogitsConfig(sequence=True, return_embeddings=True)
)
print(f"✓ Logits computed")
print(f"Type: {type(logits_output)}")

# Check output attributes
print("\nLogits output attributes:")
for attr in dir(logits_output):
    if not attr.startswith('_'):
        val = getattr(logits_output, attr)
        if not callable(val):
            print(f"  {attr}: {type(val)}")
            if hasattr(val, 'shape'):
                print(f"    shape: {val.shape}")
                if attr == 'embeddings':
                    print(f"    mean: {val.mean():.4f}")
                    print(f"    std: {val.std():.4f}")
