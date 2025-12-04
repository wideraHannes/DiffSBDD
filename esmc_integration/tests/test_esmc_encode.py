#!/usr/bin/env python3
"""Test ESM-C encoding and embedding extraction"""

from esm.sdk import client
from esm.sdk.api import ESMProtein

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

# Create protein object
protein = ESMProtein(sequence=test_sequence)
print(f"Protein object: {type(protein)}")

# Encode the protein
print("\nEncoding protein...")
encoded = model.encode(protein)
print(f"✓ Encoded")
print(f"Type: {type(encoded)}")

# Check what's in the encoded object
print("\nEncoded object attributes:")
for attr in dir(encoded):
    if not attr.startswith('_'):
        val = getattr(encoded, attr)
        if not callable(val):
            print(f"  {attr}: {type(val)}")
            if hasattr(val, 'shape'):
                print(f"    shape: {val.shape}")
