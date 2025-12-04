#!/usr/bin/env python3
"""Check ESM-C API methods"""

from esm.sdk import client

# Read API token
with open('.env', 'r') as f:
    token = f.read().strip()

# Initialize client
model = client(
    model="esmc-300m-2024-12",
    url="https://forge.evolutionaryscale.ai",
    token=token
)

print("Client type:", type(model))
print("\nAvailable methods:")
for attr in dir(model):
    if not attr.startswith('_'):
        print(f"  - {attr}")
