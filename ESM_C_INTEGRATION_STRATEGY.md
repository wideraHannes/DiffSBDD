# ESM-C Integration Strategy: Two-Stage Approach (Least Invasive)

**Goal**: Integrate ESM-C protein language model embeddings into DiffSBDD with minimal changes to existing code.

**Date**: 2025-11-18

---

## Executive Summary

**Chosen Approach**: **Two-Stage Processing**

1. **Stage 1**: Minimally modify `process_crossdock.py` to add `pocket_residue_ids` field
2. **Stage 2**: Separate script to compute ESM-C embeddings and augment NPZ files

**Advantages**:
- Keeps `process_crossdock.py` nearly unchanged (only add residue ID tracking)
- ESM-C computation separate from data processing
- Can re-run ESM-C with different models without reprocessing raw data
- Can experiment with different ESM-C layers/configurations easily

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│ Stage 1: Data Processing (Minor Modification)                   │
└─────────────────────────────────────────────────────────────────┘
    Raw PDB/SDF files
         ↓
    process_crossdock.py (minimally modified)
         ↓
    NPZ with pocket_residue_ids
    [lig_coords, lig_one_hot, pocket_coords, pocket_one_hot,
     pocket_ids, pocket_residue_ids]  ← NEW FIELD

┌─────────────────────────────────────────────────────────────────┐
│ Stage 2: ESM-C Augmentation (New Script)                        │
└─────────────────────────────────────────────────────────────────┘
    NPZ with pocket_residue_ids
         ↓
    scripts/augment_with_esmc.py (NEW SCRIPT)
    - Load NPZ
    - For each sample:
      * Extract pocket sequence from pocket_ids
      * Compute ESM-C embedding (960-dim per residue)
      * Broadcast to atoms using pocket_residue_ids
      * Append to pocket_esmc field
         ↓
    NPZ with pocket_esmc
    [... all previous fields ..., pocket_esmc]  ← AUGMENTED

┌─────────────────────────────────────────────────────────────────┐
│ Stage 3: Training (Minor Modifications)                         │
└─────────────────────────────────────────────────────────────────┘
    NPZ with pocket_esmc
         ↓
    Training pipeline (modified to load and use pocket_esmc)
```

---

## Stage 1: Minimal Modification to `process_crossdock.py`

### Changes Required

**Location**: Lines 105-138 (full-atom branch)

**Goal**: Track which residue each atom belongs to

### Code Modification

```python
# BEFORE (lines 105-118):
full_atoms = np.concatenate(
    [
        np.array([atom.element for atom in res.get_atoms()])
        for res in pocket_residues
    ],
    axis=0,
)
full_coords = np.concatenate(
    [
        np.array([atom.coord for atom in res.get_atoms()])
        for res in pocket_residues
    ],
    axis=0,
)

# AFTER (MODIFIED - still simple list comprehension approach):
full_atoms = []
full_coords = []
pocket_residue_ids = []  # NEW: Track parent residue for each atom

for res_idx, res in enumerate(pocket_residues):
    atoms = list(res.get_atoms())
    n_atoms_in_residue = len(atoms)

    # Collect atoms and coords as before
    for atom in atoms:
        full_atoms.append(atom.element)
        full_coords.append(atom.coord)

    # NEW: Track which residue each atom belongs to
    pocket_residue_ids.extend([res_idx] * n_atoms_in_residue)

full_atoms = np.array(full_atoms)
full_coords = np.array(full_coords)
pocket_residue_ids = np.array(pocket_residue_ids, dtype=np.int32)  # NEW
```

```python
# BEFORE (lines 134-138):
pocket_data = {
    "pocket_coords": full_coords,
    "pocket_one_hot": pocket_one_hot,
    "pocket_ids": pocket_ids,
}

# AFTER (MODIFIED):
pocket_data = {
    "pocket_coords": full_coords,
    "pocket_one_hot": pocket_one_hot,
    "pocket_ids": pocket_ids,
    "pocket_residue_ids": pocket_residue_ids,  # NEW FIELD
}
```

### Validation

```python
# Quick check that it works
assert len(pocket_residue_ids) == len(full_atoms), "Mismatch in counts"
assert pocket_residue_ids.min() == 0, "Should start at 0"
assert pocket_residue_ids.max() == len(pocket_residues) - 1, "Should end at n_residues-1"
print(f"✓ Tracked {len(pocket_residues)} residues across {len(full_atoms)} atoms")
```

**That's it!** Only ~15 lines of code changes to `process_crossdock.py`.

---

## Stage 2: New Script `scripts/augment_with_esmc.py`

### Purpose

Load existing NPZ files (with `pocket_residue_ids`) and add `pocket_esmc` field by:
1. Extracting amino acid sequence from `pocket_ids`
2. Computing ESM-C embeddings (960-dim per residue)
3. Broadcasting embeddings to atoms using `pocket_residue_ids`
4. Saving augmented NPZ

### Complete Script

```python
#!/usr/bin/env python3
"""
Augment processed NPZ files with ESM-C embeddings.

Usage:
    python scripts/augment_with_esmc.py \
        --input data/processed_crossdock_noH_full/train.npz \
        --output data/processed_crossdock_esmc/train.npz \
        --token YOUR_ESMC_TOKEN
"""

import numpy as np
import torch
import argparse
from pathlib import Path
from tqdm import tqdm
import sys
import os

# ESM-C imports
from esm.sdk import client
from esm.sdk.api import ESM3InferenceClient, ESMProtein, LogitsConfig

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from constants import dataset_params


def extract_sequence_from_pocket_ids(pocket_ids, dataset='crossdock'):
    """
    Extract amino acid sequence from pocket_ids.

    Args:
        pocket_ids: List/array of residue identifiers (e.g., ["A:123:VAL", "A:124:LEU", ...])
        dataset: Dataset name for decoder lookup

    Returns:
        sequence: String of amino acids (e.g., "VL...")
    """
    # Get the amino acid decoder for this dataset
    aa_decoder = dataset_params[dataset]['aa_decoder']

    # Extract amino acid names from pocket_ids
    # pocket_ids format: "CHAIN:RESNUM:RESNAME" or similar
    # We need to extract the 3-letter amino acid code

    sequence_3letter = []
    for res_id in pocket_ids:
        # pocket_ids might be strings like "A:123:VAL" or similar
        # Try to extract the last part (residue name)
        if isinstance(res_id, str):
            parts = res_id.split(':')
            if len(parts) >= 3:
                res_name = parts[2]  # e.g., "VAL"
            else:
                res_name = parts[-1]  # Use last part
        else:
            res_name = str(res_id)

        sequence_3letter.append(res_name)

    # Convert 3-letter codes to 1-letter codes
    aa_3to1 = {
        'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
        'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
        'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
        'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
    }

    sequence_1letter = []
    for res_3 in sequence_3letter:
        res_3_upper = res_3.upper()
        if res_3_upper in aa_3to1:
            sequence_1letter.append(aa_3to1[res_3_upper])
        else:
            # Unknown residue, use 'X'
            sequence_1letter.append('X')

    return ''.join(sequence_1letter)


def compute_esmc_embedding(sequence, model, config, layer_idx=30):
    """
    Compute ESM-C embedding for a sequence.

    Args:
        sequence: Amino acid sequence string (e.g., "MKTAYIAK...")
        model: ESM3InferenceClient
        config: LogitsConfig
        layer_idx: Which layer to extract (default: 30 = last layer)

    Returns:
        embeddings: (n_residues, 960) numpy array
    """
    try:
        # Create ESMProtein from sequence
        protein = ESMProtein(sequence=sequence)

        # Encode and get embeddings
        protein_tensor = model.encode(protein)
        output = model.logits(protein_tensor, config)

        # Extract hidden states
        # Shape: (n_layers=31, batch=1, n_residues, hidden_dim=960)
        hidden_states = output.hidden_states

        # Get specified layer
        layer_embeddings = hidden_states[layer_idx, 0, :, :]  # (n_residues, 960)

        # Convert to numpy
        embeddings = layer_embeddings.float().cpu().numpy()

        return embeddings

    except Exception as e:
        print(f"WARNING: Failed to compute ESM-C: {e}")
        return None


def broadcast_to_atoms(residue_embeddings, pocket_residue_ids):
    """
    Broadcast per-residue embeddings to per-atom embeddings.

    Args:
        residue_embeddings: (n_residues, esmc_dim) numpy array
        pocket_residue_ids: (n_atoms,) numpy array - which residue each atom belongs to

    Returns:
        atom_embeddings: (n_atoms, esmc_dim) numpy array
    """
    return residue_embeddings[pocket_residue_ids]


def augment_npz_with_esmc(
    input_path,
    output_path,
    esmc_model,
    esmc_config,
    dataset='crossdock',
    layer_idx=30
):
    """
    Augment NPZ file with ESM-C embeddings.
    """
    print(f"Loading data from {input_path}...")
    data = np.load(input_path, allow_pickle=True)

    # Check for required fields
    if 'pocket_residue_ids' not in data.keys():
        raise ValueError(
            "NPZ file missing 'pocket_residue_ids' field. "
            "Please re-process data with modified process_crossdock.py first!"
        )

    # Extract data
    names = data['names']
    pocket_ids = data['pocket_ids']
    pocket_mask = data['pocket_mask']
    pocket_residue_ids = data['pocket_residue_ids']

    n_samples = len(names)
    print(f"Processing {n_samples} samples...")

    # Initialize list to store ESM-C embeddings for all atoms
    all_pocket_esmc = []

    # Process each sample
    for sample_idx in tqdm(range(n_samples), desc="Computing ESM-C"):
        # Get pocket_ids for this sample
        sample_pocket_ids = pocket_ids[sample_idx]

        # Extract sequence
        try:
            sequence = extract_sequence_from_pocket_ids(sample_pocket_ids, dataset)
        except Exception as e:
            print(f"WARNING: Failed to extract sequence for sample {sample_idx}: {e}")
            # Fallback: create zero embeddings
            sample_mask = (pocket_mask == sample_idx)
            n_atoms_in_sample = sample_mask.sum()
            sample_esmc = np.zeros((n_atoms_in_sample, 960), dtype=np.float32)
            all_pocket_esmc.append(sample_esmc)
            continue

        # Compute ESM-C embedding (per-residue)
        residue_embeddings = compute_esmc_embedding(
            sequence, esmc_model, esmc_config, layer_idx
        )

        if residue_embeddings is None:
            # Fallback: zeros
            sample_mask = (pocket_mask == sample_idx)
            n_atoms_in_sample = sample_mask.sum()
            sample_esmc = np.zeros((n_atoms_in_sample, 960), dtype=np.float32)
            all_pocket_esmc.append(sample_esmc)
            continue

        # Verify shape
        expected_n_residues = len(sample_pocket_ids)
        actual_n_residues = residue_embeddings.shape[0]
        if actual_n_residues != expected_n_residues:
            print(f"WARNING: ESM-C returned {actual_n_residues} embeddings "
                  f"but expected {expected_n_residues} for sample {sample_idx}")
            # Fallback: zeros
            sample_mask = (pocket_mask == sample_idx)
            n_atoms_in_sample = sample_mask.sum()
            sample_esmc = np.zeros((n_atoms_in_sample, 960), dtype=np.float32)
            all_pocket_esmc.append(sample_esmc)
            continue

        # Get atoms for this sample
        sample_mask = (pocket_mask == sample_idx)
        sample_residue_ids = pocket_residue_ids[sample_mask]

        # Broadcast to atoms
        sample_esmc = broadcast_to_atoms(residue_embeddings, sample_residue_ids)
        all_pocket_esmc.append(sample_esmc)

    # Concatenate all samples
    pocket_esmc = np.concatenate(all_pocket_esmc, axis=0)

    print(f"Generated pocket_esmc with shape: {pocket_esmc.shape}")
    print(f"Expected shape: ({len(data['pocket_coords'])}, 960)")

    # Save augmented NPZ
    print(f"Saving to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Copy all original fields and add pocket_esmc
    save_dict = {key: val for key, val in data.items()}
    save_dict['pocket_esmc'] = pocket_esmc

    np.savez_compressed(output_path, **save_dict)

    print(f"✓ Done! Saved {n_samples} samples with ESM-C embeddings.")
    print(f"  Original file size: {input_path.stat().st_size / 1e6:.1f} MB")
    print(f"  Augmented file size: {output_path.stat().st_size / 1e6:.1f} MB")


def main():
    parser = argparse.ArgumentParser(
        description="Augment NPZ files with ESM-C embeddings"
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input NPZ file (with pocket_residue_ids)'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output NPZ file (will have pocket_esmc)'
    )
    parser.add_argument(
        '--token',
        type=str,
        default=None,
        help='ESM-C API token (or set ESMC_TOKEN env var)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='esmc-300m-2024-12',
        help='ESM-C model name'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='crossdock',
        help='Dataset name (for amino acid decoder)'
    )
    parser.add_argument(
        '--layer',
        type=int,
        default=30,
        help='Which ESM-C layer to use (default: 30 = last layer)'
    )

    args = parser.parse_args()

    # Get token
    token = args.token or os.getenv('ESMC_TOKEN')
    if not token:
        raise ValueError(
            "ESM-C API token required. "
            "Provide via --token or set ESMC_TOKEN environment variable."
        )

    # Initialize ESM-C client
    print(f"Initializing ESM-C client (model: {args.model})...")
    esmc_model = client(
        model=args.model,
        url="https://forge.evolutionaryscale.ai",
        token=token
    )

    esmc_config = LogitsConfig(
        sequence=True,
        return_embeddings=True,
        return_hidden_states=True
    )

    # Run augmentation
    augment_npz_with_esmc(
        input_path=Path(args.input),
        output_path=Path(args.output),
        esmc_model=esmc_model,
        esmc_config=esmc_config,
        dataset=args.dataset,
        layer_idx=args.layer
    )


if __name__ == '__main__':
    main()
```

---

## Stage 3: Model Modifications

Only need to modify **2 files** to use the ESM-C embeddings:

### 3.1 Modify `equivariant_diffusion/dynamics.py`

```python
class EGNNDynamics(nn.Module):
    def __init__(
        self,
        ...,
        esmc_dim=0,  # NEW: Add this parameter
    ):
        super().__init__()

        self.esmc_dim = esmc_dim
        self.use_esmc = (esmc_dim > 0)

        # Atom encoder (unchanged)
        self.atom_encoder = nn.Sequential(...)

        # Residue encoder: accept [residue_one_hot, esmc] concatenated
        encoder_input_dim = residue_nf + esmc_dim  # MODIFIED
        self.residue_encoder = nn.Sequential(
            nn.Linear(encoder_input_dim, 2 * encoder_input_dim),
            act_fn,
            nn.Linear(2 * encoder_input_dim, joint_nf),
        )

        # Rest unchanged...

    def forward(self, xh_atoms, xh_residues, t, mask_atoms, mask_residues,
                esmc_embeddings=None):  # NEW PARAMETER

        # Extract features
        h_residues = xh_residues[:, self.n_dims:]

        # Concatenate ESM-C before encoding
        if self.use_esmc and esmc_embeddings is not None:
            h_residues = torch.cat([h_residues, esmc_embeddings], dim=-1)

        h_residues = self.residue_encoder(h_residues)

        # Rest unchanged...
```

### 3.2 Modify `lightning_modules.py`

```python
class LigandPocketDDPM(pl.LightningModule):
    def __init__(self, ..., use_esmc=False, esmc_dim=960):
        super().__init__()

        self.use_esmc = use_esmc

        # Pass esmc_dim to dynamics
        self.dynamics = EGNNDynamics(
            ...,
            esmc_dim=esmc_dim if use_esmc else 0,
        )

    def training_step(self, batch, batch_idx):
        ligand = {...}
        pocket = {...}

        # Extract ESM-C embeddings
        esmc = batch.get('pocket_esmc', None) if self.use_esmc else None

        # Pass to ddpm
        loss = self.ddpm(ligand, pocket, esmc=esmc)
        ...
```

**Note**: Also need to propagate `esmc` parameter through `EnVariationalDiffusion` and `ConditionalDDPM` - just add parameter and pass to dynamics.

---

## Usage

### Step 1: Reprocess Data (Minor Change)

```bash
# Modify process_crossdock.py as shown above, then run:
python process_crossdock.py \
    /path/to/crossdocked/train/ \
    --no_H \
    --dist_cutoff 10.0 \
    --ca_only False \
    --output data/processed_crossdock_noH_full_v2/train.npz
```

### Step 2: Augment with ESM-C

```bash
export ESMC_TOKEN="your_token_here"

# Augment train set
python scripts/augment_with_esmc.py \
    --input data/processed_crossdock_noH_full_v2/train.npz \
    --output data/processed_crossdock_esmc/train.npz \
    --token $ESMC_TOKEN

# Augment val set
python scripts/augment_with_esmc.py \
    --input data/processed_crossdock_noH_full_v2/val.npz \
    --output data/processed_crossdock_esmc/val.npz \
    --token $ESMC_TOKEN

# Augment test set
python scripts/augment_with_esmc.py \
    --input data/processed_crossdock_noH_full_v2/test.npz \
    --output data/processed_crossdock_esmc/test.npz \
    --token $ESMC_TOKEN
```

### Step 3: Train

```bash
python train.py --config configs/crossdock_fullatom_cond_esmc.yml
```

---

## Advantages of Two-Stage Approach

1. **Minimal changes to `process_crossdock.py`**: Only ~15 lines changed
2. **Separation of concerns**: Data processing separate from ESM-C computation
3. **Flexibility**: Can re-run ESM-C with different models/layers without reprocessing
4. **Debugging**: Can test each stage independently
5. **Reusability**: Can use same augmentation script for multiple datasets

---

## Timeline

| Step | Task | Time |
|------|------|------|
| 1 | Modify `process_crossdock.py` | 30 min |
| 2 | Test on single PDB | 15 min |
| 3 | Reprocess data (if needed) | 8-12 hours |
| 4 | Write `augment_with_esmc.py` | 1 hour |
| 5 | Test augmentation on subset | 30 min |
| 6 | Augment full dataset | 6-10 hours |
| 7 | Modify model files | 2 hours |
| 8 | Debug training run | 2 hours |

**Total hands-on time**: ~6-7 hours
**Total automated time**: 14-22 hours

---

## Next Steps

1. Modify `process_crossdock.py` (lines 105-138)
2. Create `scripts/augment_with_esmc.py` (provided above)
3. Test on small subset (10 samples)
4. Run full augmentation
5. Modify model files
6. Train and evaluate
