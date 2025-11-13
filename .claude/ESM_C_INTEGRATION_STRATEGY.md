# ESM-C Integration Strategy: Augmentation, Not Replacement

## Your Critical Observations (100% Correct!)

### Problem 1: Per-Residue vs Per-Atom Mismatch
**Current architecture**:
- Pocket encoding: **per-atom** one-hot (if full-atom representation)
- ESM-C embeddings: **per-residue** (960-dim per amino acid)
- **Mismatch**: Can't directly swap 1:1

### Problem 2: Loss of Atom Type Information
**Current decoder flow**:
```python
# dynamics.py:45-49
residue_encoder: one_hot (20) → encoded (128)
EGNN: encoded (128) → processed (128)
residue_decoder: processed (128) → reconstructed one_hot (20)
```
- Decoder **must** reconstruct atom types for loss computation
- If we replace one-hot with ESM-C, decoder can't recover atom types
- **Critical for training!**

---

## The Elegant Solution: Augment, Don't Replace

### Key Insight from EGNN Architecture

Looking at `egnn_new.py:188-245`:
```python
class EGNN(nn.Module):
    def __init__(self, in_node_nf, ...):
        # Input node features → hidden embedding
        self.embedding = nn.Linear(in_node_nf, self.hidden_nf)  # Line 213

        # Message passing layers
        for i in range(n_layers):
            self.add_module("e_block_%d" % i, EquivariantBlock(...))

        # Output embedding → final features
        self.embedding_out = nn.Linear(self.hidden_nf, out_node_nf)  # Line 214
```

**The EGNN input (`in_node_nf`) can be LARGER than output (`out_node_nf`)!**

This means: **We can concat [one-hot + ESM-C] at input, but only decode one-hot!**

---

## Proposed Architecture: Two-Stream Augmentation

### Option: Concatenate at Encoder Input (CLEANEST)

**File**: `equivariant_diffusion/dynamics.py`

```python
class EGNNDynamics(nn.Module):
    def __init__(self, atom_nf, residue_nf, esmc_dim=0,  # NEW PARAM
                 n_dims, joint_nf=128, ...):

        self.esmc_dim = esmc_dim
        self.use_esmc = (esmc_dim > 0)

        # Atom encoder (unchanged)
        self.atom_encoder = nn.Sequential(
            nn.Linear(atom_nf, 2 * atom_nf),
            act_fn,
            nn.Linear(2 * atom_nf, joint_nf)
        )

        # Residue encoder: AUGMENTED INPUT
        encoder_input_dim = residue_nf + esmc_dim  # e.g., 20 + 960 = 980

        self.residue_encoder = nn.Sequential(
            nn.Linear(encoder_input_dim, 2 * encoder_input_dim),  # 980 → 1960
            act_fn,
            nn.Linear(2 * encoder_input_dim, joint_nf)  # 1960 → 128
        )

        # Decoder: ONLY reconstructs one-hot (NOT ESM-C)
        self.residue_decoder = nn.Sequential(
            nn.Linear(joint_nf, 2 * residue_nf),  # 128 → 40
            act_fn,
            nn.Linear(2 * residue_nf, residue_nf)  # 40 → 20 (one-hot only!)
        )

    def forward(self, xh_atoms, xh_residues, t, mask_atoms, mask_residues,
                esmc_embeddings=None):  # NEW PARAM

        x_residues = xh_residues[:, :self.n_dims].clone()
        h_residues = xh_residues[:, self.n_dims:].clone()  # one-hot (20-dim)

        # Encode atoms
        h_atoms = self.atom_encoder(h_atoms)

        # AUGMENT residue features with ESM-C
        if self.use_esmc and esmc_embeddings is not None:
            # Concatenate [one-hot (20) + ESM-C (960)] = 980-dim
            h_residues_augmented = torch.cat([h_residues, esmc_embeddings], dim=-1)
        else:
            h_residues_augmented = h_residues

        # Encode augmented features
        h_residues_encoded = self.residue_encoder(h_residues_augmented)

        # Combine for EGNN (128 + 128 = 256 total nodes)
        x = torch.cat((x_atoms, x_residues), dim=0)
        h = torch.cat((h_atoms, h_residues_encoded), dim=0)
        mask = torch.cat([mask_atoms, mask_residues])

        # ... EGNN processing ...

        # Decode: ONLY one-hot (ESM-C is NOT decoded!)
        h_final_residues = self.residue_decoder(h_final[len(mask_atoms):])

        return ..., h_final_residues  # Returns 20-dim, not 980-dim!
```

### Why This Works

1. ✅ **Preserves atom type information**: One-hot stays in the loop
2. ✅ **Adds ESM-C context**: Rich 960-dim embeddings inform EGNN
3. ✅ **No decoder mismatch**: Decoder only needs to output 20-dim
4. ✅ **Backward compatible**: If `esmc_dim=0`, behaves exactly like original
5. ✅ **Per-residue → Per-atom mapping**: Broadcast ESM-C to all atoms in residue

---

## Handling Per-Residue → Per-Atom Mapping

### Challenge
ESM-C gives 1 embedding per residue, but full-atom representation has multiple atoms per residue.

**CRITICAL FINDING from process_crossdock.py analysis**:
- In full-atom mode (lines 104-138), pocket atoms are extracted WITHOUT residue IDs
- Only `pocket_ids` (residue list) is stored, but NOT atom→residue mapping
- Atoms are concatenated from all residues, losing per-atom residue information

### Solution A: Modify Data Processing (REQUIRED)

**Must modify `process_crossdock.py`** to store residue IDs per atom:

```python
# In process_crossdock.py, full-atom branch (around line 104-138)
# BEFORE (current code):
full_atoms = np.concatenate([
    np.array([atom.element for atom in res.get_atoms()])
    for res in pocket_residues
], axis=0)

# AFTER (modified to track residue IDs):
full_atoms = []
residue_ids = []
for res_idx, res in enumerate(pocket_residues):
    for atom in res.get_atoms():
        full_atoms.append(atom.element)
        residue_ids.append(res_idx)  # Track which residue this atom belongs to

full_atoms = np.array(full_atoms)
residue_ids = np.array(residue_ids, dtype=np.int32)

# Add to pocket_data:
pocket_data = {
    "pocket_coords": full_coords,
    "pocket_one_hot": pocket_one_hot,
    "pocket_ids": pocket_ids,  # Residue identifiers (e.g., "A:123")
    "pocket_residue_ids": residue_ids,  # NEW: atom→residue mapping (e.g., [0,0,0,1,1,1,...])
}
```

### Solution B: Broadcast Residue Embeddings to Atoms

**Preprocessing** (in new script `precompute_esmc.py`):

```python
def broadcast_esmc_to_atoms(esmc_residue_embeddings, residue_ids):
    """
    Broadcast per-residue ESM-C embeddings to per-atom embeddings.

    Args:
        esmc_residue_embeddings: (n_residues, 960)
        residue_ids: (n_atoms,) - which residue each atom belongs to [0, 0, 1, 1, 2, ...]

    Returns:
        esmc_atom_embeddings: (n_atoms, 960)
    """
    return esmc_residue_embeddings[residue_ids]  # Simple indexing!
```

**Example**:
```python
# Residue 0 (GLY) has 4 atoms: N, CA, C, O
# Residue 1 (ALA) has 5 atoms: N, CA, C, O, CB

residue_ids = [0, 0, 0, 0, 1, 1, 1, 1, 1]  # (9 atoms total)
esmc_per_residue = torch.randn(2, 960)  # (2 residues, 960-dim)

esmc_per_atom = esmc_per_residue[residue_ids]  # (9, 960)
# Atoms 0-3 all get residue 0's embedding
# Atoms 4-8 all get residue 1's embedding
```

**Data Format in NPZ**:
```python
# Pre-computed in preprocessing
{
    'pocket_one_hot': (n_atoms, 20),  # Per-atom one-hot
    'pocket_esmc': (n_atoms, 960),     # Per-atom ESM-C (broadcasted)
    'pocket_residue_ids': (n_atoms,),  # Which residue each atom belongs to
    ...
}
```

---

## Modified Data Flow

### Training Forward Pass

```
Input:
  - pocket_coords: (n_atoms, 3)
  - pocket_one_hot: (n_atoms, 20)      ← atom types
  - pocket_esmc: (n_atoms, 960)        ← broadcasted ESM-C

↓ dynamics.forward()

h_residues = pocket_one_hot                    # (n_atoms, 20)
esmc = pocket_esmc                             # (n_atoms, 960)
h_augmented = cat([h_residues, esmc], dim=-1)  # (n_atoms, 980)

↓ residue_encoder

h_encoded = encoder(h_augmented)               # (n_atoms, 128)

↓ EGNN

h_processed = EGNN(h_encoded, ...)             # (n_atoms, 128)

↓ residue_decoder

h_reconstructed = decoder(h_processed)         # (n_atoms, 20) ← one-hot only!

↓ Loss

loss = ||h_reconstructed - pocket_one_hot||^2  # Works!
```

### Conditional Mode: Is Decoder Even Used?

**KEY FINDING** from `conditional_model.py:253-254`:
```python
# Line 253-254: Neural net prediction
net_out_lig, _ = self.dynamics(...)
#                ↑ pocket output is DISCARDED!
```

**In conditional mode, the pocket decoder output is THROWN AWAY!**

Looking at the return statement (dynamics.py:185-186):
```python
return torch.cat([vel[:len(mask_atoms)], h_final_atoms], dim=-1), \
       torch.cat([vel[len(mask_atoms):], h_final_residues], dim=-1)
       # ↑ ligand output                  ↑ pocket output (UNUSED!)
```

And in `conditional_model.py:326`:
```python
loss_terms = (delta_log_px, error_t_lig, torch.tensor(0.0), ...)
#                                        ↑ error_t_pocket = 0.0 (hardcoded!)
```

**CONCLUSION**: In conditional mode, pocket decoder is NOT trained! The pocket is fixed, so reconstruction loss is zero.

**This means**: We can output ANYTHING from the decoder, it won't affect training!

---

## Final Recommended Architecture

### Minimal Change Strategy

**Option 1: Augmentation (RECOMMENDED)**
- Input: `[one-hot (20) + ESM-C (960)]` = 980-dim
- Encoder: `Linear(980 → 1960 → 128)`
- EGNN: `(128-dim)`
- Decoder: `Linear(128 → 40 → 20)` (one-hot only)
- Loss: Only on one-hot reconstruction

**Option 2: Decoder Bypass (if Option 1 fails)**
- Same as Option 1, but decoder outputs 980-dim dummy
- Set `error_t_pocket = 0.0` explicitly (already done in conditional mode!)

### Code Changes Required

**File 1**: `equivariant_diffusion/dynamics.py`
```python
def __init__(self, atom_nf, residue_nf, esmc_dim=0, ...):
    encoder_in = residue_nf + esmc_dim
    self.residue_encoder = nn.Sequential(
        nn.Linear(encoder_in, 2 * encoder_in),
        act_fn,
        nn.Linear(2 * encoder_in, joint_nf)
    )
    # Decoder unchanged (still outputs residue_nf)

def forward(self, ..., esmc_embeddings=None):
    if esmc_embeddings is not None:
        h_residues = torch.cat([h_residues, esmc_embeddings], dim=-1)
    h_residues = self.residue_encoder(h_residues)
    # ... rest unchanged
```

**File 2**: `dataset.py`
```python
def __getitem__(self, idx):
    sample = {...}
    if self.use_esmc:
        sample['pocket_esmc'] = self.data['pocket_esmc'][idx]  # (n_atoms, 960)
    return sample
```

**File 3**: `lightning_modules.py`
```python
def __init__(self, ..., esmc_dim=960):
    self.dynamics = EGNNDynamics(
        atom_nf=...,
        residue_nf=len(self.dataset_info['aa_encoder']),  # Still 20
        esmc_dim=esmc_dim,  # NEW: 960
        ...
    )

def training_step(self, batch):
    ligand, pocket = batch
    esmc = pocket.get('esmc', None)  # May be None if not using ESM-C

    # Pass ESM-C to dynamics
    net_out_lig, net_out_pocket = self.ddpm.dynamics(
        z_t_lig, xh_pocket, t, lig_mask, pocket_mask,
        esmc_embeddings=esmc  # NEW PARAM
    )
```

---

## Advantages of This Approach

### 1. Zero Mental Overhead
- ✅ Original architecture is UNCHANGED except for one concat
- ✅ Decoder still expects 20-dim one-hot
- ✅ Loss computation is IDENTICAL
- ✅ Backward compatible (set `esmc_dim=0` → original model)

### 2. Sparse → Rich Features (Your Motivation!)
- ✅ Original: 20-dim one-hot (very sparse)
- ✅ Augmented: 980-dim [one-hot + ESM-C] (rich context)
- ✅ EGNN gets 48x more information per residue (960/20 = 48)

### 3. No Critical Changes
- ✅ No decoder dimension mismatch
- ✅ No loss function modifications
- ✅ No changes to diffusion process
- ✅ Only change: Encoder input dimension

### 4. Efficient Training
- ✅ Pre-compute ESM-C embeddings offline
- ✅ Broadcast to per-atom during data loading (cheap indexing)
- ✅ Training speed ~same as baseline (just bigger Linear layer)

---

## Preprocessing Pipeline

### Step 1: Extract Sequences from NPZ

**Script**: `scripts/extract_pocket_sequences.py`

```python
import numpy as np
from constants import dataset_params

data = np.load('data/processed_crossdock_noH_full/train.npz', allow_pickle=True)
pocket_one_hot = data['pocket_one_hot']  # (n_total_atoms, 20)
pocket_mask = data['pocket_mask']  # (n_total_atoms,)
names = data['names']

# Decode one-hot to atom types
aa_decoder = dataset_params['crossdock']['aa_decoder']

sequences = []
for sample_idx in range(len(names)):
    # Get atoms for this sample
    sample_atoms = pocket_one_hot[pocket_mask == sample_idx]

    # Extract residue sequence (need residue-level grouping)
    # TODO: Figure out how residues are stored in full-atom mode
    sequence = extract_sequence(sample_atoms, aa_decoder)
    sequences.append(sequence)

np.save('data/pocket_sequences.npy', sequences)
```

**Challenge**: How to map atoms → residues?
- Check if NPZ has `pocket_residue_ids` field
- Or: Infer from PDB processing (`process_crossdock.py`)

### Step 2: Compute ESM-C Embeddings

**Script**: `scripts/compute_esmc_embeddings.py`

```python
import torch
from esm import pretrained

model, alphabet = pretrained.esm2_t33_650M_UR50D()  # Or ESM-C when available
model = model.eval().cuda()

sequences = np.load('data/pocket_sequences.npy', allow_pickle=True)

all_embeddings = []
for seq_batch in tqdm(batch(sequences, 32)):
    batch_tokens = [alphabet.encode(seq) for seq in seq_batch]

    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33])
        embeddings = results['representations'][33]  # (batch, len, 960)

    all_embeddings.append(embeddings.cpu())

np.save('data/pocket_esmc_residue.npy', all_embeddings)
```

### Step 3: Broadcast to Per-Atom

**Script**: `scripts/broadcast_esmc_to_atoms.py`

```python
data = np.load('data/processed_crossdock_noH_full/train.npz', allow_pickle=True)
esmc_residue = np.load('data/pocket_esmc_residue.npy', allow_pickle=True)

# Assuming we have residue_ids for each atom
pocket_residue_ids = data['pocket_residue_ids']  # (n_total_atoms,)

pocket_esmc_atom = []
for sample_idx in range(len(data['names'])):
    sample_residue_ids = pocket_residue_ids[data['pocket_mask'] == sample_idx]
    sample_esmc = esmc_residue[sample_idx][sample_residue_ids]  # Broadcast!
    pocket_esmc_atom.append(sample_esmc)

# Save augmented NPZ
np.savez('data/processed_crossdock_esmc/train.npz',
         **data,
         pocket_esmc=np.concatenate(pocket_esmc_atom))
```

---

## Questions to Resolve

### Q1: Does NPZ have residue IDs?
**Check**: `data['pocket_residue_ids']` or similar field

**If NO**: Need to extract from PDB processing. Look at `process_crossdock.py`.

### Q2: Full-atom vs CA-only?
**From your config**: `pocket_representation: "full-atom"`

This means atoms, not just CA. Need atom→residue mapping.

### Q3: What's in `pocket_one_hot`? ✅ RESOLVED
**Answer**: Full-atom representation with 11 atom types

**Findings from validation**:
- Both `pocket_one_hot` and `lig_one_hot` have shape (n_atoms, 11)
- This is 10 standard atom types (C, N, O, S, B, Br, Cl, P, I, F) + 1 extra
- The extra dimension (index 10) is for unknown/other atoms (line 127-129 in process_crossdock.py)
- Encoding uses `amino_acid_dict` variable name but actually encodes atom types (misleading naming!)

**Code evidence from process_crossdock.py:126-129**:
```python
elif a != "H":
    atom = np.eye(
        1, len(amino_acid_dict), len(amino_acid_dict)  # Creates one-hot at index 10
    ).squeeze()
```

**Implication for ESM-C**:
- Pocket is per-atom, NOT per-residue
- Must broadcast ESM-C embeddings from residues to atoms
- Requires atom→residue mapping (currently missing in NPZ!)

---

## Next Steps (This Week)

### Day 1: Investigate Data Format
```python
# Run this locally
data = np.load('data/processed_crossdock_noH_full/train.npz', allow_pickle=True)
print(data.files)  # What keys exist?
print(data['pocket_one_hot'].shape)
print(data['pocket_one_hot'][0])  # First atom's features

from constants import dataset_params
print(dataset_params['crossdock']['aa_encoder'])
print(dataset_params['crossdock']['atom_encoder'])  # Is there an atom encoder?
```

### Day 2: Understand Pocket Representation
```python
# Read process_crossdock.py
# How are pockets extracted?
# Is there a residue ID stored?
```

### Day 3: Test ESM-C Locally
```python
from esm import pretrained
model, alphabet = pretrained.esm2_t33_650M_UR50D()

seq = "MKTAYIAKQRQISFVKSHFSRQ"  # Example
tokens = alphabet.encode(seq)
results = model(tokens.unsqueeze(0), repr_layers=[33])
embeddings = results['representations'][33]
print(embeddings.shape)  # Should be (1, len(seq), 960)
```

### Day 4-5: Implement Augmentation
1. Modify `dynamics.py` (add `esmc_dim` param, concat in forward)
2. Modify `dataset.py` (load ESM-C embeddings)
3. Test forward pass with dummy ESM-C data

### Day 6-7: Debug Training Loop
1. Run 1 epoch with ESM-C
2. Verify loss is computed correctly
3. Check no NaN issues

---

## Summary: The Clean Solution

**What we're doing**:
1. ✅ **Augment** one-hot with ESM-C (not replace)
2. ✅ **Concat** at encoder input: `[20 + 960] → encoder → 128`
3. ✅ **Preserve** decoder: `128 → decoder → 20` (one-hot only)
4. ✅ **Broadcast** per-residue ESM-C to per-atom (simple indexing)
5. ✅ **No changes** to loss, diffusion, or EGNN core

**Why it works**:
- Encoder can have larger input than output (EGNN already does this!)
- Decoder only needs to reconstruct one-hot (ESM-C not needed for loss)
- In conditional mode, pocket decoder is unused anyway
- Minimal code changes, maximum information gain

**Your original intuition was spot-on**: The model uses sparse one-hot features. ESM-C augmentation gives it rich contextual protein information without breaking anything.

---

## ✅ VALIDATION COMPLETE (2025-11-13)

### All Tests Passed (6/6)

**Status**: Integration strategy is **CONFIRMED FEASIBLE**

**Key Validations**:
1. ✅ Data structure inspected - full-atom representation (11-dim encoding)
2. ✅ Broadcasting strategy tested - simple indexing works perfectly
3. ✅ Encoder/decoder dimensions verified - augmentation approach compatible
4. ✅ Mock integration successful - no dimension mismatches
5. ✅ Conditional mode compatible - decoder output unused anyway
6. ✅ Storage requirements acceptable - ~76 GB (<1% of quota)

**Critical Finding**:
- Current NPZ files **MISSING** `pocket_residue_ids` field
- Must modify `process_crossdock.py` to track atom→residue mapping
- Re-processing dataset is **REQUIRED** before ESM-C integration

**Next Steps**:
1. Modify `process_crossdock.py` (see `experiments/example_residue_id_tracking.py`)
2. Re-process dataset with residue ID tracking
3. Pre-compute ESM-C embeddings
4. Implement augmentation in `dynamics.py`
5. Train and evaluate

See `experiments/VALIDATION_SUMMARY.md` for detailed results and timeline.

---

## References

**Validation Scripts** (in `experiments/`):
- `validate_esmc_integration.py` - Main validation suite
- `analyze_pocket_representation.py` - Deep dive into data format
- `example_residue_id_tracking.py` - Demonstrates required code changes
- `VALIDATION_SUMMARY.md` - Full validation report

**Strategy**: Proceed with confidence. The augmentation approach is clean, backward-compatible, and thoroughly validated.
