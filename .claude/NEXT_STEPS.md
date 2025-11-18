# ESM-C Integration: Next Steps

**Status**: Validation Complete ✅ (6/6 tests passed)
**Date**: 2025-11-13
**Ready to proceed**: YES - all technical blockers resolved

---

## 🚨 Critical Path: Data Re-processing

**Blocker**: Current NPZ files are missing `pocket_residue_ids` field
**Impact**: Cannot broadcast ESM-C embeddings without atom→residue mapping
**Priority**: HIGHEST - must complete before any other steps

---

## Phase 0A: Baseline Evaluation (Week 1 - PARALLEL TASK)

**IMPORTANT**: Before making any changes, establish baseline performance for comparison!

### Step 0: Evaluate Existing Baseline Model

**Purpose**: Document baseline performance to compare against ESM-C model

**Checkpoint**: You should already have the pre-trained baseline model:
- `checkpoints/crossdocked_fullatom_cond.ckpt` (from Zenodo)
- If not, download from: https://zenodo.org/record/8183747

**Action Items**:
- [ ] Verify baseline checkpoint exists
- [ ] Load and test checkpoint
- [ ] Generate molecules on test set
- [ ] Compute all metrics
- [ ] Save results for comparison

---

#### Step 0.1: Verify Baseline Checkpoint

**Purpose**: Ensure the pre-trained baseline model is available and loads correctly

**Check checkpoint exists**:
```bash
# Verify checkpoint file
ls -lh checkpoints/crossdocked_fullatom_cond.ckpt

# Expected: ~17MB file
# If missing, download from Zenodo:
# wget https://zenodo.org/record/8183747/files/crossdocked_fullatom_cond.ckpt \
#     -O checkpoints/crossdocked_fullatom_cond.ckpt
```

**Test checkpoint loads**:
```python
import torch
from lightning_modules import LigandPocketDDPM

# Load checkpoint
checkpoint_path = "checkpoints/crossdocked_fullatom_cond.ckpt"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = LigandPocketDDPM.load_from_checkpoint(
    checkpoint_path, map_location=device
)
model = model.to(device)
model.eval()

print(f"✓ Loaded model: {model.__class__.__name__}")
print(f"  Mode: {model.hparams.mode}")
print(f"  Dataset: {model.hparams.dataset}")
print(f"  Pocket representation: {model.hparams.pocket_representation}")
```

**Expected Output**:
```
✓ Loaded model: LigandPocketDDPM
  Mode: pocket_conditioning
  Dataset: crossdock
  Pocket representation: full-atom
```

**Success Criteria**:
- [ ] Checkpoint file exists (should be ~17MB)
- [ ] Model loads without errors
- [ ] Configuration matches: conditional, full-atom, CrossDocked

---

#### Step 0.2: Generate Baseline Molecules (10 Test Proteins)

**Purpose**: Generate molecules for test proteins to verify pipeline and establish baseline metrics

**Why 10 proteins?**:
- Quick verification (2-5 minutes vs hours for full test set)
- Sufficient to catch issues and compute preliminary metrics
- Can expand to full test set after verification

**Official Method** (using existing `test.py` script):

```bash
# Generate 100 molecules per pocket using the official test.py script
python test.py checkpoints/crossdocked_fullatom_cond.ckpt \
    --test_dir data/processed_crossdock_noH_full_temp/test/ \
    --outdir results/baseline_10proteins/ \
    --n_samples 100 \
    --batch_size 100 \
    --sanitize \
    --skip_existing
```

**What this does**:
- Reads all SDF files from test directory (currently 10 proteins)
- For each protein:
  - Loads PDB structure from `<name>.pdb`
  - Reads pocket residues from `<name>.txt`
  - Generates 100 molecules
  - Saves raw molecules to `results/baseline_10proteins/raw/<name>_gen.sdf`
  - Saves processed (valid) molecules to `results/baseline_10proteins/processed/<name>_gen.sdf`
  - Tracks timing in `results/baseline_10proteins/pocket_times/<name>.txt`

**Output Structure**:
```
results/baseline_10proteins/
├── raw/                      # All generated molecules (including invalid)
│   ├── <pocket_name>_gen.sdf
│   └── ...
├── processed/                # Valid molecules only
│   ├── <pocket_name>_gen.sdf
│   └── ...
├── pocket_times/             # Generation time per pocket
│   └── *.txt
└── pocket_times.txt          # Summary of all times
```

**Expected Runtime**: 2-5 minutes (depends on GPU)

**Expected Output**:
```
Processing pocket 1/10: 14gs-A-rec-...
  Generated 100 molecules in 12.34 seconds
  Valid: 72/100 (72.0%)

...

Time per pocket: 14.532 ± 3.21
```

**Quick Validation**:
```python
# Verify generation results
from pathlib import Path
from rdkit import Chem

processed_dir = Path("results/baseline_10proteins/processed")
sdf_files = list(processed_dir.glob("*.sdf"))

print(f"Processed {len(sdf_files)} pockets")

total_mols = 0
for sdf in sdf_files:
    suppl = Chem.SDMolSupplier(str(sdf))
    n_mols = len([m for m in suppl if m is not None])
    total_mols += n_mols
    print(f"  {sdf.stem}: {n_mols} valid molecules")

print(f"\nTotal valid molecules: {total_mols}/{len(sdf_files)*100}")
print(f"Average validity: {total_mols/(len(sdf_files)*100):.1%}")
```

**Success Criteria**:
- [ ] Script completes without errors
- [ ] 10 pockets processed
- [ ] Both `raw/` and `processed/` directories contain SDF files
- [ ] Validity > 50% (typical baseline: 60-75%)
- [ ] Timing file created

**Notes**:
- This uses the **official test.py script** (no custom code needed)
- Can easily scale to full test set by running same command with more test data
- `--skip_existing` allows resuming if interrupted
- Results are comparable to paper benchmarks

**Performance Insights (from testing)**:
- **CPU performance**: ~60-80 seconds per molecule on MacBook (M1/M2)
- **Expected GPU performance**: ~0.5-1 second per molecule on V100/A100
- **Validity rate**: ~100% on baseline model (very high quality)
- **File format**: Each SDF file contains ALL molecules for that pocket (use RDKit iterator, not indexing!)

**Common Pitfall - Reading SDF Files**:
```python
# ❌ WRONG - Only reads first molecule
suppl = Chem.SDMolSupplier(str(sdf_file))
mol = suppl[0]  # Only gets first molecule!

# ✅ CORRECT - Reads all molecules
suppl = Chem.SDMolSupplier(str(sdf_file))
mols = [m for m in suppl if m is not None]  # Gets all molecules
```

---

### Summary of Phase 0A

**Deliverables**:
1. Baseline checkpoint verified ✓
2. Molecules generated for 10 test proteins ✓

**Time Estimate**: 5-10 minutes

**Can run in parallel with**: Data re-processing (Phase 0B)

**Files Created**:
- `results/baseline_10proteins/raw/*.sdf` - All generated molecules
- `results/baseline_10proteins/processed/*.sdf` - Valid molecules only
- `results/baseline_10proteins/pocket_times.txt` - Generation timing

**Next Steps After Verification**:
- Once verified on 10 proteins, can expand to full test set if more data available
- These preliminary results are sufficient for initial comparison with ESM-C model
- Can compute detailed metrics (QED, SA Score, docking) if needed for thesis

---

## Phase 0B: Data Re-processing (Week 1)

### Step 1: Modify `process_crossdock.py`

**File**: `process_crossdock.py`
**Location**: Lines 105-138 (full-atom branch)
**Reference**: See `experiments/example_residue_id_tracking.py` for exact code

#### Changes Required:

```python
# BEFORE (lines 105-118):
full_atoms = np.concatenate([
    np.array([atom.element for atom in res.get_atoms()])
    for res in pocket_residues
], axis=0)

full_coords = np.concatenate([
    np.array([atom.coord for atom in res.get_atoms()])
    for res in pocket_residues
], axis=0)

# AFTER (modified):
# Track residue IDs per atom
full_atoms = []
full_coords = []
residue_ids = []  # NEW

for res_idx, res in enumerate(pocket_residues):
    for atom in res.get_atoms():
        full_atoms.append(atom.element)
        full_coords.append(atom.coord)
        residue_ids.append(res_idx)  # NEW: Track parent residue

full_atoms = np.array(full_atoms)
full_coords = np.array(full_coords)
residue_ids = np.array(residue_ids, dtype=np.int32)  # NEW
```

```python
# BEFORE (line ~134-138):
pocket_data = {
    "pocket_coords": full_coords,
    "pocket_one_hot": pocket_one_hot,
    "pocket_ids": pocket_ids,
}

# AFTER (modified):
pocket_data = {
    "pocket_coords": full_coords,
    "pocket_one_hot": pocket_one_hot,
    "pocket_ids": pocket_ids,
    "pocket_residue_ids": residue_ids,  # NEW
}
```

**Action Items**:
- [ ] Open `process_crossdock.py`
- [ ] Apply changes to lines 105-138
- [ ] Save modified file
- [ ] Test on single PDB (see Step 2)

---

### Step 2: Test on Single PDB

**Purpose**: Verify modifications work before re-processing full dataset

```bash
# Test on a single structure
python process_crossdock.py \
    /path/to/test.pdb \
    /path/to/test.sdf \
    --no_H \
    --dist_cutoff 10.0

# Verify output
python -c "
import numpy as np
data = np.load('test_output.npz', allow_pickle=True)
print('Keys:', list(data.keys()))
print('Has residue IDs:', 'pocket_residue_ids' in data.keys())
if 'pocket_residue_ids' in data.keys():
    print('Residue IDs:', data['pocket_residue_ids'])
"
```

**Expected Output**:
```
Keys: [..., 'pocket_residue_ids', ...]
Has residue IDs: True
Residue IDs: [0 0 0 0 1 1 1 1 1 2 2 2 ...]
```

**Action Items**:
- [ ] Run on single PDB file
- [ ] Verify `pocket_residue_ids` field exists
- [ ] Check residue IDs are contiguous and start from 0
- [ ] Verify number of residues matches `len(pocket_ids)`

---

### Step 3: Re-process Full Dataset

**Dataset**: CrossDock (~100K samples)
**Time estimate**: 8-16 hours (depends on hardware)
**Storage**: Keep both old and new NPZ for safety

```bash
# Create output directory
mkdir -p data/processed_crossdock_noH_full_esmc

# Re-process train set
python process_crossdock.py \
    /path/to/crossdocked/train/ \
    --no_H \
    --dist_cutoff 10.0 \
    --ca_only False \
    --output data/processed_crossdock_noH_full_esmc/train.npz

# Re-process val set
python process_crossdock.py \
    /path/to/crossdocked/val/ \
    --no_H \
    --dist_cutoff 10.0 \
    --ca_only False \
    --output data/processed_crossdock_noH_full_esmc/val.npz

# Re-process test set
python process_crossdock.py \
    /path/to/crossdocked/test/ \
    --no_H \
    --dist_cutoff 10.0 \
    --ca_only False \
    --output data/processed_crossdock_noH_full_esmc/test.npz
```

**HPC Submission** (if processing is slow):
```bash
# Create PBS script
cat > .claude/hpc/reprocess_data.pbs << 'EOF'
#!/bin/bash
#PBS -N reprocess_crossdock
#PBS -l select=1:ncpus=4:mem=32gb
#PBS -l walltime=24:00:00
#PBS -A your_project_id

cd $PBS_O_WORKDIR
module load python/3.9

python process_crossdock.py \
    /path/to/crossdocked/train/ \
    --no_H \
    --dist_cutoff 10.0 \
    --ca_only False \
    --output data/processed_crossdock_noH_full_esmc/train.npz
EOF

# Submit
qsub .claude/hpc/reprocess_data.pbs
```

**Action Items**:
- [ ] Run re-processing (locally or on HPC)
- [ ] Monitor progress (check log files)
- [ ] Verify all three splits (train/val/test) complete
- [ ] Check file sizes are similar to original NPZ files
- [ ] Run validation script on new data (see Step 4)

---

### Step 4: Validate Re-processed Data

```bash
# Check new NPZ structure
python -c "
import numpy as np

for split in ['train', 'val', 'test']:
    print(f'\n=== {split.upper()} ===')
    data = np.load(f'data/processed_crossdock_noH_full_esmc/{split}.npz', allow_pickle=True)

    print(f'Keys: {list(data.keys())}')
    print(f'Samples: {len(data[\"names\"])}')
    print(f'Pocket atoms: {len(data[\"pocket_coords\"])}')

    if 'pocket_residue_ids' in data.keys():
        res_ids = data['pocket_residue_ids']
        print(f'Residue IDs: {len(res_ids)} atoms')
        print(f'Unique residues per sample: {len(set(res_ids[:100]))}')  # First sample
        print('✓ Residue IDs present!')
    else:
        print('✗ ERROR: pocket_residue_ids missing!')
"
```

**Expected Output**:
```
=== TRAIN ===
Keys: ['names', 'lig_coords', ..., 'pocket_residue_ids']
Samples: 100000
Pocket atoms: 35301000
Residue IDs: 35301000 atoms
Unique residues per sample: ~100
✓ Residue IDs present!
```

**Action Items**:
- [ ] Verify `pocket_residue_ids` exists in all splits
- [ ] Check residue ID ranges are correct (start from 0 per sample)
- [ ] Verify total atom counts match
- [ ] Spot-check a few samples manually

---

## Phase 1: ESM-C Pre-computation (Week 1-2)

### Step 5: Install ESM-C

```bash
# Install ESM (fair-esm package)
uv pip install fair-esm

# Test installation
python -c "
from esm import pretrained
print('ESM installation successful!')
print('Available models:')
for name in dir(pretrained):
    if not name.startswith('_'):
        print(f'  - {name}')
"
```

**Model Selection**:
- `esm2_t33_650M_UR50D`: ESM-2 650M params (recommended, 1280-dim)
- `esmc_300M`: ESM-C 300M params (960-dim, if available)
- `esmc_600M`: ESM-C 600M params (960-dim, if available)

**Action Items**:
- [ ] Install fair-esm
- [ ] Test model loading locally
- [ ] Check available models
- [ ] Decide on model version (ESM-2 or ESM-C)

---

### Step 6: Write `scripts/precompute_esmc.py`

**File**: `scripts/precompute_esmc.py`

```python
"""
Pre-compute ESM-C embeddings for all pockets in dataset.

Usage:
    python scripts/precompute_esmc.py \
        --input data/processed_crossdock_noH_full_esmc/train.npz \
        --output data/processed_crossdock_esmc/train.npz \
        --model esm2_t33_650M_UR50D \
        --batch_size 32 \
        --device cuda
"""

import numpy as np
import torch
from esm import pretrained
from tqdm import tqdm
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from constants import dataset_params


def extract_sequences_from_pocket_ids(pocket_ids):
    """
    Extract amino acid sequences from pocket_ids.

    Args:
        pocket_ids: List of residue identifiers (e.g., ["A:123", "A:124", ...])

    Returns:
        sequence: String of amino acids (e.g., "MKTAYIAK...")
    """
    # TODO: Implement sequence extraction
    # Need to map residue IDs to amino acid types
    # May need to read from original PDB or infer from pocket_one_hot
    raise NotImplementedError("Sequence extraction not yet implemented")


def broadcast_to_atoms(esmc_residue_embeddings, residue_ids):
    """
    Broadcast per-residue embeddings to per-atom embeddings.

    Args:
        esmc_residue_embeddings: (n_residues, esmc_dim)
        residue_ids: (n_atoms,) - which residue each atom belongs to

    Returns:
        esmc_atom_embeddings: (n_atoms, esmc_dim)
    """
    return esmc_residue_embeddings[residue_ids]


def compute_esmc_embeddings(npz_path, model_name, batch_size, device, output_path):
    """
    Compute ESM-C embeddings for all samples in NPZ file.
    """
    print(f"Loading data from {npz_path}...")
    data = np.load(npz_path, allow_pickle=True)

    # Check for residue IDs
    if 'pocket_residue_ids' not in data.keys():
        raise ValueError("NPZ file missing 'pocket_residue_ids' field. "
                        "Re-process data with modified process_crossdock.py first!")

    # Load ESM model
    print(f"Loading ESM model: {model_name}...")
    model, alphabet = getattr(pretrained, model_name)()
    model = model.eval().to(device)
    esmc_dim = model.embed_dim

    print(f"ESM embedding dimension: {esmc_dim}")

    # Extract sequences
    print("Extracting sequences...")
    sequences = []
    n_samples = len(data['names'])

    for i in tqdm(range(n_samples)):
        pocket_ids_sample = data['pocket_ids'][i]
        sequence = extract_sequences_from_pocket_ids(pocket_ids_sample)
        sequences.append(sequence)

    # Compute embeddings in batches
    print(f"Computing ESM-C embeddings (batch_size={batch_size})...")
    all_embeddings = []

    for i in tqdm(range(0, len(sequences), batch_size)):
        batch_seqs = sequences[i:i+batch_size]

        # Tokenize
        batch_tokens = [alphabet.encode(seq) for seq in batch_seqs]
        max_len = max(len(t) for t in batch_tokens)
        batch_tokens_padded = torch.full((len(batch_tokens), max_len),
                                        alphabet.padding_idx, dtype=torch.long)

        for j, tokens in enumerate(batch_tokens):
            batch_tokens_padded[j, :len(tokens)] = tokens

        # Forward pass
        with torch.no_grad():
            batch_tokens_padded = batch_tokens_padded.to(device)
            results = model(batch_tokens_padded, repr_layers=[model.num_layers])
            embeddings = results['representations'][model.num_layers]  # (batch, len, dim)

        # Extract per-residue embeddings (skip BOS/EOS tokens)
        for j, seq in enumerate(batch_seqs):
            # tokens: <cls> seq <eos>
            # embeddings[j, 1:-1, :] are the residue embeddings
            residue_embeddings = embeddings[j, 1:len(seq)+1, :].cpu().numpy()
            all_embeddings.append(residue_embeddings)

    # Broadcast to atoms
    print("Broadcasting to atoms...")
    pocket_esmc = []
    pocket_mask = data['pocket_mask']
    pocket_residue_ids_all = data['pocket_residue_ids']

    for i in tqdm(range(n_samples)):
        # Get atoms for this sample
        sample_mask = (pocket_mask == i)
        sample_residue_ids = pocket_residue_ids_all[sample_mask]

        # Broadcast
        sample_embeddings = all_embeddings[i]
        sample_esmc = broadcast_to_atoms(sample_embeddings, sample_residue_ids)
        pocket_esmc.append(sample_esmc)

    pocket_esmc = np.concatenate(pocket_esmc, axis=0)

    # Save augmented NPZ
    print(f"Saving to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_path,
        **{k: v for k, v in data.items()},  # Copy all original fields
        pocket_esmc=pocket_esmc,  # Add ESM-C embeddings
    )

    print(f"✓ Done! Saved {n_samples} samples with ESM-C embeddings.")
    print(f"  pocket_esmc shape: {pocket_esmc.shape}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Input NPZ file')
    parser.add_argument('--output', type=str, required=True, help='Output NPZ file')
    parser.add_argument('--model', type=str, default='esm2_t33_650M_UR50D',
                       help='ESM model name')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--device', type=str, default='cuda', help='Device')

    args = parser.parse_args()

    compute_esmc_embeddings(
        npz_path=Path(args.input),
        model_name=args.model,
        batch_size=args.batch_size,
        device=args.device,
        output_path=Path(args.output),
    )


if __name__ == '__main__':
    main()
```

**Action Items**:
- [ ] Create `scripts/precompute_esmc.py`
- [ ] Implement `extract_sequences_from_pocket_ids()` function
- [ ] Test on small subset (10 samples)
- [ ] Profile runtime and memory usage

---

### Step 7: Run ESM-C Pre-computation

**Local Test** (small subset):
```bash
# Test on 100 samples
python scripts/precompute_esmc.py \
    --input data/processed_crossdock_noH_full_esmc/train_subset_100.npz \
    --output data/processed_crossdock_esmc/train_subset_100.npz \
    --model esm2_t33_650M_UR50D \
    --batch_size 16 \
    --device cuda
```

**HPC Submission** (full dataset):
```bash
# Create PBS script for each split
cat > .claude/hpc/precompute_esmc_train.pbs << 'EOF'
#!/bin/bash
#PBS -N esmc_train
#PBS -l select=1:ncpus=4:ngpus=1:mem=32gb
#PBS -l walltime=24:00:00
#PBS -A your_project_id

cd $PBS_O_WORKDIR
module load cuda/11.8
module load python/3.9

python scripts/precompute_esmc.py \
    --input data/processed_crossdock_noH_full_esmc/train.npz \
    --output data/processed_crossdock_esmc/train.npz \
    --model esm2_t33_650M_UR50D \
    --batch_size 32 \
    --device cuda
EOF

# Submit
qsub .claude/hpc/precompute_esmc_train.pbs
```

**Time Estimate**:
- 100K samples × ~100 residues/sample = 10M residues
- ESM-2 throughput: ~500 residues/sec on V100
- Total time: 10M / 500 = 20K seconds ≈ 5-6 hours

**Action Items**:
- [ ] Test on subset (100 samples)
- [ ] Profile runtime per sample
- [ ] Submit HPC jobs for train/val/test splits
- [ ] Monitor job progress
- [ ] Verify output files (see Step 8)

---

### Step 8: Validate ESM-C Embeddings

```bash
# Check ESM-C augmented NPZ
python -c "
import numpy as np

data = np.load('data/processed_crossdock_esmc/train.npz', allow_pickle=True)

print('Keys:', list(data.keys()))
print('Has ESM-C:', 'pocket_esmc' in data.keys())

if 'pocket_esmc' in data.keys():
    esmc = data['pocket_esmc']
    print(f'ESM-C shape: {esmc.shape}')
    print(f'Expected: ({len(data[\"pocket_coords\"])}, 960 or 1280)')

    # Verify broadcasting worked
    pocket_mask = data['pocket_mask']
    pocket_residue_ids = data['pocket_residue_ids']

    # Check first sample
    sample_mask = (pocket_mask == 0)
    sample_esmc = esmc[sample_mask]
    sample_res_ids = pocket_residue_ids[sample_mask]

    # Atoms from same residue should have identical embeddings
    for res_id in np.unique(sample_res_ids)[:5]:
        atom_indices = np.where(sample_res_ids == res_id)[0]
        embeddings = sample_esmc[atom_indices]
        all_equal = np.allclose(embeddings, embeddings[0])
        print(f'Residue {res_id}: {len(atom_indices)} atoms, embeddings equal: {all_equal}')
"
```

**Expected Output**:
```
Keys: [..., 'pocket_esmc']
Has ESM-C: True
ESM-C shape: (35301000, 960)
Expected: (35301000, 960 or 1280)
Residue 0: 4 atoms, embeddings equal: True
Residue 1: 5 atoms, embeddings equal: True
...
```

**Action Items**:
- [ ] Verify `pocket_esmc` field exists in all splits
- [ ] Check embedding dimensions (960 or 1280 depending on model)
- [ ] Verify broadcasting (atoms from same residue have identical embeddings)
- [ ] Spot-check embeddings are not all zeros or NaNs

---

## Phase 2: Model Implementation (Week 2)

### Step 9: Modify `equivariant_diffusion/dynamics.py`

**File**: `equivariant_diffusion/dynamics.py`

**Changes**:

```python
# Line 11: Add esmc_dim parameter
class EGNNDynamics(nn.Module):
    def __init__(self, atom_nf, residue_nf,
                 n_dims, joint_nf=16, hidden_nf=64, device='cpu',
                 act_fn=torch.nn.SiLU(), n_layers=4, attention=False,
                 condition_time=True, tanh=False, mode='egnn_dynamics',
                 norm_constant=0, inv_sublayers=2, sin_embedding=False,
                 normalization_factor=100, aggregation_method='sum',
                 update_pocket_coords=True, edge_cutoff_ligand=None,
                 edge_cutoff_pocket=None, edge_cutoff_interaction=None,
                 reflection_equivariant=True, edge_embedding_dim=None,
                 esmc_dim=0):  # NEW PARAMETER
        super().__init__()

        self.esmc_dim = esmc_dim  # NEW
        self.use_esmc = (esmc_dim > 0)  # NEW

        # ... (atom encoder unchanged)

        # Line 39-43: Modify residue encoder
        encoder_input_dim = residue_nf + esmc_dim  # NEW: augmented input
        self.residue_encoder = nn.Sequential(
            nn.Linear(encoder_input_dim, 2 * encoder_input_dim),  # MODIFIED
            act_fn,
            nn.Linear(2 * encoder_input_dim, joint_nf)  # MODIFIED
        )

        # Decoder unchanged (still outputs residue_nf)

# Line 87: Add esmc_embeddings parameter
    def forward(self, xh_atoms, xh_residues, t, mask_atoms, mask_residues,
                esmc_embeddings=None):  # NEW PARAMETER

        x_residues = xh_residues[:, :self.n_dims].clone()
        h_residues = xh_residues[:, self.n_dims:].clone()

        # ... (atom encoding unchanged)

        # NEW: Augment with ESM-C before encoding
        if self.use_esmc and esmc_embeddings is not None:
            h_residues = torch.cat([h_residues, esmc_embeddings], dim=-1)

        h_residues = self.residue_encoder(h_residues)

        # ... (rest unchanged)
```

**Action Items**:
- [ ] Add `esmc_dim` parameter to `__init__`
- [ ] Modify `encoder_input_dim` calculation
- [ ] Update encoder Linear layers
- [ ] Add `esmc_embeddings` parameter to `forward()`
- [ ] Add concatenation logic before encoding
- [ ] Test forward pass with dummy data

---

### Step 10: Modify `dataset.py`

**File**: `dataset.py`

**Changes**:

```python
# Line 8: Add use_esmc parameter
class ProcessedLigandPocketDataset(Dataset):
    def __init__(self, npz_path, center=True, transform=None, use_esmc=False):  # NEW

        self.transform = transform
        self.use_esmc = use_esmc  # NEW

        with np.load(npz_path, allow_pickle=True) as f:
            data = {key: val for key, val in f.items()}

        # ... (split data as before)

        # NEW: Check for ESM-C embeddings if requested
        if self.use_esmc and 'pocket_esmc' not in data.keys():
            raise ValueError(f"use_esmc=True but NPZ missing 'pocket_esmc' field!")

# Line 46: Return ESM-C embeddings in __getitem__
    def __getitem__(self, idx):
        data = {key: val[idx] for key, val in self.data.items()}

        # NEW: Handle ESM-C embeddings
        if self.use_esmc:
            data['pocket_esmc'] = self.data['pocket_esmc'][idx]

        if self.transform is not None:
            data = self.transform(data)
        return data

# Line 53: Handle in collate_fn
    @staticmethod
    def collate_fn(batch):
        out = {}
        for prop in batch[0].keys():

            if prop == 'names' or prop == 'receptors':
                out[prop] = [x[prop] for x in batch]
            elif prop == 'num_lig_atoms' or prop == 'num_pocket_nodes' \
                    or prop == 'num_virtual_atoms':
                out[prop] = torch.tensor([x[prop] for x in batch])
            elif 'mask' in prop:
                out[prop] = torch.cat([i * torch.ones(len(x[prop]))
                                       for i, x in enumerate(batch)], dim=0)
            else:
                out[prop] = torch.cat([x[prop] for x in batch], dim=0)

        return out
```

**Action Items**:
- [ ] Add `use_esmc` parameter to `__init__`
- [ ] Add ESM-C validation check
- [ ] Return `pocket_esmc` in `__getitem__`
- [ ] Test data loading with ESM-C

---

### Step 11: Modify `lightning_modules.py`

**File**: `lightning_modules.py`

**Changes**:

```python
# Line ~50: Add use_esmc to __init__
class LigandPocketDDPM(pl.LightningModule):
    def __init__(self, ..., use_esmc=False, esmc_dim=960):  # NEW
        super().__init__()

        self.use_esmc = use_esmc  # NEW

        # ... (dataset info)

        # Line ~80: Pass esmc_dim to dynamics
        self.dynamics = EGNNDynamics(
            atom_nf=len(self.dataset_info['atom_encoder']),
            residue_nf=len(self.dataset_info['aa_encoder']),  # Still 20
            esmc_dim=esmc_dim if use_esmc else 0,  # NEW
            # ... (other params)
        )

# Line ~130: Modify setup() to pass use_esmc to dataset
    def setup(self, stage=None):
        self.train_dataset = ProcessedLigandPocketDataset(
            os.path.join(self.data_path, 'train.npz'),
            center=True,
            transform=self.transform,
            use_esmc=self.use_esmc,  # NEW
        )
        # ... (val/test datasets similarly)

# Line ~274: Extract ESM-C in training_step
    def training_step(self, batch, batch_idx):
        ligand = {
            'x': batch['lig_coords'],
            'one_hot': batch['lig_one_hot'],
            'size': batch['num_lig_atoms'],
            'mask': batch['lig_mask'].long(),
        }
        pocket = {
            'x': batch['pocket_coords'],
            'one_hot': batch['pocket_one_hot'],
            'size': batch['num_pocket_nodes'],
            'mask': batch['pocket_mask'].long(),
        }

        # NEW: Extract ESM-C embeddings
        esmc = batch.get('pocket_esmc', None) if self.use_esmc else None

        # Pass to ddpm (need to modify ddpm.forward to accept esmc)
        delta_log_px, error_t_lig, error_t_pocket, SNR_weight, \
            loss_0_x_ligand, loss_0_x_pocket, loss_0_h, \
            neg_log_const_0, kl_prior = self.ddpm(ligand, pocket, esmc=esmc)  # NEW

        # ... (rest unchanged)
```

**Note**: Also need to modify `EnVariationalDiffusion` and `ConditionalDDPM` to accept and pass `esmc` to dynamics.

**Action Items**:
- [ ] Add `use_esmc` and `esmc_dim` parameters
- [ ] Pass `esmc_dim` to dynamics initialization
- [ ] Pass `use_esmc` to dataset
- [ ] Extract ESM-C from batch in training/validation steps
- [ ] Modify DDPM classes to accept and pass ESM-C

---

### Step 12: Create ESM-C Config File

**File**: `configs/crossdock_fullatom_cond_esmc.yml`

```yaml
run_name: "ESM-C-augmented"
logdir: "logs"
wandb_params:
  mode: "online"
  entity: "your-wandb-username"
  group: "crossdock-esmc"

dataset: "crossdock"
datadir: "data/processed_crossdock_esmc"  # NEW: ESM-C augmented data
use_esmc: True  # NEW FLAG
esmc_dim: 960  # NEW: ESM-2 is 1280, ESM-C is 960

mode: "pocket_conditioning"
pocket_representation: "full-atom"
virtual_nodes: False

batch_size: 8  # Reduced (larger encoder)
lr: 1.0e-3
n_epochs: 500
num_workers: 0
gpus: 4
clip_grad: True

egnn_params:
  joint_nf: 128
  hidden_nf: 256
  n_layers: 6
  attention: True
  edge_cutoff_pocket: 5.0
  edge_cutoff_interaction: 5.0

diffusion_params:
  diffusion_steps: 500
  diffusion_noise_schedule: "polynomial_2"
  diffusion_loss_type: "l2"
  normalize_factors: [1, 4]

eval_epochs: 25
visualize_sample_epoch: 50
eval_params:
  n_eval_samples: 100
  eval_batch_size: 50
```

**Action Items**:
- [ ] Create config file
- [ ] Update `datadir` to ESM-C augmented path
- [ ] Set `use_esmc: True` and `esmc_dim` appropriately
- [ ] Reduce batch size if needed (start with 8)

---

## Phase 3: Testing (Week 3)

### Step 13: Debug Run (Small Subset)

**Purpose**: Verify implementation works before full training

**Create debug config**: `configs/crossdock_fullatom_cond_esmc_debug.yml`
```yaml
# Copy from crossdock_fullatom_cond_esmc.yml
# Modify:
n_epochs: 10
batch_size: 4
eval_epochs: 5
```

**Create small subset**:
```bash
python -c "
import numpy as np

# Load full data
data = np.load('data/processed_crossdock_esmc/train.npz', allow_pickle=True)

# Take first 100 samples
subset = {}
for key in data.keys():
    if key in ['names', 'receptors']:
        subset[key] = data[key][:100]
    elif 'mask' in key:
        # Find cutoff for 100 samples
        cutoff = np.where(data[key] >= 100)[0][0]
        subset[key] = data[key][:cutoff]
    else:
        if key.startswith('lig'):
            cutoff = np.where(data['lig_mask'] >= 100)[0][0]
        else:
            cutoff = np.where(data['pocket_mask'] >= 100)[0][0]
        subset[key] = data[key][:cutoff]

np.savez('data/processed_crossdock_esmc/train_debug.npz', **subset)
print('Created debug subset: 100 samples')
"
```

**Run debug training**:
```bash
python train.py \
    --config configs/crossdock_fullatom_cond_esmc_debug.yml \
    --datadir data/processed_crossdock_esmc
```

**What to check**:
- [ ] No shape errors in forward pass
- [ ] No NaN losses
- [ ] GPU memory usage acceptable
- [ ] Loss decreases (even slightly)
- [ ] Validation runs without errors
- [ ] Sample generation works

**Common Issues & Fixes**:

| Issue | Fix |
|-------|-----|
| Shape mismatch in encoder | Check `encoder_input_dim = residue_nf + esmc_dim` |
| OOM error | Reduce batch_size to 2 or 4 |
| NaN losses | Adjust `normalize_factors` to [0.1, 1] |
| ESM-C not found in batch | Verify dataset `use_esmc=True` |

---

### Step 14: Small-scale Test (1K Samples)

**Purpose**: Verify training dynamics are reasonable

**Create 1K subset**:
```bash
# Similar to debug subset but with 1000 samples
python create_subset.py --n_samples 1000 --output train_1k.npz
```

**Config**: `configs/crossdock_fullatom_cond_esmc_1k.yml`
```yaml
# Same as main config but:
n_epochs: 50
datadir: "data/processed_crossdock_esmc_1k"
```

**Run**:
```bash
python train.py --config configs/crossdock_fullatom_cond_esmc_1k.yml
```

**Monitor**:
- [ ] Training loss curve (should decrease)
- [ ] Validation loss (should track training)
- [ ] Generated molecules (validity > 0%)
- [ ] No catastrophic failures

**Success Criteria**:
- Trains without crashes for 50 epochs
- Validation loss decreases or plateaus
- Can generate at least some valid molecules (>10% validity)

---

## Phase 4: Full Training (Week 3-5)

### Step 15: Submit HPC Training Job

**PBS Script**: `.claude/hpc/train_esmc.pbs`

```bash
#!/bin/bash
#PBS -N train_esmc
#PBS -l select=1:ncpus=16:ngpus=4:mem=128gb
#PBS -l walltime=336:00:00  # 14 days
#PBS -A your_project_id

cd $PBS_O_WORKDIR
module load cuda/11.8
module load python/3.9

# Activate environment
source venv/bin/activate

# Run training
python train.py \
    --config configs/crossdock_fullatom_cond_esmc.yml \
    --gpus 4 \
    --num_workers 4

echo "Training complete!"
```

**Submit**:
```bash
qsub .claude/hpc/train_esmc.pbs
```

**Monitor**:
```bash
# Check job status
qstat -u $USER

# Monitor WandB
# https://wandb.ai/your-entity/crossdock-esmc

# Check log files
tail -f train_esmc.o<job_id>
```

**Action Items**:
- [ ] Create PBS script
- [ ] Submit job
- [ ] Monitor via WandB
- [ ] Check for errors in first few epochs
- [ ] Let run for ~10-14 days

---

### Step 16: Baseline Comparison (Parallel)

**Important**: While ESM-C model trains, evaluate baseline for comparison

**Use existing checkpoint**:
```bash
# Baseline is already trained
checkpoint="checkpoints/crossdocked_fullatom_cond.ckpt"

# Generate on test set
python generate_ligands.py $checkpoint \
    --test_set data/processed_crossdock_noH_full/test.npz \
    --n_samples 100 \
    --outdir results/baseline/
```

**Compute metrics**:
```bash
python analyze_results.py \
    --checkpoint $checkpoint \
    --test_set data/processed_crossdock_noH_full/test.npz \
    --output results/baseline_metrics.json
```

**Action Items**:
- [ ] Load existing baseline checkpoint
- [ ] Generate 100 molecules per test pocket
- [ ] Compute validity, uniqueness, QED, SA scores
- [ ] Run Vina docking if available
- [ ] Save results for comparison

---

## Phase 5: Evaluation (Week 5-6)

### Step 17: Generate with ESM-C Model

**After training completes**:

```bash
# Load best checkpoint
checkpoint="logs/ESM-C-augmented/checkpoints/best.ckpt"

# Generate on test set
python generate_ligands.py $checkpoint \
    --test_set data/processed_crossdock_esmc/test.npz \
    --n_samples 100 \
    --outdir results/esmc/
```

---

### Step 18: Comparative Analysis

**Metrics to compute**:
- Validity (% RDKit-valid)
- Uniqueness (% unique SMILES)
- Novelty (% not in training)
- QED (drug-likeness)
- SA Score (synthetic accessibility)
- Vina docking scores
- Tanimoto similarity to reference

**Statistical tests**:
```python
from scipy.stats import wilcoxon

# Paired comparison (same pockets)
baseline_vina = [...]  # Baseline Vina scores
esmc_vina = [...]      # ESM-C Vina scores

statistic, p_value = wilcoxon(baseline_vina, esmc_vina)
print(f"p-value: {p_value}")
if p_value < 0.05:
    print("✓ Statistically significant improvement!")
```

**Action Items**:
- [ ] Compute all metrics for both models
- [ ] Run statistical tests
- [ ] Create comparison tables and plots
- [ ] Analyze failure cases

---

## Timeline Summary

| Week | Phase | Key Deliverables | Estimated Hours |
|------|-------|------------------|-----------------|
| 1 | Data Re-processing | Modified NPZ with residue IDs | 8-16 |
| 1-2 | ESM-C Pre-computation | Augmented NPZ with embeddings | 4-8 |
| 2 | Implementation | Modified codebase | 8-16 |
| 3 | Testing | Debug + 1K sample runs | 12 |
| 3-5 | Full Training | Trained model (HPC) | - |
| 5-6 | Evaluation | Metrics + comparison | 8-16 |

**Total hands-on time**: 40-68 hours
**Total elapsed time**: 5-6 weeks

---

## Success Criteria

### Minimum (Thesis-worthy)
- ✅ Model trains without crashes
- ✅ Data processing works correctly
- ✅ Any metric improves by >5%

### Good
- ✅ 2+ metrics improve (p < 0.05)
- ✅ Validity or QED increases
- ✅ Vina scores improve in tail distribution

### Excellent
- ✅ All metrics improve
- ✅ Mean Vina shift >0.5 kcal/mol
- ✅ Novel high-affinity molecules generated

---

## Getting Help

**Validation scripts**:
- `experiments/validate_esmc_integration.py` - Run all validation tests
- `experiments/example_residue_id_tracking.py` - Code modification examples

**Documentation**:
- `.claude/ESM_C_INTEGRATION_STRATEGY.md` - Full technical strategy
- `.claude/PLAN.md` - Thesis plan and timeline
- `experiments/VALIDATION_SUMMARY.md` - Detailed validation results

**Common issues**:
- See `experiments/VALIDATION_SUMMARY.md` section "Potential Issues & Mitigations"

---

## Checklist

### Phase 0A: Baseline Evaluation (PARALLEL - Week 1)
- [ ] Step 0.1: Verify baseline checkpoint
- [ ] Step 0.2: Generate baseline molecules (10 test proteins)

### Phase 0B: Data Re-processing (Week 1)
- [ ] Step 1: Modify process_crossdock.py
- [ ] Step 2: Test on single PDB
- [ ] Step 3: Re-process full dataset
- [ ] Step 4: Validate re-processed data

### Phase 1: ESM-C Pre-computation (Week 1-2)
- [ ] Step 5: Install ESM-C
- [ ] Step 6: Write precompute_esmc.py
- [ ] Step 7: Run ESM-C pre-computation
- [ ] Step 8: Validate ESM-C embeddings

### Phase 2: Implementation (Week 2)
- [ ] Step 9: Modify dynamics.py
- [ ] Step 10: Modify dataset.py
- [ ] Step 11: Modify lightning_modules.py
- [ ] Step 12: Create ESM-C config file

### Phase 3: Testing (Week 3)
- [ ] Step 13: Debug run (100 samples)
- [ ] Step 14: Small-scale test (1K samples)

### Phase 4: Full Training (Week 3-5)
- [ ] Step 15: Submit HPC training job
- [ ] Step 16: Monitor training progress

### Phase 5: Evaluation (Week 5-6)
- [ ] Step 17: Generate with ESM-C model
- [ ] Step 18: Comparative analysis (vs baseline)

---

**Current Status**: Ready to start Phase 0A and 0B (in parallel)
**Next Actions**:
1. **Step 0.1**: Verify baseline checkpoint (can start immediately)
2. **Step 1**: Modify `process_crossdock.py` (can start immediately)
