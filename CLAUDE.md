# DiffSBDD + ESM-C: Thesis Project Guide

> **Project Goal**: Enhance structure-based drug design by conditioning diffusion models on protein language model embeddings.

---

## Quick Reference

**Main Thesis Plan**: See `.claude/THESIS_PLAN.md` for complete details

**Core Analogy**:
```
Text Prompt → CLIP → Stable Diffusion → Image
     ↓           ↓            ↓            ↓
Pocket Seq  →  ESM-C  →   DiffSBDD   →  Ligand
```

**Key Insight**: Use a **single global 960-dim ESM-C embedding** per pocket as a steering signal via FiLM conditioning.

---

## Project Structure

```
DiffSBDD/
├── .claude/                          # Thesis planning documents
│   ├── THESIS_PLAN.md               # ⭐ Main thesis plan (read this first!)
│   ├── 06_GLOBAL_POCKET_CONDITIONING.md  # Why global embedding is brilliant
│   ├── 05_ESM_C_INTEGRATION_PLAN.md     # Original per-residue plan (deprecated)
│   └── README.md                     # Original codebase overview
│
├── data/                            # Datasets
│   ├── crossdocked_noH_full/       # Processed training data
│   ├── esmc_train.npz              # ESM-C embeddings (to be generated)
│   ├── esmc_val.npz
│   └── esmc_test.npz
│
├── equivariant_diffusion/          # Core model code
│   ├── dynamics.py                 # ⚡ MODIFY: Add FiLM conditioning here
│   ├── en_diffusion.py             # UPDATE: Pass ESM-C through
│   ├── conditional_model.py        # UPDATE: Pass ESM-C through
│   └── egnn_new.py                 # EGNN layers (no changes needed)
│
├── dataset.py                       # ⚡ MODIFY: Load ESM-C embeddings
├── lightning_modules.py             # Training loop
├── train.py                         # Training entry point
├── generate_ligands.py              # Inference entry point
│
├── configs/                         # Training configurations
│   ├── crossdock_fullatom_cond.yml      # Baseline config
│   └── crossdock_fullatom_cond_esmc.yml # ⚡ NEW: ESM-C config
│
├── analysis/                        # Evaluation code
│   ├── metrics.py                  # Validity, QED, SA, etc.
│   ├── molecule_builder.py         # RDKit molecule construction
│   └── docking.py                  # Vina docking scores
│
└── scripts/                         # ⚡ NEW: Helper scripts
    ├── extract_esmc_embeddings.py  # Phase 1: Extract and cache ESM-C
    ├── evaluate.py                 # Phase 6: Compare baseline vs ESM-C
    └── analyze_film.py             # Phase 7: Analyze FiLM parameters
```

---

## Current Status

**Phase**: Planning ✅ → Implementation (ready to start)

**Baseline**: Authors provide trained weights (no need to retrain!)
- Checkpoint: `checkpoints/crossdocked_fullatom_cond.ckpt`
- Dataset: CrossDocked (100k training pairs, 100 test proteins)

**Next Steps**:
1. ✅ Finalize thesis plan
2. ⏳ Set up ESM-C extraction
3. ⏳ Modify codebase for FiLM conditioning
4. ⏳ Train ESM-C variant
5. ⏳ Evaluate and analyze

---

## Key Architecture Details

### Current DiffSBDD (Baseline)

**File**: `equivariant_diffusion/dynamics.py:87-186`

```python
def forward(self, xh_atoms, xh_residues, t, mask_atoms, mask_residues):
    # Extract coords and features
    x_atoms, h_atoms = xh_atoms[:, :3], xh_atoms[:, 3:]
    x_residues, h_residues = xh_residues[:, :3], xh_residues[:, 3:]

    # Encode to joint space
    h_atoms = self.atom_encoder(h_atoms)        # (N_lig, 128)
    h_residues = self.residue_encoder(h_residues)  # (N_pocket, 128)

    # Unified graph
    x = torch.cat((x_atoms, x_residues), dim=0)
    h = torch.cat((h_atoms, h_residues), dim=0)
    mask = torch.cat([mask_atoms, mask_residues])

    # Add timestep
    h_time = t[mask]
    h = torch.cat([h, h_time], dim=1)  # (N_total, 129)

    # EGNN message passing
    h_final, x_final = self.egnn(h, x, edges, ...)

    # Decode
    h_final_atoms = self.atom_decoder(h_final[:len(mask_atoms)])
    h_final_residues = self.residue_decoder(h_final[len(mask_atoms):])

    return ...
```

**Current pocket representation**:
- Coordinates: `(N_pocket, 3)` - 3D positions
- Features: `(N_pocket, 20)` - One-hot amino acid types
- **Information**: Only amino acid identity (ALA, VAL, LEU, ...)

**Limitation**: No evolutionary, structural, or functional context!

---

### Proposed Architecture (+ ESM-C)

**Modified**: `equivariant_diffusion/dynamics.py`

```python
class EGNNDynamics(nn.Module):
    def __init__(self, ..., esmc_dim=960):
        # Existing encoders
        self.atom_encoder = nn.Sequential(...)
        self.residue_encoder = nn.Sequential(...)

        # ⚡ NEW: FiLM conditioning network
        self.pocket_film = nn.Sequential(
            nn.Linear(esmc_dim, esmc_dim),
            nn.SiLU(),
            nn.Linear(esmc_dim, joint_nf * 2)  # γ and β
        )

        self.egnn = EGNN(...)
        # ... rest unchanged

    def forward(self, xh_atoms, xh_residues, pocket_emb, t, mask_atoms, mask_residues):
        #                                      ^^^^^^^^^^^ NEW parameter

        # Encode features
        h_atoms = self.atom_encoder(h_atoms)      # (N_lig, 128)
        h_residues = self.residue_encoder(h_residues)  # (N_pocket, 128)

        # ⚡ NEW: FiLM conditioning from ESM-C
        film_params = self.pocket_film(pocket_emb)  # (960,) → (256,)
        gamma = film_params[:self.joint_nf]   # (128,)
        beta = film_params[self.joint_nf:]    # (128,)

        # Apply to ligand features (steering!)
        h_atoms = gamma.unsqueeze(0) * h_atoms + beta.unsqueeze(0)

        # Rest unchanged: unified graph, EGNN, decode
        x = torch.cat((x_atoms, x_residues), dim=0)
        h = torch.cat((h_atoms, h_residues), dim=0)
        ...
```

**New pocket representation**:
- Coordinates: `(N_pocket, 3)` - 3D positions (unchanged)
- Features: `(N_pocket, 20)` - One-hot AA types (unchanged)
- **NEW**: ESM-C embedding: `(960,)` - Global evolutionary context

**Information added**:
- Evolutionary conservation
- Structural motifs from full protein
- Functional context (binding site character)
- Physicochemical patterns

---

## How FiLM Steering Works

### Mathematical View

**Input**:
```
h_ligand: (N_lig, 128) - Ligand node features
pocket_emb: (960,) - ESM-C global embedding
```

**FiLM Network**:
```python
# MLP projects to scale and shift
[γ, β] = MLP(pocket_emb)  # γ, β ∈ ℝ^128

# Modulate ligand features
h_ligand' = γ ⊙ h_ligand + β  # Element-wise
```

**Interpretation**:
- **γ (scale)**: Amplifies or suppresses feature channels
  - γ > 1: Enhance this feature
  - γ < 1: Suppress this feature
- **β (shift)**: Adds bias to features
  - Shifts activation patterns

**Why this works**:
- Multiplicative conditioning is stronger than additive
- Applied uniformly to all ligand atoms
- Differentiable → backprop updates ESM-C influence
- Proven in StyleGAN, FiLM-GAN, image conditioning

### Geometric vs Semantic Conditioning

| Type | Source | Mechanism | Purpose |
|------|--------|-----------|---------|
| **Geometric** | Pocket coords + EGNN | Message passing | Prevents clashes, respects distances |
| **Semantic** | ESM-C embedding + FiLM | Feature modulation | Guides toward realistic binders |

**Clean separation**: Geometry ensures validity, semantics ensures quality!

---

## Implementation Roadmap

### ✅ Phase 0: Baseline Validation (Week 0)

**Goal**: Confirm baseline model works

```bash
# Download checkpoint
wget https://zenodo.org/record/8183747/files/crossdocked_fullatom_cond.ckpt

# Test generation
python generate_ligands.py \
    checkpoints/crossdocked_fullatom_cond.ckpt \
    --pdbfile example/3rfm.pdb \
    --ref_ligand A:330 \
    --n_samples 10 \
    --outfile baseline_test.sdf
```

**Success criteria**: Generates 10 valid molecules

---

### ⏳ Phase 1: ESM-C Extraction (Week 1)

**Goal**: Create cached embeddings for all pockets

**Script**: `scripts/extract_esmc_embeddings.py`

```python
from esm.models.esmc import ESMC
from Bio import PDB
import torch
import numpy as np

def extract_global_pocket_embedding(pdb_file, pocket_residue_ids):
    """
    Extract 960-dim global embedding for a pocket

    Args:
        pdb_file: Path to PDB structure
        pocket_residue_ids: List of residue indices in pocket

    Returns:
        embedding: (960,) numpy array
    """
    # 1. Parse structure and get full sequence
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_file)
    chain = list(structure.get_chains())[0]
    full_sequence = ''.join([res.get_resname() for res in chain])

    # 2. Run ESM-C on FULL protein (preserves context!)
    model = ESMC.from_pretrained("esmc_600m").eval().cuda()
    with torch.no_grad():
        embeddings = model.encode(full_sequence)  # (N_total, 960)

    # 3. Extract pocket residues
    pocket_embeddings = embeddings[pocket_residue_ids]  # (N_pocket, 960)

    # 4. Global mean pooling
    global_emb = pocket_embeddings.mean(dim=0).cpu().numpy()  # (960,)

    return global_emb

# Process all splits
for split in ['train', 'val', 'test']:
    print(f"Processing {split}...")
    data = np.load(f'data/crossdocked_noH_full/{split}.npz', allow_pickle=True)
    cache = {}

    for i, receptor_name in enumerate(data['receptors']):
        if i % 100 == 0:
            print(f"  {i}/{len(data['receptors'])}")

        pdb_file = f'data/crossdocked_noH_full/proteins/{receptor_name}.pdb'
        pocket_res_ids = get_pocket_residue_ids(data, i)  # Define pocket

        cache[receptor_name] = extract_global_pocket_embedding(pdb_file, pocket_res_ids)

    np.savez(f'data/esmc_{split}.npz', **cache)
    print(f"Saved data/esmc_{split}.npz")
```

**Run**:
```bash
uv run python scripts/extract_esmc_embeddings.py
```

**Output**:
- `data/esmc_train.npz` (~400 MB)
- `data/esmc_val.npz` (~10 MB)
- `data/esmc_test.npz` (~10 MB)

**Validation**:
```python
# Check one embedding
esmc_data = np.load('data/esmc_test.npz', allow_pickle=True)
first_receptor = list(esmc_data.keys())[0]
emb = esmc_data[first_receptor]

print(f"Shape: {emb.shape}")  # Should be (960,)
print(f"Mean: {emb.mean():.3f}")  # Should be ~0
print(f"Std: {emb.std():.3f}")   # Should be ~1
```

---

### ⏳ Phase 2: Modify Dataset (Week 1-2)

**Goal**: Load ESM-C alongside ligand/pocket data

**File**: `dataset.py:12-71`

```python
class ProcessedLigandPocketDataset(Dataset):
    def __init__(self, data_path, esmc_path=None, ...):
        """
        Args:
            data_path: Path to .npz with coordinates and one-hot
            esmc_path: Path to .npz with ESM-C embeddings (NEW!)
        """
        self.data = np.load(data_path, allow_pickle=True)

        # ⚡ NEW: Load ESM-C embeddings if provided
        self.esmc = None
        if esmc_path is not None:
            self.esmc = np.load(esmc_path, allow_pickle=True)
            print(f"Loaded ESM-C embeddings from {esmc_path}")

        # ... rest unchanged

    def __getitem__(self, idx):
        # Original fields
        out = {
            'lig_coords': ...,
            'lig_one_hot': ...,
            'pocket_coords': ...,
            'pocket_one_hot': ...,
            'lig_mask': ...,
            'pocket_mask': ...,
            'names': ...,
        }

        # ⚡ NEW: Add ESM-C embedding if available
        if self.esmc is not None:
            receptor_name = self.data['receptors'][idx]
            out['pocket_emb'] = torch.from_numpy(
                self.esmc[receptor_name]
            ).float()  # (960,)

        return out
```

**Modify**: `lightning_modules.py` to pass `esmc_path`:

```python
# lightning_modules.py:162-174
def setup(self, stage=None):
    # Determine ESM-C path if enabled
    esmc_path = None
    if self.esmc_conditioning:
        esmc_path = f'data/esmc_{stage}.npz'

    self.train_dataset = ProcessedLigandPocketDataset(
        data_path='data/crossdocked_noH_full/train.npz',
        esmc_path=esmc_path,  # NEW
        ...
    )
```

**Test**:
```python
dataset = ProcessedLigandPocketDataset(
    'data/crossdocked_noH_full/test.npz',
    'data/esmc_test.npz'
)
sample = dataset[0]
print(sample.keys())  # Should include 'pocket_emb'
print(sample['pocket_emb'].shape)  # Should be torch.Size([960])
```

---

### ⏳ Phase 3: Add FiLM Conditioning (Week 2-3)

**Goal**: Modify dynamics network to apply ESM-C conditioning

**File**: `equivariant_diffusion/dynamics.py`

**Changes**:

1. **Add FiLM network in `__init__`** (line ~50):
```python
# After self.residue_decoder
self.pocket_film = nn.Sequential(
    nn.Linear(960, 960),
    nn.SiLU(),
    nn.Linear(960, joint_nf * 2)
) if esmc_conditioning else None
```

2. **Update `forward` signature** (line 87):
```python
def forward(self, xh_atoms, xh_residues, pocket_emb=None, t=None,
            mask_atoms=None, mask_residues=None):
    #                                      ^^^^^^^^^^^ NEW
```

3. **Apply FiLM conditioning** (after line 101):
```python
h_atoms = self.atom_encoder(h_atoms)
h_residues = self.residue_encoder(h_residues)

# ⚡ NEW: Apply FiLM conditioning if ESM-C provided
if pocket_emb is not None and self.pocket_film is not None:
    film_params = self.pocket_film(pocket_emb)  # (960,) → (256,)
    gamma = film_params[..., :self.joint_nf]   # (..., 128)
    beta = film_params[..., self.joint_nf:]    # (..., 128)

    # Broadcast and apply to ligand atoms
    # gamma/beta shape: (batch_size, 128) or (128,)
    # h_atoms shape: (N_lig_total, 128)
    # Need to expand correctly based on mask_atoms

    # Get batch indices for each atom
    batch_idx = mask_atoms  # (N_lig_total,)
    gamma_expanded = gamma[batch_idx]  # (N_lig_total, 128)
    beta_expanded = beta[batch_idx]    # (N_lig_total, 128)

    # Apply FiLM
    h_atoms = gamma_expanded * h_atoms + beta_expanded

# Continue as normal
x = torch.cat((x_atoms, x_residues), dim=0)
h = torch.cat((h_atoms, h_residues), dim=0)
...
```

---

### ⏳ Phase 4: Update Diffusion Calls (Week 3)

**Goal**: Pass ESM-C through the pipeline

**Files to modify**:
- `equivariant_diffusion/en_diffusion.py`
- `equivariant_diffusion/conditional_model.py`

**Pattern**: Find all `self.dynamics(` calls and add `pocket_emb=...`

**Example** (`en_diffusion.py:~800`):
```python
# OLD:
eps_ligand, eps_pocket = self.dynamics(
    xh_ligand_t, xh_pocket_t, t, lig_mask, pocket_mask
)

# NEW:
eps_ligand, eps_pocket = self.dynamics(
    xh_ligand_t, xh_pocket_t,
    pocket_emb=pocket['emb'] if 'emb' in pocket else None,  # NEW
    t=t,
    mask_atoms=lig_mask,
    mask_residues=pocket_mask
)
```

**Test end-to-end**:
```bash
# Dry run training for 1 step
uv run python train.py \
    --config configs/crossdock_fullatom_cond_esmc.yml \
    --fast_dev_run True
```

---

### ⏳ Phase 5: Training (Week 4-5)

**Config**: `configs/crossdock_fullatom_cond_esmc.yml`

```yaml
# Copy from crossdock_fullatom_cond.yml and add:

# ESM-C conditioning
esmc_conditioning: True
esmc_dim: 960
esmc_train_path: "data/esmc_train.npz"
esmc_val_path: "data/esmc_val.npz"

# Training (same as baseline)
batch_size: 16
lr: 1.0e-3
n_epochs: 500
gpus: 1
clip_grad: True

# Model (same as baseline)
egnn_params:
  joint_nf: 128
  hidden_nf: 256
  n_layers: 6
  attention: True
  ...
```

**Run**:
```bash
uv run python train.py \
    --config configs/crossdock_fullatom_cond_esmc.yml \
    --run_name "diffsbdd_esmc_v1"
```

**Monitor**:
```bash
# Watch logs
tensorboard --logdir logs/

# Check WandB (if configured)
open https://wandb.ai/your-project
```

**Expected**:
- ~3-5 days on single GPU
- Loss should decrease similarly to baseline
- Validate metrics every N epochs

---

### ⏳ Phase 6: Evaluation (Week 6)

**Script**: `scripts/evaluate.py`

```python
import torch
from lightning_modules import LigandPocketDDPM
from analysis.metrics import BasicMolecularMetrics

# Load models
baseline = LigandPocketDDPM.load_from_checkpoint('checkpoints/baseline.ckpt')
esmc = LigandPocketDDPM.load_from_checkpoint('checkpoints/esmc.ckpt')

# Load test set
test_pockets = load_test_pockets()

results = {'baseline': [], 'esmc': []}

for pocket in test_pockets:
    # Generate 100 molecules per pocket
    mols_baseline = baseline.generate_ligands(pocket, n_samples=100)
    mols_esmc = esmc.generate_ligands(pocket, n_samples=100)

    # Compute metrics
    metrics_baseline = compute_all_metrics(mols_baseline, pocket)
    metrics_esmc = compute_all_metrics(mols_esmc, pocket)

    results['baseline'].append(metrics_baseline)
    results['esmc'].append(metrics_esmc)

# Statistical comparison
compare_results(results)
```

**Metrics**:
1. **Validity**: % RDKit-valid molecules
2. **Connectivity**: % connected graphs
3. **Uniqueness**: % unique SMILES
4. **Novelty**: % not in training set
5. **QED**: Drug-likeness (0-1, higher better)
6. **SA Score**: Synthetic accessibility (1-10, lower better)
7. **Vina Score**: Docking score (lower better)
8. **Diversity**: Average pairwise Tanimoto distance

---

### ⏳ Phase 7: Analysis (Week 7-8)

**FiLM Parameter Analysis**:
```python
# scripts/analyze_film.py

# Extract γ and β for all test samples
gammas, betas = [], []

for pocket in test_set:
    pocket_emb = load_esmc_embedding(pocket)
    film_params = model.dynamics.pocket_film(pocket_emb)
    gamma, beta = film_params.chunk(2, dim=-1)

    gammas.append(gamma.cpu().numpy())
    betas.append(beta.cpu().numpy())

# Visualize
plt.figure(figsize=(12, 4))
plt.subplot(121)
plt.plot(np.mean(gammas, axis=0), label='Mean γ')
plt.fill_between(range(128),
                 np.percentile(gammas, 25, axis=0),
                 np.percentile(gammas, 75, axis=0),
                 alpha=0.3)
plt.axhline(1.0, color='k', linestyle='--', label='Identity')
plt.xlabel('Feature Channel')
plt.ylabel('Scale (γ)')
plt.legend()

plt.subplot(122)
plt.plot(np.mean(betas, axis=0), label='Mean β')
plt.fill_between(range(128),
                 np.percentile(betas, 25, axis=0),
                 np.percentile(betas, 75, axis=0),
                 alpha=0.3)
plt.axhline(0.0, color='k', linestyle='--', label='No shift')
plt.xlabel('Feature Channel')
plt.ylabel('Shift (β)')
plt.legend()
plt.savefig('figures/film_parameters.png')
```

**Gradient Attribution**:
```python
# Which ESM-C dimensions matter most?
from captum.attr import IntegratedGradients

ig = IntegratedGradients(model)

for pocket in test_set:
    pocket_emb = load_esmc_embedding(pocket)

    # Attribute Vina score to ESM-C dimensions
    attributions = ig.attribute(pocket_emb, target=vina_score)

    # Visualize top dimensions
    top_dims = np.argsort(np.abs(attributions))[-20:]
    plt.barh(range(20), attributions[top_dims])
    plt.xlabel('Attribution to Vina Score')
    plt.ylabel('ESM-C Dimension')
```

---

## Troubleshooting

### Issue: ESM-C extraction is slow

**Solution**:
```python
# Use batch processing
embeddings = model.encode_batch(sequences, batch_size=32)

# Use smaller ESM model for testing
model = ESMC.from_pretrained("esmc_300m")  # Faster, 640-dim
```

### Issue: GPU out of memory during training

**Solution**:
```yaml
# In config:
batch_size: 8  # Reduce from 16
gradient_checkpointing: True
```

### Issue: FiLM parameters don't change (γ ≈ 1, β ≈ 0)

**Analysis**:
```python
# Model ignoring ESM-C - this IS a result!
# Means local geometry dominates
# Proceed with analysis of why
```

**Debugging**:
```python
# Check gradients flowing to FiLM
for name, param in model.dynamics.pocket_film.named_parameters():
    print(f"{name}: grad_norm = {param.grad.norm()}")

# If grad_norm ≈ 0: ESM-C not affecting loss
```

### Issue: No improvement over baseline

**This is okay!** Analyze why:
1. Is ESM-C signal present? (mutual information analysis)
2. Is model using it? (gradient flow, FiLM params)
3. Is dataset quality limiting? (test on Binding MOAD)
4. Is global pooling too aggressive? (try per-residue)

**Thesis contribution**: "We show that in CrossDocked, local geometry dominates binding..."

---

## Key Design Decisions

### Why Global Embedding (not per-residue)?

| Aspect | Per-Residue | Global (Chosen) |
|--------|-------------|-----------------|
| **Complexity** | Variable length | Fixed size |
| **Implementation** | Modify graph | Simple FiLM |
| **Conceptual** | Mixed geometric/semantic | Clean separation |
| **Analogy** | Less clear | **Text-to-image** ✓ |
| **Code changes** | ~150 lines | ~50 lines |
| **Risk** | Higher | Lower |

**Decision**: Start with global (simpler, faster). Can extend to per-residue later if needed.

### Why FiLM (not cross-attention)?

| Technique | Pros | Cons |
|-----------|------|------|
| **FiLM** (Chosen) | Simple, proven, fast, interpretable | Less expressive than attention |
| Cross-attention | More flexible, per-atom conditioning | Complex, slower, harder to debug |
| Adaptive LayerNorm | Clean, used in DiT | Less direct modulation |

**Decision**: FiLM for simplicity and proven effectiveness. Can try attention in future work.

### Why Mean Pooling (not max/attention)?

| Pooling | Pros | Cons |
|---------|------|------|
| **Mean** (Chosen) | Simple, smooth, no parameters | May lose important residues |
| Max | Captures salient features | Sensitive to outliers |
| Attention | Learnable, weighted | Extra parameters, complexity |

**Decision**: Start with mean (standard, simple). Ablate other pooling strategies in Phase 7.

---

## Success Metrics

### Minimum Viable (Pass Thesis)
- ✅ Implement ESM-C conditioning
- ✅ Train successfully
- ✅ Evaluate quantitatively
- ✅ Document approach clearly

### Good Thesis
- ✅ All minimum requirements
- ✅ Improvement OR insightful analysis
- ✅ Ablation studies
- ✅ Clear visualizations

### Excellent Thesis
- ✅ All good requirements
- ✅ Significant metric improvements
- ✅ Deep mechanistic analysis
- ✅ Publication-ready quality

---

## Important Reminders

1. **Baseline exists** - Don't retrain! Use authors' checkpoint.
2. **Either outcome is valuable** - Positive or negative result both publishable.
3. **Keep it simple** - Global embedding, FiLM conditioning, done.
4. **Document everything** - Keep lab notebook, save figures, track experiments.
5. **Maintain the analogy** - Always frame as "steering like text-to-image".
6. **Timeline is tight** - 8 weeks, stay focused, don't over-engineer.

---

## References

**Main Papers**:
- DiffSBDD: Schneuing et al., Nature Comp. Sci. 2024 (see `.claude/s43588-024-00737-x.pdf`)
- ESM-C: Hayes et al., bioRxiv 2024
- FiLM: Perez et al., AAAI 2018

**Code**:
- DiffSBDD: https://github.com/arneschneuing/DiffSBDD
- ESM: https://github.com/evolutionaryscale/esm

---

**Last Updated**: December 2024
**Status**: Planning complete, ready for implementation
