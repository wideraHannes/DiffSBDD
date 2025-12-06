# Code Reference: ESM-C Integration Patterns

> Quick reference for configuration, code patterns, and integration points.

---

## 1. Configuration System

### Master Flag: `esmc_conditioning`

This single flag controls all ESM-C behavior:

```yaml
# BASELINE MODE (original DiffSBDD)
esmc_conditioning: False
esmc_path: null

# ESM-C MODE (with conditioning)
esmc_conditioning: True
esmc_dim: 960
esmc_path: "path/to/embeddings.npz"
```

### Config File Locations

| Config | Purpose |
|--------|---------|
| `configs/crossdock_fullatom_cond.yml` | Baseline (production) |
| `thesis_work/experiments/day3_overfit/configs/` | Overfit experiments |

### Key Config Parameters

```yaml
# Model architecture
egnn_params:
  joint_nf: 128          # Feature dimension (FiLM γ, β size)
  hidden_nf: 256         # EGNN hidden size
  n_layers: 6            # EGNN depth

# Diffusion
diffusion_params:
  diffusion_steps: 500   # Denoising steps
  diffusion_loss_type: "l2"

# Training
batch_size: 8
lr: 1.0e-3
n_epochs: 1000

# Evaluation
eval_epochs: 50
eval_params:
  n_eval_samples: 100
```

---

## 2. Code Integration Points

### Files Modified for ESM-C

| File | Lines Changed | What |
|------|---------------|------|
| `dataset.py` | ~20 | Load embeddings from .npz |
| `dynamics.py` | ~30 | FiLM network, apply conditioning |
| `conditional_model.py` | ~10 | Pass pocket_emb (4 locations) |
| `en_diffusion.py` | ~5 | Pass pocket_emb (2 locations) |
| `lightning_modules.py` | ~20 | Handle esmc_path config |

### Where dynamics() is Called

Update all 6 locations to pass `pocket_emb`:

```python
# conditional_model.py (4 calls)
Line ~253: self.dynamics(..., pocket_emb=pocket.get('emb'))
Line ~306: self.dynamics(..., pocket_emb=pocket.get('emb'))
Line ~445: self.dynamics(..., pocket_emb=pocket.get('emb'))
Line ~119: self.dynamics(..., pocket_emb=pocket.get('emb'))

# en_diffusion.py (2 calls)
Line ~516: self.dynamics(..., pocket_emb=pocket.get('emb'))
Line ~270: self.dynamics(..., pocket_emb=pocket.get('emb'))
```

---

## 3. Core Code Patterns

### Dataset: Loading ESM-C Embeddings

```python
# dataset.py
class ProcessedLigandPocketDataset(Dataset):
    def __init__(self, data_path, esmc_path=None, ...):
        self.data = np.load(data_path, allow_pickle=True)
        self.esmc = None
        if esmc_path:
            self.esmc = np.load(esmc_path, allow_pickle=True)
    
    def __getitem__(self, idx):
        out = {
            'lig_coords': ...,
            'pocket_coords': ...,
            # ... standard fields
        }
        
        if self.esmc is not None:
            pocket_name = self.data['names'][idx]
            out['pocket_emb'] = torch.from_numpy(
                self.esmc[pocket_name]
            ).float()  # (960,)
        
        return out
```

### Dynamics: FiLM Conditioning

```python
# dynamics.py
class EGNNDynamics(nn.Module):
    def __init__(self, ..., esmc_conditioning=False, esmc_dim=960):
        super().__init__()
        self.esmc_conditioning = esmc_conditioning
        
        # FiLM network (only if enabled)
        if esmc_conditioning:
            self.pocket_film = nn.Sequential(
                nn.Linear(esmc_dim, esmc_dim),
                nn.SiLU(),
                nn.Linear(esmc_dim, joint_nf * 2)  # γ and β
            )
    
    def forward(self, xh_atoms, xh_residues, t, ..., pocket_emb=None):
        # Encode features
        h_atoms = self.atom_encoder(h_atoms)  # (N, 128)
        
        # Apply FiLM conditioning
        if self.esmc_conditioning and pocket_emb is not None:
            film_out = self.pocket_film(pocket_emb)  # (256,)
            gamma = film_out[..., :self.joint_nf]    # (128,)
            beta = film_out[..., self.joint_nf:]     # (128,)
            h_atoms = gamma * h_atoms + beta         # Modulate!
        
        # Continue with EGNN...
```

### Safe Parameter Passing

```python
# Always use .get() for optional pocket_emb
pocket_emb = pocket.get('emb', None)

# Pass to dynamics
self.dynamics(..., pocket_emb=pocket_emb)
```

---

## 4. ESM-C Embedding Extraction

### Script: `esmc_integration/extraction/extract_esmc_embeddings.py`

```python
from esm.sdk import client

# Initialize ESM-C
model = client(
    model="esmc-300m-2024-12",
    url="https://forge.evolutionaryscale.ai",
    token=YOUR_TOKEN
)

# Extract embedding for one pocket
def extract_pocket_embedding(protein_sequence, pocket_residue_ids):
    # 1. Get full protein embeddings
    embeddings = model.encode(protein_sequence)  # (N_total, 960)
    
    # 2. Extract pocket residues
    pocket_embs = embeddings[pocket_residue_ids]  # (N_pocket, 960)
    
    # 3. Mean pool to single vector
    global_emb = pocket_embs.mean(dim=0)  # (960,)
    
    return global_emb
```

### Cached Embedding Format

```python
# Saved as .npz file
embeddings = {
    "pocket_name_1": np.array([...]),  # shape: (960,)
    "pocket_name_2": np.array([...]),
    ...
}
np.savez("esmc_embeddings.npz", **embeddings)
```

---

## 5. Debugging Patterns

### Check if ESM-C is Active

```python
# In dynamics.py
print(f"ESM-C conditioning: {self.esmc_conditioning}")
print(f"FiLM network exists: {hasattr(self, 'pocket_film')}")

# In training
if pocket_emb is not None:
    print(f"Pocket embedding shape: {pocket_emb.shape}")
    print(f"Embedding stats: mean={pocket_emb.mean():.3f}, std={pocket_emb.std():.3f}")
```

### Check FiLM Parameters

```python
# After training
gamma, beta = model.dynamics.pocket_film(pocket_emb).chunk(2, dim=-1)
print(f"γ: mean={gamma.mean():.3f}, std={gamma.std():.3f}")
print(f"β: mean={beta.mean():.3f}, std={beta.std():.3f}")

# If γ ≈ 1 and β ≈ 0, model is ignoring ESM-C
```

### Check Gradient Flow

```python
# Verify FiLM network is learning
for name, param in model.dynamics.pocket_film.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad_norm={param.grad.norm():.4f}")
```

---

## 6. Training Commands

### Baseline Training
```bash
uv run python train.py --config configs/crossdock_fullatom_cond.yml
```

### ESM-C Training
```bash
uv run python train.py --config configs/crossdock_fullatom_cond_esmc.yml
```

### Overfit Test (Current)
```bash
# Create dataset
uv run python thesis_work/experiments/day3_overfit/create_overfit_dataset.py \
    --n_train 5 --n_val 2

# Train
uv run python train.py \
    --config thesis_work/experiments/day3_overfit/configs/day3_overfit_5sample.yml
```

### Generation
```bash
uv run python generate_ligands.py \
    checkpoints/model.ckpt \
    --pdbfile protein.pdb \
    --ref_ligand A:LIG \
    --n_samples 100 \
    --outdir output/
```

---

## 7. File Structure Reference

```
DiffSBDD/
├── train.py                    # Training entry point
├── generate_ligands.py         # Inference entry point
├── test.py                     # Evaluation
├── dataset.py                  # Data loading (+ESM-C)
├── lightning_modules.py        # Training orchestration
├── constants.py                # Atom types, bond tables
│
├── equivariant_diffusion/
│   ├── dynamics.py            # Denoising network (+FiLM)
│   ├── en_diffusion.py        # Diffusion model (joint)
│   ├── conditional_model.py   # Diffusion model (conditional)
│   └── egnn_new.py            # EGNN implementation
│
├── analysis/
│   ├── metrics.py             # Validity, connectivity, QED, SA
│   └── molecule_builder.py    # Bond inference, RDKit
│
├── esmc_integration/
│   ├── extraction/            # ESM-C embedding scripts
│   └── tests/                 # Integration tests
│
├── configs/                   # Training configs
└── thesis_work/experiments/   # Experiment-specific configs
```

---

## 8. Metric Computation

### Where Metrics Are Computed

```python
# analysis/metrics.py
class BasicMolecularMetrics:
    def compute_validity(mol)      # RDKit sanitization
    def compute_connectivity(mol)  # Largest fragment = 100%
    def compute_uniqueness(mols)   # Unique SMILES
    def compute_novelty(mols)      # Not in training set

class MoleculeProperties:
    def compute_qed(mol)           # Drug-likeness (0-1)
    def compute_sa(mol)            # Synth. accessibility (1-10)
```

### Understanding Connectivity

```python
# Molecule is "connected" if largest fragment has ALL atoms
from rdkit import Chem

def is_connected(mol):
    frags = Chem.GetMolFrags(mol, asMols=True)
    largest = max(frags, key=lambda m: m.GetNumAtoms())
    return largest.GetNumAtoms() == mol.GetNumAtoms()
```

**0% Connectivity** means atoms are too far apart for bonds to form. Check:
1. Atom distance distribution (should be 1-2 Å for bonds)
2. Loss value (should be < 0.2 for good atom positions)

---

*For full verbose documentation, see `.claude/archive/`*
