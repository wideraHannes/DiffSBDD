# Configuration Guide: Baseline vs ESM-C Conditioning

> **Key Principle**: The codebase must support both baseline (original DiffSBDD) and ESM-C conditioning modes through configuration flags, ensuring backward compatibility and fair comparisons.

---

## 1. Overview

### Design Goals

1. **Backward Compatibility**: Original baseline mode must work unchanged
2. **Clean Toggling**: Single flag controls ESM-C conditioning
3. **No Performance Overhead**: Baseline mode has zero ESM-C overhead
4. **Easy Comparison**: Same codebase, different configs
5. **Extensible**: Easy to add future conditioning methods

### Architecture Philosophy

```
Baseline Mode (esmc_conditioning: False):
  Pocket → One-hot + Coords → EGNN → Ligand

ESM-C Mode (esmc_conditioning: True):
  Pocket → One-hot + Coords + ESM-C Embedding → FiLM + EGNN → Ligand
```

---

## 2. Configuration System

### Master Configuration Flag

**Single source of truth**: `esmc_conditioning: bool`

This flag controls:
- Whether to load ESM-C embeddings from disk
- Whether to initialize FiLM conditioning network
- Whether to pass embeddings through the pipeline
- Which config file to use

### Configuration Files

#### Location: `configs/`

```
configs/
├── crossdock_fullatom_cond.yml        # Baseline (original)
├── crossdock_fullatom_cond_esmc.yml   # ESM-C variant
└── README.md                           # Config documentation
```

---

## 3. Configuration File Details

### Baseline Config: `crossdock_fullatom_cond.yml`

```yaml
# Original DiffSBDD configuration (unchanged)

# Dataset
dataset: "crossdocked_noH_full"
data_dir: "data/crossdocked_noH_full"

# ESM-C Conditioning
esmc_conditioning: False              # ← KEY FLAG: Baseline mode

# Model architecture (original)
model_type: "conditional_diffusion"
egnn_params:
  joint_nf: 128
  hidden_nf: 256
  n_layers: 6
  attention: True
  tanh: False
  norm_constant: 0
  inv_sublayers: 2
  sin_embedding: False
  aggregation_method: "sum"

# Dynamics
atom_nf: 27                            # Atom feature dimension
residue_nf: 20                         # Residue one-hot dimension
joint_nf: 128                          # Joint embedding dimension

# Training
batch_size: 16
lr: 1.0e-3
n_epochs: 500
lr_scheduler: "reduce_on_plateau"
clip_grad: True
max_grad_norm: 8.0

# Diffusion
timesteps: 500
noise_schedule: "learned"
noise_precision: 1.0e-5

# Evaluation
n_stability_samples: 500
eval_epochs: 20

# Hardware
gpus: 1
num_workers: 4
```

---

### ESM-C Config: `crossdock_fullatom_cond_esmc.yml`

```yaml
# ESM-C conditioned variant

# Dataset
dataset: "crossdocked_noH_full"
data_dir: "data/crossdocked_noH_full"

# ESM-C Conditioning
esmc_conditioning: True               # ← KEY FLAG: ESM-C mode
esmc_dim: 960                         # ESM-C embedding dimension
esmc_train_path: "data/esmc_train.npz"
esmc_val_path: "data/esmc_val.npz"
esmc_test_path: "data/esmc_test.npz"

# FiLM conditioning
film_hidden_dim: 960                  # MLP hidden dimension
film_use_layer_norm: False            # Optional layer norm in FiLM

# Model architecture (same as baseline)
model_type: "conditional_diffusion"
egnn_params:
  joint_nf: 128
  hidden_nf: 256
  n_layers: 6
  attention: True
  tanh: False
  norm_constant: 0
  inv_sublayers: 2
  sin_embedding: False
  aggregation_method: "sum"

# Dynamics (same as baseline)
atom_nf: 27
residue_nf: 20
joint_nf: 128

# Training (same as baseline)
batch_size: 16
lr: 1.0e-3
n_epochs: 500
lr_scheduler: "reduce_on_plateau"
clip_grad: True
max_grad_norm: 8.0

# Diffusion (same as baseline)
timesteps: 500
noise_schedule: "learned"
noise_precision: 1.0e-5

# Evaluation (same as baseline)
n_stability_samples: 500
eval_epochs: 20

# Hardware (same as baseline)
gpus: 1
num_workers: 4
```

---

## 4. Code Implementation Strategy

### Principle: Conditional Initialization

**Pattern**: Check `esmc_conditioning` flag, initialize accordingly

### Key Files to Modify

```
Modified files:
├── dataset.py                        # Conditionally load ESM-C
├── equivariant_diffusion/
│   ├── dynamics.py                   # Conditional FiLM network
│   ├── en_diffusion.py               # Pass embedding if present
│   └── conditional_model.py          # Pass embedding if present
└── lightning_modules.py              # Load ESM-C data if enabled
```

---

## 5. Implementation Details

### 5.1 Dataset (`dataset.py`)

```python
class ProcessedLigandPocketDataset(Dataset):
    def __init__(
        self,
        data_path,
        esmc_path=None,           # Optional: None for baseline
        esmc_conditioning=False,   # Flag from config
        ...
    ):
        self.data = np.load(data_path, allow_pickle=True)

        # Conditional ESM-C loading
        self.esmc = None
        self.esmc_conditioning = esmc_conditioning

        if esmc_conditioning:
            if esmc_path is None:
                raise ValueError(
                    "esmc_conditioning=True but esmc_path not provided!"
                )
            self.esmc = np.load(esmc_path, allow_pickle=True)
            print(f"✓ Loaded ESM-C embeddings from {esmc_path}")
        else:
            print("✓ Running in baseline mode (no ESM-C)")

    def __getitem__(self, idx):
        # Original fields (always present)
        out = {
            'lig_coords': ...,
            'lig_one_hot': ...,
            'pocket_coords': ...,
            'pocket_one_hot': ...,
            'lig_mask': ...,
            'pocket_mask': ...,
        }

        # Conditional ESM-C field
        if self.esmc_conditioning:
            receptor_name = self.data['receptors'][idx]
            out['pocket_emb'] = torch.from_numpy(
                self.esmc[receptor_name]
            ).float()  # (960,)

        return out
```

**Key points**:
- `esmc_path=None` by default → baseline mode
- Only load ESM-C if `esmc_conditioning=True`
- `pocket_emb` field only present in ESM-C mode

---

### 5.2 Dynamics Network (`equivariant_diffusion/dynamics.py`)

```python
class EGNNDynamics(nn.Module):
    def __init__(
        self,
        atom_nf,
        residue_nf,
        joint_nf=128,
        esmc_conditioning=False,    # Flag from config
        esmc_dim=960,                # ESM-C embedding dimension
        film_hidden_dim=960,         # FiLM MLP hidden size
        ...
    ):
        super().__init__()

        # Store flag
        self.esmc_conditioning = esmc_conditioning

        # Original encoders (always present)
        self.atom_encoder = nn.Sequential(
            nn.Linear(atom_nf, joint_nf),
            nn.ReLU(),
            nn.Linear(joint_nf, joint_nf)
        )

        self.residue_encoder = nn.Sequential(
            nn.Linear(residue_nf, joint_nf),
            nn.ReLU(),
            nn.Linear(joint_nf, joint_nf)
        )

        # Conditional FiLM network
        self.pocket_film = None
        if esmc_conditioning:
            self.pocket_film = nn.Sequential(
                nn.Linear(esmc_dim, film_hidden_dim),
                nn.SiLU(),
                nn.Linear(film_hidden_dim, joint_nf * 2)  # γ and β
            )
            print("✓ Initialized FiLM conditioning network")
        else:
            print("✓ Running without FiLM conditioning (baseline)")

        # EGNN (always present)
        self.egnn = EGNN(...)

        # Decoders (always present)
        self.atom_decoder = nn.Sequential(...)
        self.residue_decoder = nn.Sequential(...)

    def forward(
        self,
        xh_atoms,
        xh_residues,
        t,
        mask_atoms,
        mask_residues,
        pocket_emb=None    # Optional: None in baseline mode
    ):
        # Extract coordinates and features
        x_atoms, h_atoms = xh_atoms[:, :3], xh_atoms[:, 3:]
        x_residues, h_residues = xh_residues[:, :3], xh_residues[:, 3:]

        # Encode to joint space
        h_atoms = self.atom_encoder(h_atoms)      # (N_lig, 128)
        h_residues = self.residue_encoder(h_residues)  # (N_pocket, 128)

        # Conditional FiLM application
        if self.esmc_conditioning:
            if pocket_emb is None:
                raise ValueError(
                    "esmc_conditioning=True but pocket_emb not provided!"
                )

            # Apply FiLM
            film_params = self.pocket_film(pocket_emb)  # (960,) → (256,)
            gamma = film_params[..., :self.joint_nf]   # (..., 128)
            beta = film_params[..., self.joint_nf:]    # (..., 128)

            # Broadcast to all ligand atoms
            # Handle batching correctly
            if gamma.dim() == 1:
                gamma = gamma.unsqueeze(0)  # (1, 128)
                beta = beta.unsqueeze(0)

            h_atoms = gamma * h_atoms + beta

        # Continue with unified graph (unchanged)
        x = torch.cat((x_atoms, x_residues), dim=0)
        h = torch.cat((h_atoms, h_residues), dim=0)
        mask = torch.cat([mask_atoms, mask_residues])

        # Add timestep
        h_time = t[mask]
        h = torch.cat([h, h_time], dim=1)

        # EGNN message passing
        h_final, x_final = self.egnn(h, x, edges, ...)

        # Decode
        h_atoms_final = h_final[:len(mask_atoms)]
        h_residues_final = h_final[len(mask_atoms):]

        h_atoms_decoded = self.atom_decoder(h_atoms_final)
        h_residues_decoded = self.residue_decoder(h_residues_final)

        return ...
```

**Key points**:
- FiLM network only initialized if `esmc_conditioning=True`
- `pocket_emb=None` by default → safe for baseline calls
- Conditional FiLM application only if enabled
- Zero overhead in baseline mode

---

### 5.3 Diffusion Model (`equivariant_diffusion/en_diffusion.py`)

```python
class EnVariationalDiffusion(nn.Module):
    def forward_diffusion(self, ligand, pocket, t):
        """
        Forward diffusion step

        Args:
            ligand: dict with 'coords', 'one_hot', 'mask'
            pocket: dict with 'coords', 'one_hot', 'mask', ['emb']
            t: timesteps
        """
        # ... add noise to ligand ...

        # Call dynamics
        # Extract embedding if present
        pocket_emb = pocket.get('emb', None)  # None if not present

        eps_lig, eps_pocket = self.dynamics(
            xh_atoms=xh_lig_t,
            xh_residues=xh_pocket_t,
            t=t_normalized,
            mask_atoms=lig_mask,
            mask_residues=pocket_mask,
            pocket_emb=pocket_emb    # Pass through (None in baseline)
        )

        return eps_lig, eps_pocket
```

**Key points**:
- Use `.get('emb', None)` to safely extract embedding
- Pass `pocket_emb=None` in baseline mode
- Dynamics handles None appropriately

---

### 5.4 Lightning Module (`lightning_modules.py`)

```python
class LigandPocketDDPM(pl.LightningModule):
    def __init__(self, config, ...):
        super().__init__()
        self.config = config
        self.esmc_conditioning = config.get('esmc_conditioning', False)

        # Build model
        self.model = EnVariationalDiffusion(
            dynamics=EGNNDynamics(
                esmc_conditioning=self.esmc_conditioning,
                esmc_dim=config.get('esmc_dim', 960),
                ...
            ),
            ...
        )

    def setup(self, stage=None):
        # Determine ESM-C paths
        esmc_path = None
        if self.esmc_conditioning:
            esmc_path = self.config[f'esmc_{stage}_path']

        # Create datasets
        if stage == 'fit':
            self.train_dataset = ProcessedLigandPocketDataset(
                data_path='data/crossdocked_noH_full/train.npz',
                esmc_path=esmc_path,
                esmc_conditioning=self.esmc_conditioning,
                ...
            )

            self.val_dataset = ProcessedLigandPocketDataset(
                data_path='data/crossdocked_noH_full/val.npz',
                esmc_path=self.config.get('esmc_val_path'),
                esmc_conditioning=self.esmc_conditioning,
                ...
            )

        elif stage == 'test':
            self.test_dataset = ProcessedLigandPocketDataset(
                data_path='data/crossdocked_noH_full/test.npz',
                esmc_path=self.config.get('esmc_test_path'),
                esmc_conditioning=self.esmc_conditioning,
                ...
            )
```

---

## 6. Command-Line Usage

### Training

#### Baseline Mode
```bash
# Use original config (esmc_conditioning: False)
python train.py \
    --config configs/crossdock_fullatom_cond.yml \
    --run_name "baseline_v1" \
    --gpus 1
```

#### ESM-C Mode
```bash
# Use ESM-C config (esmc_conditioning: True)
python train.py \
    --config configs/crossdock_fullatom_cond_esmc.yml \
    --run_name "esmc_v1" \
    --gpus 1
```

#### Override Flags (optional)
```bash
# Override config file settings from command line
python train.py \
    --config configs/crossdock_fullatom_cond.yml \
    --esmc_conditioning True \
    --esmc_train_path data/esmc_train.npz \
    --esmc_val_path data/esmc_val.npz
```

---

### Inference

#### Baseline Mode
```bash
python generate_ligands.py \
    checkpoints/baseline.ckpt \
    --pdbfile example/3rfm.pdb \
    --ref_ligand A:330 \
    --n_samples 100 \
    --outfile baseline_output.sdf
```

#### ESM-C Mode
```bash
# Automatically detects ESM-C conditioning from checkpoint
python generate_ligands.py \
    checkpoints/esmc.ckpt \
    --pdbfile example/3rfm.pdb \
    --ref_ligand A:330 \
    --n_samples 100 \
    --outfile esmc_output.sdf \
    --esmc_embedding_path data/esmc_test.npz  # If needed
```

---

## 7. Validation & Testing

### Unit Tests

Create `tests/test_conditioning.py`:

```python
import torch
import pytest
from equivariant_diffusion.dynamics import EGNNDynamics

def test_baseline_mode():
    """Test that baseline mode works without ESM-C"""
    dynamics = EGNNDynamics(
        atom_nf=27,
        residue_nf=20,
        joint_nf=128,
        esmc_conditioning=False  # Baseline
    )

    # Create dummy data
    batch_size = 2
    n_atoms = 10
    n_residues = 20

    xh_atoms = torch.randn(n_atoms, 30)      # 3 + 27
    xh_residues = torch.randn(n_residues, 23)  # 3 + 20
    t = torch.rand(batch_size, 1)
    mask_atoms = torch.ones(n_atoms, dtype=torch.bool)
    mask_residues = torch.ones(n_residues, dtype=torch.bool)

    # Should work without pocket_emb
    output = dynamics(
        xh_atoms, xh_residues, t, mask_atoms, mask_residues,
        pocket_emb=None  # No embedding
    )

    assert output is not None


def test_esmc_mode():
    """Test that ESM-C mode requires embedding"""
    dynamics = EGNNDynamics(
        atom_nf=27,
        residue_nf=20,
        joint_nf=128,
        esmc_conditioning=True,  # ESM-C enabled
        esmc_dim=960
    )

    # Create dummy data
    xh_atoms = torch.randn(10, 30)
    xh_residues = torch.randn(20, 23)
    t = torch.rand(2, 1)
    mask_atoms = torch.ones(10, dtype=torch.bool)
    mask_residues = torch.ones(20, dtype=torch.bool)
    pocket_emb = torch.randn(960)  # ESM-C embedding

    # Should work with embedding
    output = dynamics(
        xh_atoms, xh_residues, t, mask_atoms, mask_residues,
        pocket_emb=pocket_emb
    )

    assert output is not None

    # Should fail without embedding
    with pytest.raises(ValueError):
        dynamics(
            xh_atoms, xh_residues, t, mask_atoms, mask_residues,
            pocket_emb=None  # Missing!
        )


def test_film_initialization():
    """Test that FiLM network only initialized when needed"""
    # Baseline: no FiLM
    dynamics_baseline = EGNNDynamics(
        atom_nf=27,
        residue_nf=20,
        esmc_conditioning=False
    )
    assert dynamics_baseline.pocket_film is None

    # ESM-C: has FiLM
    dynamics_esmc = EGNNDynamics(
        atom_nf=27,
        residue_nf=20,
        esmc_conditioning=True
    )
    assert dynamics_esmc.pocket_film is not None
```

Run tests:
```bash
pytest tests/test_conditioning.py -v
```

---

### Integration Tests

```python
# tests/test_end_to_end.py

def test_baseline_training_step():
    """Test full training step in baseline mode"""
    config = {
        'esmc_conditioning': False,
        'batch_size': 2,
        ...
    }

    model = LigandPocketDDPM(config)

    # Create batch without pocket_emb
    batch = {
        'lig_coords': ...,
        'lig_one_hot': ...,
        'pocket_coords': ...,
        'pocket_one_hot': ...,
    }

    # Should work
    loss = model.training_step(batch, 0)
    assert torch.isfinite(loss)


def test_esmc_training_step():
    """Test full training step with ESM-C"""
    config = {
        'esmc_conditioning': True,
        'esmc_dim': 960,
        'batch_size': 2,
        ...
    }

    model = LigandPocketDDPM(config)

    # Create batch with pocket_emb
    batch = {
        'lig_coords': ...,
        'lig_one_hot': ...,
        'pocket_coords': ...,
        'pocket_one_hot': ...,
        'pocket_emb': torch.randn(2, 960),  # Batch of embeddings
    }

    # Should work
    loss = model.training_step(batch, 0)
    assert torch.isfinite(loss)
```

---

## 8. Checkpoint Compatibility

### Saving Checkpoints

Both baseline and ESM-C checkpoints include config:

```python
# Automatic in PyTorch Lightning
checkpoint = {
    'state_dict': model.state_dict(),
    'hyper_parameters': {
        'esmc_conditioning': config['esmc_conditioning'],
        'esmc_dim': config.get('esmc_dim', None),
        ...
    }
}
```

### Loading Checkpoints

```python
# Automatically detect mode from checkpoint
model = LigandPocketDDPM.load_from_checkpoint(
    'checkpoints/model.ckpt'
)

print(f"Loaded model with esmc_conditioning={model.esmc_conditioning}")
```

---

## 9. Performance Considerations

### Baseline Mode
- **Zero overhead**: No ESM-C loading, no FiLM network
- **Memory**: Same as original DiffSBDD
- **Speed**: Identical to original DiffSBDD

### ESM-C Mode
- **Preprocessing**: One-time cost to extract embeddings
- **Memory**: +960 floats per sample (~4 KB)
- **Compute**: +1 FiLM forward pass per batch (~0.1ms)
- **Overall**: Negligible overhead (<1% slower)

---

## 10. Comparison Script

Create `scripts/compare_modes.py`:

```python
"""
Compare baseline vs ESM-C modes on same test set
"""

import torch
from lightning_modules import LigandPocketDDPM
from analysis.metrics import compute_all_metrics

# Load both models
baseline = LigandPocketDDPM.load_from_checkpoint(
    'checkpoints/baseline.ckpt'
)
esmc = LigandPocketDDPM.load_from_checkpoint(
    'checkpoints/esmc.ckpt'
)

# Load test set
test_dataset_baseline = ProcessedLigandPocketDataset(
    'data/crossdocked_noH_full/test.npz',
    esmc_conditioning=False
)

test_dataset_esmc = ProcessedLigandPocketDataset(
    'data/crossdocked_noH_full/test.npz',
    esmc_path='data/esmc_test.npz',
    esmc_conditioning=True
)

# Compare on same pockets
results = {'baseline': [], 'esmc': []}

for i in range(len(test_dataset_baseline)):
    sample_baseline = test_dataset_baseline[i]
    sample_esmc = test_dataset_esmc[i]

    # Generate molecules
    mols_baseline = baseline.generate(sample_baseline, n_samples=100)
    mols_esmc = esmc.generate(sample_esmc, n_samples=100)

    # Compute metrics
    metrics_baseline = compute_all_metrics(mols_baseline)
    metrics_esmc = compute_all_metrics(mols_esmc)

    results['baseline'].append(metrics_baseline)
    results['esmc'].append(metrics_esmc)

# Statistical comparison
print_comparison_table(results)
```

---

## 11. Documentation Checklist

### For Each Code File Modified:

- [ ] Add docstring explaining ESM-C parameter
- [ ] Add inline comments for conditional logic
- [ ] Update type hints to include Optional[...]
- [ ] Add assertions for required parameters

### Example:

```python
def forward(
    self,
    xh_atoms: torch.Tensor,
    xh_residues: torch.Tensor,
    t: torch.Tensor,
    mask_atoms: torch.Tensor,
    mask_residues: torch.Tensor,
    pocket_emb: Optional[torch.Tensor] = None  # NEW
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Forward pass through dynamics network.

    Args:
        xh_atoms: Ligand atom features, shape (N_lig, 3+nf)
        xh_residues: Pocket residue features, shape (N_pocket, 3+nf)
        t: Timesteps, shape (batch_size, 1)
        mask_atoms: Ligand atom mask, shape (N_lig,)
        mask_residues: Pocket residue mask, shape (N_pocket,)
        pocket_emb: ESM-C pocket embedding, shape (960,).
                    Required if esmc_conditioning=True,
                    ignored if esmc_conditioning=False.

    Returns:
        eps_atoms: Predicted ligand noise
        eps_residues: Predicted pocket noise
    """
```

---

## 12. Summary

### Key Principles

✅ **Single flag**: `esmc_conditioning` controls everything
✅ **Backward compatible**: Baseline mode unchanged
✅ **Zero overhead**: No performance cost in baseline mode
✅ **Safe defaults**: `pocket_emb=None`, `esmc_path=None`
✅ **Clear errors**: Fail fast with helpful messages
✅ **Easy comparison**: Same codebase, different configs

### Files Modified

```
Core changes:
├── configs/crossdock_fullatom_cond_esmc.yml    [NEW]
├── dataset.py                                   [MODIFIED]
├── equivariant_diffusion/dynamics.py           [MODIFIED]
├── equivariant_diffusion/en_diffusion.py       [MODIFIED]
├── equivariant_diffusion/conditional_model.py  [MODIFIED]
└── lightning_modules.py                         [MODIFIED]

Testing:
├── tests/test_conditioning.py                   [NEW]
└── tests/test_end_to_end.py                     [NEW]

Scripts:
└── scripts/compare_modes.py                     [NEW]
```

### Usage Summary

```bash
# Train baseline
python train.py --config configs/crossdock_fullatom_cond.yml

# Train ESM-C
python train.py --config configs/crossdock_fullatom_cond_esmc.yml

# Generate baseline
python generate_ligands.py checkpoints/baseline.ckpt ...

# Generate ESM-C
python generate_ligands.py checkpoints/esmc.ckpt ...

# Compare
python scripts/compare_modes.py
```

---

**This design ensures the codebase is maintainable, extensible, and easy to use for fair comparisons between baseline and ESM-C conditioning!** 🎯
