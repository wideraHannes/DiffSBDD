# Code Reference: FiLM Fine-Tuning

> Quick reference for implementing FiLM-only training on pretrained DiffSBDD.

---

## 1. Key Files to Modify

| File | What to Add | Lines |
|------|-------------|-------|
| `lightning_modules.py` | `load_pretrained_with_esmc()`, `_init_film_identity()` | ~200 |
| `lightning_modules.py` | `film_only_training` flag in `__init__` | ~60 |
| `lightning_modules.py` | FiLM-only optimizer in `configure_optimizers()` | ~400 |
| `lightning_modules.py` | `pocket_emb` param in `generate_ligands()` | ~909 |
| `conditional_model.py` | Thread `pocket_emb` through `sample_given_pocket()` | ~481 |
| `generate_ligands.py` | Add `--esmc_emb` argument | ~28 |

**Already done:**
- `dynamics.py:55-61` — FiLM network defined
- `dynamics.py:119-131` — Forward pass handles `pocket_emb`

---

## 2. Architecture Overview

### Current FiLM Network (dynamics.py:55-61)

```python
self.film_network = nn.Sequential(
    nn.Linear(960, hidden_nf),      # 960 → 128
    act_fn,                          # SiLU
    nn.Linear(hidden_nf, 2 * joint_nf)  # 128 → 64 (32 gamma + 32 beta)
)
```

**Parameter count:** ~131K (960×128 + 128×64 + biases)

### FiLM Forward Pass (dynamics.py:119-131)

```python
if pocket_emb is not None:
    film_params = self.film_network(pocket_emb)  # [batch, 64]
    gamma, beta = torch.chunk(film_params, 2, dim=-1)  # each [batch, 32]
    gamma_expanded = gamma[mask.long()]  # [num_nodes, 32]
    beta_expanded = beta[mask.long()]
    h = gamma_expanded * h + beta_expanded  # FiLM modulation!
```

---

## 3. Implementation Snippets

### Checkpoint Loading with Identity FiLM

```python
# Add to lightning_modules.py (after line ~200)

@classmethod
def load_pretrained_with_esmc(cls, checkpoint_path, device='cpu'):
    """Load pretrained checkpoint, init FiLM to identity."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    hparams = ckpt['hyper_parameters']
    model = cls(**hparams)

    # Load with strict=False (FiLM weights missing from checkpoint)
    missing, unexpected = model.load_state_dict(ckpt['state_dict'], strict=False)
    print(f"Missing keys (expected): {missing}")

    # Init FiLM to identity: h' = 1*h + 0 = h
    model._init_film_identity()
    return model.to(device)

def _init_film_identity(self):
    """Initialize FiLM for identity transformation (gamma=1, beta=0)."""
    film = self.ddpm.dynamics.film_network
    with torch.no_grad():
        for m in film.modules():
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Set gamma=1 in final layer bias
        final = film[-1]
        joint_nf = final.out_features // 2
        final.bias.data[:joint_nf] = 1.0   # gamma = 1
        final.bias.data[joint_nf:] = 0.0   # beta = 0
```

### FiLM-Only Training Flag

```python
# Add to __init__ signature in lightning_modules.py

def __init__(
    self,
    ...,
    film_only_training=False,  # NEW
    ...
):
    ...
    self.film_only_training = film_only_training
    self.save_hyperparameters()
```

### FiLM-Only Optimizer

```python
# Modify configure_optimizers() in lightning_modules.py

def configure_optimizers(self):
    if self.film_only_training:
        # Freeze all parameters
        for p in self.ddpm.parameters():
            p.requires_grad = False

        # Unfreeze only FiLM
        for p in self.ddpm.dynamics.film_network.parameters():
            p.requires_grad = True

        # Count trainable
        trainable = sum(p.numel() for p in self.ddpm.parameters() if p.requires_grad)
        print(f"FiLM-only training: {trainable:,} trainable parameters")

        return torch.optim.AdamW(
            self.ddpm.dynamics.film_network.parameters(),
            lr=self.lr,
            amsgrad=True,
            weight_decay=1e-6
        )

    # Original optimizer (full training)
    return torch.optim.AdamW(
        self.ddpm.parameters(),
        lr=self.lr,
        amsgrad=True,
        weight_decay=1e-12
    )
```

### Thread pocket_emb Through Inference

```python
# Modify generate_ligands() in lightning_modules.py (line ~909)

def generate_ligands(
    self,
    pdb_file,
    n_samples,
    ...,
    pocket_emb=None,  # NEW
    **kwargs,
):
    ...
    pocket = self.prepare_pocket(residues, repeats=n_samples)

    # ADD: include pocket embedding if provided
    if pocket_emb is not None:
        pocket['pocket_emb'] = pocket_emb.unsqueeze(0).expand(n_samples, -1)
    ...
```

```python
# Modify sample_given_pocket() in conditional_model.py (line ~481)

def sample_given_pocket(self, pocket, num_nodes_lig, ...):
    pocket_emb = pocket.get('pocket_emb', None)  # ADD
    ...
    # Pass pocket_emb to sampling methods
    z_lig, xh_pocket = self.sample_p_zs_given_zt(
        ..., pocket_emb=pocket_emb  # ADD
    )
```

### Command Line Argument

```python
# Add to generate_ligands.py (after line ~28)

parser.add_argument('--esmc_emb', type=Path, default=None,
                    help='Path to ESM-C embedding (.pt or .npy)')

# In main:
pocket_emb = None
if args.esmc_emb:
    if args.esmc_emb.suffix == '.pt':
        pocket_emb = torch.load(args.esmc_emb)
    else:
        pocket_emb = torch.from_numpy(np.load(args.esmc_emb)['embedding'])

# Pass to generate_ligands
molecules_batch = model.generate_ligands(..., pocket_emb=pocket_emb)
```

---

## 4. Training Config

```yaml
# configs/film_finetuning.yml
run_name: "film-finetuning-v1"
resume: "checkpoints/crossdocked_fullatom_cond.ckpt"
film_only_training: true

# Higher LR ok for small adapter
lr: 1.0e-3
n_epochs: 50
batch_size: 16

# Must match pretrained checkpoint
egnn_params:
  joint_nf: 32
  hidden_nf: 128
  n_layers: 5
```

---

## 5. Commands

### Verify Pretrained Works

```bash
uv run python generate_ligands.py checkpoints/crossdocked_fullatom_cond.ckpt \
    --pdbfile example/3rfm.pdb --outfile test.sdf --ref_ligand A:330 --n_samples 5
```

### FiLM-Only Training

```bash
uv run python train.py --config thesis_work/experiments/day5_film_finetuning/configs/film_finetuning.yml
```

### Generate with ESM-C

```bash
uv run python generate_ligands.py checkpoints/film_finetuned.ckpt \
    --pdbfile example/3rfm.pdb --outfile esmc_gen.sdf \
    --ref_ligand A:330 --n_samples 20 --esmc_emb data/esmc/3rfm.pt
```

---

## 6. Debugging

### Verify Identity Init

```python
# After loading, FiLM should output gamma=1, beta=0
pocket_emb = torch.randn(1, 960)
film_out = model.ddpm.dynamics.film_network(pocket_emb)
gamma, beta = film_out.chunk(2, dim=-1)
print(f"gamma mean: {gamma.mean():.4f} (should be ~1.0)")
print(f"beta mean: {beta.mean():.4f} (should be ~0.0)")
```

### Verify Only FiLM is Training

```python
for name, p in model.named_parameters():
    if p.requires_grad:
        print(f"Trainable: {name}")
# Should only show film_network parameters
```

### Check Gradient Flow

```python
# After one backward pass
for name, p in model.ddpm.dynamics.film_network.named_parameters():
    if p.grad is not None:
        print(f"{name}: grad_norm={p.grad.norm():.4f}")
```

---

## 7. File Structure

```
DiffSBDD/
├── train.py                    # Training entry
├── generate_ligands.py         # Inference entry (+esmc_emb arg)
├── lightning_modules.py        # +load_pretrained_with_esmc, +film_only
├── equivariant_diffusion/
│   ├── dynamics.py             # FiLM network (exists)
│   └── conditional_model.py    # +pocket_emb threading
├── checkpoints/
│   └── crossdocked_fullatom_cond.ckpt  # Pretrained baseline
└── thesis_work/experiments/day5_film_finetuning/
    └── configs/film_finetuning.yml
```
