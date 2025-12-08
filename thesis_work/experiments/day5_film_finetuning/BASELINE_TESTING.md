# FiLM Baseline Testing Guide

This document explains how to run the three baseline experiments for validating the FiLM fine-tuning approach.

## Overview

According to the tuning plan (`.claude/tuningplan.md` Part 2), we need three baselines:

1. **Baseline 1**: Pretrained checkpoint only (no FiLM) - Ground truth
2. **Baseline 2**: Identity FiLM (γ=1, β=0) - No-op verification
3. **Baseline 3**: Random FiLM - Negative control

## Quick Start

### Using Configuration Files

Three config files are provided in `configs/`:

```bash
# Baseline 1: Pretrained without FiLM
uv run python train.py --config thesis_work/experiments/day5_film_finetuning/configs/baseline1_pretrained_no_film.yml

# Baseline 2: Identity FiLM
uv run python train.py --config thesis_work/experiments/day5_film_finetuning/configs/baseline2_identity_film.yml

# Baseline 3: Random FiLM
uv run python train.py --config thesis_work/experiments/day5_film_finetuning/configs/baseline3_random_film.yml
```

### Using Python API

Alternatively, load models programmatically:

```python
from lightning_modules import LigandPocketDDPM

checkpoint_path = "checkpoints/crossdocked_fullatom_cond.ckpt"

# Baseline 1: No FiLM
model1 = LigandPocketDDPM.load_pretrained_with_esmc(
    checkpoint_path,
    use_film=False
)

# Baseline 2: Identity FiLM
model2 = LigandPocketDDPM.load_pretrained_with_esmc(
    checkpoint_path,
    use_film=True,
    film_mode="identity"
)

# Baseline 3: Random FiLM
model3 = LigandPocketDDPM.load_pretrained_with_esmc(
    checkpoint_path,
    use_film=True,
    film_mode="random"
)
```

See `baseline_example.py` for a complete example.

## Expected Results

### Baseline 1: Pretrained (Ground Truth)

**Purpose**: Establish what "good" looks like

| Metric | Expected |
|--------|----------|
| Loss | 0.5-0.6 |
| Connectivity | >95% |
| Validity | >95% |
| QED | 0.4-0.6 |

**Success Criterion**: If connectivity < 90%, STOP - pretrained checkpoint or data is broken.

### Baseline 2: Identity FiLM (No-Op)

**Purpose**: Verify FiLM initialization works correctly

| Metric | Expected |
|--------|----------|
| Loss | ≈ Baseline 1 (±0.05) |
| Connectivity | ≈ Baseline 1 (±5%) |
| Validity | ≈ Baseline 1 |
| QED | ≈ Baseline 1 |

**Success Criterion**: Results should be nearly identical to Baseline 1. If significantly different, identity initialization is broken.

**Verification**:
```python
# Check gamma=1, beta=0
film = model.ddpm.dynamics.film_network
final_layer = film[-1]
joint_nf = final_layer.out_features // 2

gamma = final_layer.bias.data[:joint_nf]
beta = final_layer.bias.data[joint_nf:]

assert torch.allclose(gamma, torch.ones_like(gamma), atol=1e-6)
assert torch.allclose(beta, torch.zeros_like(beta), atol=1e-6)
```

### Baseline 3: Random FiLM (Negative Control)

**Purpose**: Verify FiLM actually affects the model

| Metric | Expected |
|--------|----------|
| Loss | 1.0-2.0+ (much worse) |
| Connectivity | 0-30% (poor) |
| Validity | 50-80% (degraded) |
| QED | Lower than baseline |

**Success Criterion**: Random should be MUCH WORSE than baselines 1&2.

If random ≈ identity → **BUG**: FiLM is not being used in the forward pass!

## Interpreting Results

### ✅ Success Pattern

```
Baseline 1: Connectivity 95% | Loss 0.55
Baseline 2: Connectivity 94% | Loss 0.56  (≈ Baseline 1)
Baseline 3: Connectivity 15% | Loss 1.80  (<< Baseline 1)
```

**Interpretation**:
- Identity works correctly (Baseline 2 ≈ Baseline 1)
- FiLM is active (Baseline 3 << Baseline 1)
- ✅ Ready to proceed with FiLM training

### ❌ Failure Pattern 1: Identity Broken

```
Baseline 1: Connectivity 95% | Loss 0.55
Baseline 2: Connectivity 60% | Loss 0.85  (worse than Baseline 1)
Baseline 3: Connectivity 15% | Loss 1.80
```

**Interpretation**: Identity FiLM doesn't reproduce baseline
**Action**: Check `_init_film_identity()` implementation in lightning_modules.py:286

### ❌ Failure Pattern 2: FiLM Not Used

```
Baseline 1: Connectivity 95% | Loss 0.55
Baseline 2: Connectivity 94% | Loss 0.56  (≈ Baseline 1, good)
Baseline 3: Connectivity 92% | Loss 0.58  (≈ Baseline 1, BAD!)
```

**Interpretation**: Random FiLM performs as well as identity → FiLM not being applied
**Action**: Check dynamics.py:150 - verify `if self.use_film and pocket_emb is not None:`

### ❌ Failure Pattern 3: Broken Pretrained Checkpoint

```
Baseline 1: Connectivity 20% | Loss 0.55
Baseline 2: Connectivity 18% | Loss 0.57
Baseline 3: Connectivity 10% | Loss 1.80
```

**Interpretation**: Even baseline has poor connectivity
**Action**:
1. Check checkpoint file integrity (md5sum)
2. Verify data preprocessing (10-feature encoding)
3. Check RDKit version compatibility

## Config File Parameters

Key parameters in baseline config files:

```yaml
# Baseline control
film_only_training: false    # Not training, just evaluating
use_film: true/false         # Enable/disable FiLM
film_mode: "identity"/"random"  # Initialization method

# Evaluation
n_epochs: 1                  # Just 1 epoch for evaluation
eval_epochs: 1               # Evaluate immediately
eval_params:
  n_eval_samples: 50         # More samples for reliable metrics
```

## Running All Baselines Sequentially

```bash
#!/bin/bash
# run_all_baselines.sh

configs=(
    "baseline1_pretrained_no_film.yml"
    "baseline2_identity_film.yml"
    "baseline3_random_film.yml"
)

for config in "${configs[@]}"; do
    echo "Running $config..."
    uv run python train.py --config "thesis_work/experiments/day5_film_finetuning/configs/$config"
done

echo "All baselines completed!"
```

## Analyzing Results

After running all baselines, compare results:

```python
import wandb

# Get runs from WandB
api = wandb.Api()
runs = api.runs("your-entity/ligand-pocket-ddpm", filters={"group": "film-baselines"})

for run in runs:
    metrics = run.summary
    print(f"{run.name}:")
    print(f"  Connectivity: {metrics.get('Connectivity/val', 'N/A')}")
    print(f"  Loss: {metrics.get('loss/val', 'N/A')}")
    print(f"  QED: {metrics.get('QED/val', 'N/A')}")
```

## Next Steps

After successful baseline validation:

1. **If all baselines pass** → Proceed with FiLM training using `film_finetuning.yml`
2. **If Baseline 1 fails** → Fix data/checkpoint issues before proceeding
3. **If Baseline 2 fails** → Fix identity initialization
4. **If Baseline 3 fails** → Fix FiLM forward pass logic

## Troubleshooting

### "Missing keys (expected FiLM)" warning

This is **expected** when loading pretrained checkpoints. The warning indicates FiLM weights are missing from the pretrained checkpoint and will be initialized.

### ESM-C embeddings not found

For Baseline 1, ESM-C embeddings are not needed (`esmc_path: null`).
For Baselines 2&3, ensure embeddings exist:
```bash
ls data/dummy_dataset/*_esmc.npz
# Should show: train_esmc.npz, val_esmc.npz, test_esmc.npz
```

### CUDA out of memory

Reduce batch size in config:
```yaml
batch_size: 1  # Reduce from 2
eval_params:
  eval_batch_size: 2  # Reduce from 5
```

## References

- Tuning Plan: `.claude/tuningplan.md` Part 2
- Implementation: `lightning_modules.py:232` (load_pretrained_with_esmc)
- FiLM Network: `dynamics.py:73` (film_network definition)
- Example Usage: `baseline_example.py`
