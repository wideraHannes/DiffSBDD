# Day 5: FiLM-Only Fine-Tuning

## Goal

Train only the FiLM adapter (~131K params) while keeping pretrained EGNN frozen (~2M params).

## Approach

```
1. Load pretrained checkpoint (strict=False)
2. Initialize FiLM to identity (γ=1, β=0)
3. Freeze all params except FiLM
4. Train on CrossDock with ESM-C embeddings
5. Evaluate binding affinity improvement
```

## Directory Structure

```
day5_film_finetuning/
├── README.md           # This file
├── configs/            # Training configs
│   └── film_finetuning.yml
├── outputs/            # Checkpoints & logs
└── scripts/            # Helper scripts
```

## Quick Commands

```bash
# Verify baseline works
uv run python generate_ligands.py checkpoints/crossdocked_fullatom_cond.ckpt \
    --pdbfile example/3rfm.pdb --outfile outputs/baseline_test.sdf \
    --ref_ligand A:330 --n_samples 5 --timesteps 100

# Train FiLM-only (after implementing changes)
uv run python train.py --config thesis_work/experiments/day5_film_finetuning/configs/film_finetuning.yml

# Generate with ESM-C (after implementing changes)
uv run python generate_ligands.py outputs/film_finetuned.ckpt \
    --pdbfile example/3rfm.pdb --outfile outputs/esmc_test.sdf \
    --ref_ligand A:330 --n_samples 20 --esmc_emb data/esmc_embeddings/3rfm.pt
```

## Implementation Checklist

- [ ] Verify pretrained checkpoint generates valid molecules
- [ ] Add `load_pretrained_with_esmc()` to lightning_modules.py
- [ ] Add `_init_film_identity()` method
- [ ] Modify `configure_optimizers()` for FiLM-only training
- [ ] Thread `pocket_emb` through inference pipeline
- [ ] Run first FiLM-only training (small subset)
- [ ] Compare baseline vs FiLM binding affinity

## Expected Results

| Metric           | Baseline | FiLM + ESM-C   |
| ---------------- | -------- | -------------- |
| SMINA (kcal/mol) | -X.XX    | -X.XX (better) |
| Validity         | >90%     | >90%           |
| Connectivity     | >80%     | >80%           |

## Notes

- FiLM network already exists in `dynamics.py:55-61`
- Forward pass already handles `pocket_emb` in `dynamics.py:119-131`
- Main work: checkpoint loading, optimizer config, inference pipeline

## Training Results (Initial Run)

### Architecture Issues Identified

**Current FiLM Network:**

- Input: 960D (ESM-C embeddings)
- Hidden: 256 → 128
- Output: 64D (2 × joint_nf=32 for scale/shift)
- **Total params: ~131K-287K** (discrepancy in counting)

**Problem: Severe Information Bottleneck**

- Compressing 960D → 64D is a **15x compression**
- ESM-C embeddings contain rich pre-trained protein knowledge
- Current FiLM network too small to effectively utilize this information

### Observed Training Behavior

```
Epoch 0-10:   Train loss: 0.53 → 0.19 (good)
              Val loss: 2184 → 286 (very unstable)

Epoch 11-35:  Train loss: 0.12 → 0.13 (plateaued)
              Val loss: -28 to 5427 (extremely volatile)

Gradient clipping: Frequently triggered (2.2-4.1 vs limit 1.4-2.3)
```

**Diagnosis:**

1. **Severe overfitting**: Train loss improves but val loss explodes
2. **FiLM too small**: Can't effectively map 960D → 64D conditioning
3. **Training instability**: Frequent gradient clipping
4. **Good molecule quality**: 100% validity, 80-100% connectivity at eval

### Recommendations for Next Iteration

#### 1. Increase FiLM Capacity (PRIMARY FIX)

**Option A: Deeper, gradual compression (recommended)**

```python
960 → 512 → 256 → 64
# Parameters: ~640K (2-5x current)
```

**Option B: Wider network**

```python
960 → 768 → 384 → 64
# Parameters: ~1.1M (4-8x current)
```

**Option C: With residual connection**

```python
960 → 512 → 256 → 64 (+ skip from 960)
# Parameters: ~640K + projection
```

**Why not trim ESM-C embeddings:**

- ESM-C is pre-trained on massive protein data
- Trimming (PCA) would lose valuable information
- Compute cost is in inference (already done), not 960D vectors

#### 2. Add Regularization for Overfitting

```python
film_network = nn.Sequential(
    nn.Linear(960, 512),
    nn.Dropout(0.15),  # Add dropout
    act_fn,
    nn.Linear(512, 256),
    nn.Dropout(0.15),
    act_fn,
    nn.Linear(256, 64),
)
```

**Additional regularization:**

- Increase weight decay: 1e-6 → 1e-5 or 1e-4
- Add early stopping callback on validation loss
- Consider gradient clipping at higher threshold (3.0)

#### 3. Adjust Training Hyperparameters

```yaml
lr: 5.0e-4 # Reduce from 1e-3
scheduler: "cosine" # Add learning rate scheduling
weight_decay: 1.0e-4 # Increase from 1e-6
max_grad_norm: 3.0 # Increase clip threshold
```

#### 4. Improve Validation Strategy

- Current val set may be too small (1 batch)
- Add more frequent validation (every 5 epochs → every 2 epochs)
- Log validation metrics to detect overfitting earlier
- Consider validation set size increase

### Implementation Priority

1. **High**: Increase FiLM to 960 → 512 → 256 → 64 with dropout
2. **High**: Add early stopping on validation loss
3. **Medium**: Reduce learning rate to 5e-4 with cosine schedule
4. **Medium**: Increase weight decay to 1e-4
5. **Low**: Experiment with residual connections if still unstable
