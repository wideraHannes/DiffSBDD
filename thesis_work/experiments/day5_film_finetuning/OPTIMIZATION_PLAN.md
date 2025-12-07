# FiLM Fine-Tuning Optimization Plan

## Executive Summary

Initial FiLM-only training revealed severe overfitting and an architectural bottleneck. The current FiLM network (960D → 64D) is too small to effectively utilize rich ESM-C embeddings.

## Current Architecture Analysis

### FiLM Network Structure

```python
# Current (dynamics.py:72-77)
nn.Sequential(
    nn.Linear(960, 256),      # 246,016 params
    act_fn,
    nn.Linear(256, 128),      # 32,896 params
    act_fn,
    nn.Linear(128, 64),       # 8,256 params
)
# Total: ~287K params (or ~131K as reported - needs verification)
```

### Information Bottleneck

- **Input**: 960D ESM-C embeddings (pre-trained on billions of proteins)
- **Output**: 64D conditioning (2 × 32 joint_nf for γ and β)
- **Compression ratio**: 15:1 - far too aggressive
- **Result**: Cannot effectively utilize ESM-C information

### Training Issues Observed

| Issue          | Evidence                              | Severity     |
| -------------- | ------------------------------------- | ------------ |
| Overfitting    | Train: 0.53→0.19, Val: 2184→5427      | **Critical** |
| Instability    | Gradient clip 2.2-4.1 (limit 1.4-2.3) | **High**     |
| Val volatility | Val loss ranges -28 to 5427           | **High**     |
| Small capacity | 131K params for 960→64 mapping        | **High**     |

**Positive signs:**

- Molecule quality: 100% validity, 80-100% connectivity
- Training loss decreases steadily
- No NaN issues

## Proposed Solutions

### Solution 1: Larger FiLM Architecture (PRIMARY)

#### Option A: Deeper Gradual Compression ⭐ **RECOMMENDED**

```python
nn.Sequential(
    nn.Linear(960, 512),      # 492,032 params
    nn.Dropout(0.15),
    nn.SiLU(),
    nn.Linear(512, 256),      # 131,328 params
    nn.Dropout(0.15),
    nn.SiLU(),
    nn.Linear(256, 64),       # 16,448 params
)
# Total: ~640K params (2.2-4.9x increase)
```

**Rationale:**

- More gradual compression: 960 → 512 → 256 → 64 (2x steps)
- Each layer has sufficient capacity to learn meaningful transformations
- Dropout prevents overfitting on small training set
- ~640K params is reasonable for adapter (vs ~1M frozen EGNN)

#### Option B: Wider Network

```python
nn.Sequential(
    nn.Linear(960, 768),      # 737,536 params
    nn.Dropout(0.15),
    nn.SiLU(),
    nn.Linear(768, 384),      # 295,296 params
    nn.Dropout(0.15),
    nn.SiLU(),
    nn.Linear(384, 64),       # 24,640 params
)
# Total: ~1.06M params (3.7-8.1x increase)
```

**Rationale:**

- Keep more ESM-C dimensions longer in pipeline
- Larger capacity for complex protein → ligand mappings
- Risk: Might overfit more, needs stronger regularization

#### Option C: Residual/Skip Connections

```python
class FiLMNetworkResidual(nn.Module):
    def __init__(self):
        self.main_path = nn.Sequential(
            nn.Linear(960, 512),
            nn.Dropout(0.15),
            nn.SiLU(),
            nn.Linear(512, 256),
            nn.Dropout(0.15),
            nn.SiLU(),
        )
        self.skip_projection = nn.Linear(960, 256)
        self.output = nn.Linear(256, 64)

    def forward(self, x):
        main = self.main_path(x)
        skip = self.skip_projection(x)
        combined = main + skip
        return self.output(combined)
# Total: ~640K + 245K = ~885K params
```

**Rationale:**

- Preserve information flow from ESM-C embeddings
- Better gradient flow during training
- More complex to implement

### Solution 2: Regularization Strategy

```python
# Add to dynamics.py __init__
self.film_network = nn.Sequential(
    nn.Linear(960, 512),
    nn.Dropout(p=0.15),        # Dropout rate 1
    nn.SiLU(),
    nn.Linear(512, 256),
    nn.Dropout(p=0.15),        # Dropout rate 2
    nn.SiLU(),
    nn.Linear(256, 2 * joint_nf),
)
```

**Dropout rates to try:**

- Conservative: 0.1
- Moderate: 0.15 (recommended)
- Aggressive: 0.2

**Weight decay (in lightning_modules.py):**

```python
torch.optim.AdamW(
    self.ddpm.dynamics.film_network.parameters(),
    lr=self.lr,
    weight_decay=1e-4,  # Increase from 1e-6
    amsgrad=True,
)
```

### Solution 3: Training Hyperparameters

#### Learning Rate & Schedule

```yaml
# film_finetuning.yml
lr: 5.0e-4 # Reduce from 1e-3 (current too high for fine-tuning)

# Add scheduler (requires code changes)
scheduler:
  type: "cosine"
  T_max: 50 # n_epochs
  eta_min: 1.0e-5
```

#### Gradient Clipping

```yaml
# Increase threshold slightly
clip_grad: true
max_grad_norm: 3.0 # From ~2.0, allow more gradient flow
```

#### Early Stopping

```python
# Add to lightning_modules.py or train.py
from pytorch_lightning.callbacks import EarlyStopping

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,          # Stop if no improvement for 10 epochs
    mode='min',
    verbose=True,
    min_delta=1.0,       # Minimum change to qualify as improvement
)
```

### Solution 4: Validation Improvements

**Current issues:**

- Val set seems very small (1 batch in logs)
- Val loss is extremely noisy
- Hard to detect overfitting early

**Improvements:**

```yaml
# film_finetuning.yml
eval_epochs: 2 # From 10 - more frequent validation
batch_size: 16 # Keep same

# Ensure validation set is reasonable size
# Check data split in dataset.py
```

**Logging enhancements:**

```python
# In lightning_modules.py training_step / validation_step
self.log('train_loss', loss, on_step=True, on_epoch=True)
self.log('val_loss', val_loss, on_step=False, on_epoch=True)
self.log('grad_norm', grad_norm)  # Track gradient magnitude
```

## Implementation Roadmap

### Phase 1: Quick Wins (1-2 hours)

1. ✅ Document current findings (this file)
2. [ ] Implement larger FiLM network (Option A: 960→512→256→64)
3. [ ] Add dropout (0.15) after each hidden layer
4. [ ] Reduce learning rate: 1e-3 → 5e-4
5. [ ] Increase weight decay: 1e-6 → 1e-4
6. [ ] Run training for 50 epochs

### Phase 2: Monitoring & Tuning (ongoing)

7. [ ] Add early stopping callback
8. [ ] Increase eval frequency to every 2 epochs
9. [ ] Log gradient norms to track stability
10. [ ] Monitor validation loss curve
11. [ ] Generate molecules at checkpoints to verify quality

### Phase 3: Advanced Optimizations (if needed)

12. [ ] Try Option B (wider network) if Option A still overfits
13. [ ] Implement residual connections (Option C) if stability issues persist
14. [ ] Add learning rate scheduler (cosine annealing)
15. [ ] Experiment with label smoothing or other regularization

## Expected Outcomes

### After Phase 1

- **Training stability**: Gradient clipping should trigger less frequently
- **Validation loss**: Should be less volatile, track train loss more closely
- **Overfitting**: Should reduce but not eliminate
- **Model size**: ~640K trainable params (reasonable for adapter)

### Success Metrics

| Metric               | Current       | Target                |
| -------------------- | ------------- | --------------------- |
| Val loss stability   | ±2000         | ±100                  |
| Train/val gap        | 0.19 vs 2000+ | <5x difference        |
| Gradient clips/epoch | 2-3           | <1                    |
| Molecule validity    | 100%          | >95%                  |
| Binding improvement  | N/A           | -0.5 to -1.5 kcal/mol |

## Alternative Approaches (if FiLM still fails)

### 1. Cross-Attention Instead of FiLM

Replace FiLM with cross-attention between ligand features and ESM-C embeddings.

- More parameters but more expressive
- Better for learning complex protein-ligand interactions

### 2. Fine-tune Last EGNN Layer Too

Unfreeze last 1-2 EGNN layers along with FiLM.

- More capacity to adapt to ESM-C conditioning
- Risk: More overfitting, need even more data

### 3. Contrastive Pre-training

Pre-train FiLM network on larger dataset with contrastive objective.

- Better initialization before fine-tuning
- Requires additional data preparation

## References

- Current implementation: `equivariant_diffusion/dynamics.py:72-77`
- Training logs: `thesis_work/experiments/day5_film_finetuning/outputs/wandb/`
- Config: `thesis_work/experiments/day5_film_finetuning/configs/film_finetuning.yml`

## Notes

- **Why not trim ESM-C embeddings?** They're pre-trained on massive data, contain valuable information. Compute is in inference (already done), not vector size.
- **Why focus on FiLM size?** It's the only component learning to map ESM-C → diffusion conditioning. If it's too small, we can't leverage ESM-C effectively.
- **Parameter count discrepancy**: Code calculates ~287K but logs show 131K. Needs investigation, but doesn't change core recommendation to increase size.
