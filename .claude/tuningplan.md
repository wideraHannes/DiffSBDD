Systematic Experimental Framework for FiLM Fine-Tuning Validation

Following Google Research Deep Learning Tuning Playbook

Executive Summary

Current Problem:

- Training loss: 0.605 → 0.197 (appears to converge)
- Validation loss: 122-10991 (extremely volatile)
- Connectivity: 0% at epoch 9 (CRITICAL ISSUE)
- No systematic baselines established
- Unclear if loss of 0.6 is "good" or "bad"

Root Cause Analysis:

1.  Loss interpretation confusion: 0.6 is NORMAL for diffusion models (expected: 0.4-0.6)
2.  Severe overfitting: Train/val gap ~50x, FiLM bottleneck (960→128→64 too small)
3.  No baseline validation: Haven't verified pretrained checkpoint works correctly
4.  Volatile validation metrics: 1-sample validation set provides unreliable estimates

Solution Approach:
This plan establishes a systematic experimental framework with:

- Scientific baselines (pretrained, identity FiLM, random FiLM)
- Clear metrics and interpretation guidelines
- Validation scripts for reproducibility
- Step-by-step debugging playbook

---

Part 1: Understanding Diffusion Loss (Answer: Is 0.6 Good?)

Loss Formula

# From conditional_model.py:412-440

loss = loss_t + loss_0 + kl_prior

where:
loss_t = ||ε_predicted - ε_true||² / (3\*n_atoms + n_features) # Denoising
loss_0 = reconstruction error at t=0 # Generation quality
kl_prior = KL(q(z_T|x) || N(0,1)) # Should be ~0

Interpretation Guide

| Loss Value | Interpretation                  | Action                                     |
| ---------- | ------------------------------- | ------------------------------------------ |
| < 0.2      | Suspicious (likely overfitting) | Check for data leakage, too small dataset  |
| 0.2-0.4    | Very good                       | Continue training, validate on generations |
| 0.4-0.6    | NORMAL (expected)               | Focus on connectivity, QED, not loss       |
| 0.6-1.0    | Acceptable                      | Monitor, ensure improving                  |
| > 1.0      | Problem                         | Check LR, gradients, data loading          |

Key Insight: Diffusion models don't need low loss to generate valid molecules. A loss of 0.6 means ~60% MSE predicting noise
at each step, which is sufficient for high-quality generation.

Your Status: Loss 0.6 is FINE. The problem is 0% connectivity, not loss value.

---

Part 2: Three Scientific Baselines (Start Simple)

Baseline 1: Pretrained Checkpoint Only

Purpose: Establish ground truth for what "good" performance looks like

Method:

# Load pretrained WITHOUT FiLM modifications

model = LigandPocketDDPM.load_from_checkpoint(
"checkpoints/crossdocked_fullatom_cond.ckpt"
)

# Evaluate on dummy_dataset/val

Expected Results:

- Loss: 0.5-0.6
- Connectivity: >95%
- Validity: >95%
- QED: 0.4-0.6

Success Criterion: If connectivity < 90%, STOP - pretrained checkpoint is broken or data preprocessing is wrong.

Files:

- Script: thesis_work/experiments/day5_film_finetuning/scripts/validate_baseline.py (NEW)
- Run: uv run python scripts/validate_baseline.py --checkpoint checkpoints/crossdocked_fullatom_cond.ckpt

---

Baseline 2: Identity-Initialized FiLM (No-Op Verification)

Purpose: Verify that FiLM with γ=1, β=0 produces identical results to pretrained baseline

Method:

# Initialize FiLM to identity: h' = γ*h + β = 1*h + 0 = h

model = load_pretrained()
model.\_init_film_identity() # NEW method to implement

# Verify identity

film = model.ddpm.dynamics.film_network
gamma = film[-1].bias[:joint_nf]
beta = film[-1].bias[joint_nf:]
assert torch.allclose(gamma, torch.ones_like(gamma), atol=1e-6)
assert torch.allclose(beta, torch.zeros_like(beta), atol=1e-6)

Implementation (lightning_modules.py):
def \_init_film_identity(self):
"""Initialize FiLM to identity transformation (γ=1, β=0)."""
film = self.ddpm.dynamics.film_network
joint_nf = self.ddpm.dynamics.joint_nf

     with torch.no_grad():
         # Zero all weights (linear transformation becomes just bias)
         for layer in film:
             if hasattr(layer, 'weight'):
                 layer.weight.zero_()
                 layer.bias.zero_()

         # Set final layer bias: [gamma=1, beta=0]
         film[-1].bias[:joint_nf] = 1.0   # gamma
         film[-1].bias[joint_nf:] = 0.0   # beta

Validation Tests:

1.  Unit test (tests/test_film_identity.py): Forward pass with/without pocket_emb should produce identical outputs (within
    1e-6)
2.  Integration test: Generate 50 molecules, connectivity should match baseline ±5%
3.  Loss test: Evaluate on validation set, loss should be within ±0.05 of baseline

Success Criterion: All tests pass. If fails → identity initialization is broken, must fix before proceeding.

---

Baseline 3: Random-Initialized FiLM (Negative Control)

Purpose: Verify that FiLM actually affects the model (not just a no-op)

Method:

# Load pretrained with DEFAULT random FiLM initialization

model = load_pretrained()

# DON'T call \_init_film_identity() - use random weights

Expected Results:

- Loss: 1.0-2.0+ (much worse than identity)
- Connectivity: 0-30% (poor)
- Validity: 50-80% (degraded)

Success Criterion: Random init is MUCH WORSE than identity init. If random ≈ identity → BUG: FiLM not being used in forward
pass.

---

Part 3: Enhanced Metrics Dashboard

Training Metrics (Log Every Step)

Current logging (lightning_modules.py:541):
self.log_metrics(info, "train", batch_size=len(data["num_lig_atoms"]))

ADD these FiLM diagnostics (NEW):

# After training_step, before return

if self.global_step % 10 == 0: # Every 10 steps
film = self.ddpm.dynamics.film_network
film_bias = film[-1].bias.data
joint_nf = self.ddpm.dynamics.joint_nf

     gamma = film_bias[:joint_nf]
     beta = film_bias[joint_nf:]

     self.log('film/gamma_mean', gamma.mean(), on_step=True)
     self.log('film/gamma_std', gamma.std(), on_step=True)
     self.log('film/beta_mean', beta.mean(), on_step=True)
     self.log('film/beta_std', beta.std(), on_step=True)
     self.log('film/gamma_drift', (gamma.mean() - 1.0).abs(), on_step=True)

     # Gradient norm
     grad_norm = torch.nn.utils.clip_grad_norm_(
         self.ddpm.dynamics.film_network.parameters(), max_norm=1e9
     )
     self.log('film/grad_norm', grad_norm, on_step=True)

What to Watch:

- gamma_mean starts at 1.0, should stay in [0.5, 2.0] during training
- gamma_drift measures deviation from identity (larger = more learned)
- If gamma_mean > 10 or < 0.1 → FiLM exploded (reduce LR)
- If gamma_drift < 0.01 after 20 epochs → FiLM not learning (increase capacity/LR)

---

Validation Metrics (Log Every Epoch)

Current: Connectivity, validity, QED, SA logged at eval_epochs (10)

ADD frequent checkpointing:

# film_finetuning.yml - change from:

eval_epochs: 10

# to:

eval_epochs: 2 # Evaluate every 2 epochs to catch issues early

ADD early stopping callback (train.py):
from lightning.pytorch.callbacks import EarlyStopping

early_stop = EarlyStopping(
monitor='Connectivity/val',
patience=5,
mode='max',
min_delta=0.05, # Stop if connectivity drops >5%
verbose=True
)
trainer.callbacks.append(early_stop)

---

Red Flags (Stop Training If Detected)

| Metric               | Threshold     | Meaning             | Action                        |
| -------------------- | ------------- | ------------------- | ----------------------------- |
| Connectivity/val     | < 50%         | Model broken        | Stop, use previous checkpoint |
| Val/Train loss ratio | > 10x         | Severe overfitting  | Stop, use best checkpoint     |
| film/grad_norm       | > 100         | Exploding gradients | Reduce LR by 10x              |
| film/gamma_mean      | > 10 or < 0.1 | FiLM exploded       | Restart with lower LR         |

---

Part 4: Debugging the 0% Connectivity Issue

Step-by-Step Diagnosis

Step 1: Validate Pretrained Baseline
cd thesis_work/experiments/day5_film_finetuning
uv run python scripts/validate_baseline.py \
 --checkpoint ../../../checkpoints/crossdocked_fullatom_cond.ckpt \
 --test_dir ../../../data/dummy_dataset/val \
 --n_samples 50

Expected output:
=== BASELINE VALIDATION ===
Connectivity: 95.0% ± 5.0%
Validity: 98.0% ± 2.0%
QED: 0.45 ± 0.15
Loss: 0.55 ± 0.10
✅ Baseline validated successfully

If connectivity < 90%: Pretrained checkpoint or data is broken. Check:

- Checkpoint file integrity (md5sum)
- Data preprocessing (10-feature encoding matches checkpoint)
- RDKit version compatibility

---

Step 2: Test Identity FiLM Unit Test
uv run pytest tests/test_film_identity.py -v

Expected output:
tests/test_film_identity.py::test_film_identity_noop PASSED
✅ Identity FiLM test passed (max diff: 3.45e-07)

If fails: Identity initialization logic is wrong. Check:

- \_init_film_identity() implementation in lightning_modules.py
- Correct indexing for gamma/beta split
- joint_nf value matches architecture (32)

---

Step 3: Check FiLM Parameters at Epoch 9

From your WandB logs, check values at the epoch where connectivity dropped to 0%:

Expected (healthy training):

- gamma_mean: 1.0 → 1.2-1.5 by epoch 9
- gamma_std: 0.0 → 0.2-0.5
- beta_mean: 0.0 → small values
- grad_norm: < 10

If gamma_mean > 5 or < 0.2: FiLM diverged too far from identity

- Cause: Overfitting, FiLM bottleneck too small (960→128→64)
- Fix: Use checkpoint from epoch 5-7 (before divergence)

---

Step 4: Check Validation Data Quality

Your validation set has only 1 sample → extremely noisy metrics.

Issue: Val loss 122-10991 is just noise from 1 sample

Fix: Create larger validation set

# In prepare_data_with_esmc.py, change:

n_train = 9 # Currently
n_val = 1 # Currently → change to 10

# Re-run data prep with 50 train, 10 val, 10 test

---

Step 5: Analyze Overfitting

Current status:

- Train loss: 0.605 → 0.197 (dropped 3x)
- Val loss: ~5000 average
- Gap: 25x (severe overfitting)

Diagnosis: FiLM is memorizing training data, can't generalize

Solution: Use checkpoint BEFORE overfitting occurs

# Check connectivity at each checkpoint

for epoch in 2 4 6 8 10; do
python test.py outputs/best-model-epoch=$epoch.ckpt --n_samples 50
done

# Expected: Connectivity high at epoch 2-6, drops at epoch 8-10

# Use epoch 6 as final model

---

Part 5: Recommended Fixes (After Validation)

Fix 1: Increase FiLM Capacity

Problem: 960→128→64 is 15:1 compression, too aggressive for rich ESM-C embeddings

Solution (dynamics.py:72-76):

# CURRENT:

self.film_network = nn.Sequential(
nn.Linear(960, hidden_nf), # 960 → 128
act_fn,
nn.Linear(hidden_nf, 2\*joint_nf) # 128 → 64
)

# PROPOSED:

self.film_network = nn.Sequential(
nn.Linear(960, 512), # 960 → 512
nn.Dropout(0.15), # Regularization
act_fn,
nn.Linear(512, 256), # 512 → 256
nn.Dropout(0.15),
act_fn,
nn.Linear(256, 2\*joint_nf) # 256 → 64
)

Rationale: Less aggressive compression (4:1 instead of 15:1), more capacity to learn ESM-C modulation

---

Fix 2: Enhanced Regularization

Config changes (film_finetuning.yml):

# CURRENT:

lr: 1.0e-3

# clip_grad: true (default threshold: 2.0)

# PROPOSED:

lr: 3.0e-4 # Reduce by 3x
weight_decay: 1.0e-4 # Add L2 regularization
clip_grad_value: 3.0 # Increase clip threshold

Rationale: Lower LR prevents FiLM from diverging too fast, weight decay prevents overfitting

---

Fix 3: Frequent Evaluation & Early Stopping

Config changes:

# CURRENT:

eval_epochs: 10 # Only evaluate every 10 epochs

# PROPOSED:

eval_epochs: 2 # Evaluate every 2 epochs

Callback (train.py):
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

callbacks = [
EarlyStopping(
monitor='Connectivity/val',
patience=5,
mode='max',
min_delta=0.05
),
ModelCheckpoint(
monitor='Connectivity/val',
mode='max',
save_top_k=3,
filename='best-connectivity-{epoch}-{Connectivity/val:.2f}'
)
]

Rationale: Catch connectivity degradation early, save best checkpoints before overfitting

---

Part 6: Systematic Validation Protocol

Week 1: Establish Baselines

Day 1: Validate Pretrained

# 1. Create validation script

touch thesis_work/experiments/day5_film_finetuning/scripts/validate_baseline.py

# 2. Run baseline validation

uv run python scripts/validate_baseline.py

# 3. Expected: >90% connectivity, loss ~0.5-0.6

# 4. If fails: STOP, investigate checkpoint/data

Day 2: Test Identity FiLM

# 1. Implement \_init_film_identity() in lightning_modules.py

# 2. Create unit test

mkdir -p tests
touch tests/test_film_identity.py

# 3. Run test

uv run pytest tests/test_film_identity.py -v

# 4. Expected: Test passes (diff < 1e-6)

# 5. If fails: Fix identity init logic

Day 3: Test Random FiLM

# 1. Load pretrained with random FiLM, evaluate

# 2. Expected: Much worse than identity (connectivity <30%)

# 3. If random ≈ identity: BUG in forward pass

---

Week 2: Fix Architecture & Retrain

Day 1-2: Implement Larger FiLM

- Modify dynamics.py to use 960→512→256→64 architecture
- Add dropout layers (0.15)
- Run unit test to verify identity still works

Day 3: Update Config & Retrain

- Reduce LR to 3e-4
- Add weight decay 1e-4
- Change eval_epochs to 2
- Retrain for 50 epochs

Day 4-5: Monitor Training

- Watch FiLM diagnostics (gamma_mean, gamma_drift)
- Check connectivity every 2 epochs
- Stop if connectivity drops below 80%
- Expected best checkpoint: epoch 15-25

---

Week 3: Systematic Comparison

Day 1: Generate Molecules at All Checkpoints
for epoch in 2 4 6 8 10 12 14 16 18 20; do
uv run python test.py \
 outputs/film-v2/best-connectivity-epoch=${epoch}.ckpt \
         --test_dir data/dummy_dataset/test \
         --n_samples 100 \
         --outdir results/epoch_${epoch}
done

Day 2: Analyze All Results
for epoch in 2 4 6 8 10 12 14 16 18 20; do
uv run python analysis/analyze*results.py results/epoch*${epoch}
done

Day 3: Statistical Comparison

# Compare best FiLM checkpoint vs baseline

# Use t-test for QED, SA, LogP

# Report effect sizes (Cohen's d)

# Check p-values (significance at p<0.05)

Day 4-5: Create Report

- Table: Baseline vs FiLM metrics
- Plots: Loss curves, connectivity over epochs, FiLM parameter evolution
- Molecules: Visualize top 10 from each method
- Conclusion: Did FiLM improve over baseline?

---

Part 7: Scripts to Implement

Script 1: validate_baseline.py

Location: thesis_work/experiments/day5_film_finetuning/scripts/validate_baseline.py

Purpose: Measure pretrained baseline performance

Key functions:

- Load checkpoint without FiLM modifications
- Generate 50-100 molecules on test set
- Compute connectivity, validity, QED, SA
- Save results to JSON
- Print pass/fail status

---

Script 2: test_film_identity.py

Location: tests/test_film_identity.py

Purpose: Unit test for identity initialization

Key tests:

- Initialize FiLM to identity
- Forward pass with/without pocket_emb should match
- Check gamma=1, beta=0 exactly
- Verify no gradient flow through frozen EGNN

---

Script 3: detect_overfitting.py

Location: thesis_work/experiments/day5_film_finetuning/scripts/detect_overfitting.py

Purpose: Automatically detect overfitting from WandB logs

Key functions:

- Load training/validation loss curves
- Calculate train/val gap
- Identify first epoch where gap > threshold (5x)
- Plot loss curves with overfitting markers
- Recommend best checkpoint (before overfitting)

---

Script 4: compare_checkpoints.py

Location: thesis_work/experiments/day5_film_finetuning/scripts/compare_checkpoints.py

Purpose: Systematic comparison across multiple checkpoints

Key functions:

- Load all checkpoints from directory
- Generate molecules for each
- Analyze metrics
- Create comparison table (CSV/markdown)
- Statistical tests (t-test, effect size)

---

Part 8: Success Criteria

Minimum Viable Result

| Metric       | Baseline | Target  | Status           |
| ------------ | -------- | ------- | ---------------- |
| Connectivity | 95%      | ≥85%    | ❌ 0% (must fix) |
| Validity     | 98%      | ≥95%    | ✅ 100%          |
| QED          | 0.45     | ≥0.45   | ✅ 0.52          |
| Loss         | 0.55     | 0.4-0.6 | ✅ 0.60          |

Current Status: FAILED due to 0% connectivity. Must validate baseline and fix architecture before proceeding.

---

Publication-Quality Result

Requirements:

1.  Baseline validation: Pretrained checkpoint achieves >90% connectivity ✅
2.  Identity test: Identity FiLM reproduces baseline exactly (diff < 1e-6) ✅
3.  Statistical significance: FiLM improves QED/SA over baseline (p < 0.05, n≥100)
4.  Multiple seeds: 3 random seeds show consistent improvement
5.  Robustness: Best checkpoint maintains >85% connectivity
6.  Ablation study: Compare baseline, identity, random, trained FiLM

---

Critical Files to Modify

1.  lightning_modules.py

Lines 507-560 (training/validation steps)

Changes:

- Add \_init_film_identity() method
- Add FiLM diagnostic logging (gamma/beta mean/std)
- Add gradient norm logging
- Implement early stopping callback

---

2.  dynamics.py

Lines 72-76 (FiLM network definition)

Changes:

- Increase capacity: 960→512→256→64
- Add dropout layers (0.15)
- Keep identity initialization compatible

---

3.  film_finetuning.yml

Lines 21, 57 (training config)

Changes:

- Reduce LR: 1e-3 → 3e-4
- Add weight_decay: 1e-4
- Change eval_epochs: 10 → 2
- Add early stopping patience: 5

---

4.  New Scripts (Create)

- scripts/validate_baseline.py - baseline measurement
- tests/test_film_identity.py - unit test
- scripts/detect_overfitting.py - automatic diagnosis
- scripts/compare_checkpoints.py - systematic comparison

---

Summary: Answering Your Questions

Q1: Is loss of 0.6 good or bad?

Answer: GOOD.

For diffusion models, loss of 0.4-0.6 is normal and expected. The loss measures noise prediction error, not generation
quality. Your loss of 0.6 is fine.

Stop worrying about loss. Focus on connectivity (currently 0%).

---

Q2: Does identity initialization work correctly?

Answer: UNKNOWN - Must validate.

The identity initialization should make FiLM act as a no-op (h' = 1\*h + 0 = h). This needs validation:

1.  Implement \_init_film_identity() method
2.  Run unit test comparing forward pass with/without pocket_emb
3.  Expected: identical outputs (diff < 1e-6)

If test passes: Identity init works, proceed to training
If test fails: Identity init broken, must fix first

---

Q3: Is training actually improving the model?

Answer: UNCLEAR - Overfitting detected.

Evidence:

- Train loss: 0.605 → 0.197 (improving)
- Val loss: 122-10991 (extremely volatile)
- Connectivity: 100% → 0% (degrading)
- Train/val gap: ~50x (severe overfitting)

Diagnosis: FiLM is memorizing training data due to:

1.  Too small capacity (960→128→64 bottleneck)
2.  Too high LR (1e-3)
3.  No regularization
4.  Tiny dataset (9 samples)

Solution:

1.  Validate baseline first
2.  Increase FiLM capacity (960→512→256→64)
3.  Add regularization (dropout, weight decay, lower LR)
4.  Frequent checkpointing (every 2 epochs)
5.  Use best checkpoint BEFORE overfitting (likely epoch 10-15)

---

Next Steps

1.  Validate baseline (scripts/validate_baseline.py) → Verify pretrained works
2.  Test identity FiLM (tests/test_film_identity.py) → Verify no-op behavior
3.  Increase FiLM capacity (dynamics.py) → Fix bottleneck
4.  Add regularization (film_finetuning.yml) → Reduce overfitting
5.  Retrain with monitoring → Catch issues early
6.  Systematic comparison → Baseline vs FiLM statistical test

Expected Timeline: 2-3 weeks to complete full validation framework
