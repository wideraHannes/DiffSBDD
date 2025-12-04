# ESM-C Integration Implementation Plan

> **Goal**: Implement ESM-C conditioning for DiffSBDD using validation-first approach

**Created**: December 2024
**Methodology**: Google Research Tuning Playbook
**Baseline Strategy**: Use test.py to validate existing checkpoint

---

## Overview

We're implementing global pocket conditioning using ESM-C embeddings via FiLM modulation. This follows the text-to-image analogy where:

- **Text prompt → Image** becomes **Pocket sequence → Ligand**
- **CLIP embeddings** becomes **ESM-C embeddings**
- **FiLM/cross-attention** stays **FiLM conditioning**

**Key Design**: Single 960-dim global embedding per pocket (not per-residue) for:

- Clean separation of geometric (EGNN) and semantic (ESM-C) information
- Simple implementation (~50 lines of code)
- Fast iteration and debugging
- Clear experimental design (with/without ESM-C)

---

## Scientific Validation Approach (Tuning Playbook)

**Inspired by**: [Google Research Tuning Playbook](https://github.com/google-research/tuning_playbook)

### Core Principles

1. ✅ **Start simple, add complexity incrementally**
2. ✅ **Change one thing at a time** (isolate variables)
3. ✅ **Form hypothesis → test → analyze → iterate**
4. ✅ **Understand model before scaling** (small data first)
5. ✅ **Focus on validation metrics, not training loss**

---

## Timeline: Validation-First Approach

### Phase 0: Scientific Validation (Week 1)

**Goal**: Validate architecture works and ESM-C embeddings are properly integrated

| Day       | Experiment                    | Hypothesis                           | Hours |
| --------- | ----------------------------- | ------------------------------------ | ----- |
| **Day 1** | ESM-C setup + integration     | Setup infrastructure                 | 6h    |
| **Day 2** | Embedding analysis            | ESM-C contains meaningful signal     | 4h    |
| **Day 3** | Overfit test (1 sample)       | Architecture can learn with ESM-C    | 4h    |
| **Day 4** | Small dataset (100 samples)   | Model trains successfully with ESM-C | 6h    |
| **Day 5** | Medium dataset (1000 samples) | Scales to larger data                | 6h    |
| **Day 6** | Gradient & FiLM analysis      | Verify ESM-C signal is used          | 4h    |
| **Day 7** | Go/No-Go decision             | Decide on full training              | 2h    |

**Decision point**: If validation passes → proceed to full training (Week 2-3)

**Note**: Baseline comparison will be done later using the existing checkpoint from the authors. Focus now is on ESM-C implementation and validation.

### Phase 1: Full Training (Week 2-3, conditional on Phase 0)

| Day           | Focus                          | Hours |
| ------------- | ------------------------------ | ----- |
| **Day 8-9**   | Launch full training           | 4h    |
| **Day 10-14** | Monitor training, early eval   | 8h    |
| **Day 15-17** | Full evaluation + analysis     | 12h   |
| **Day 18-21** | Deep analysis + thesis figures | 16h   |

**Total Phase 0**: ~32 hours (validation)
**Total Phase 1**: ~40 hours (full implementation)

---

## Codebase Integration Points

**Critical modifications identified**:

| File                   | Lines              | Modification                                            |
| ---------------------- | ------------------ | ------------------------------------------------------- |
| `dataset.py`           | 8, 12-13, 47       | Add esmc_path, load embeddings, return in **getitem**   |
| `dynamics.py`          | 49, 87, 101        | Add FiLM network, update signature, apply conditioning  |
| `conditional_model.py` | 253, 306, 445, 119 | Pass pocket_emb to 4 dynamics calls                     |
| `en_diffusion.py`      | 516, 270           | Pass pocket_emb to 2 dynamics calls                     |
| `lightning_modules.py` | 211, 266-271, 952  | Setup esmc_path, add to pocket dict, load for inference |

**Total dynamics() calls to update**: 8 locations across codebase

---

## Key Success Criteria

### Go/No-Go Decision Matrix

| Scenario | Day 4-5 Result              | Day 6 Result           | Decision                          |
| -------- | --------------------------- | ---------------------- | --------------------------------- |
| **A**    | Training converges smoothly | FiLM active & learning | **GO** - Full training            |
| **B**    | Training converges          | FiLM near identity     | **GO** - Model may not need ESM-C |
| **C**    | Training unstable           | Gradient issues found  | **STOP** - Debug architecture     |
| **D**    | Training diverges           | No clear bug           | **STOP** - Review implementation  |
| **E**    | Reasonable metrics          | FiLM active            | **GO** - Proceed with caution     |

**Key Insight**: Focus is on validating the ESM-C integration works correctly. Comparison to baseline will come later!

---

## Quick Reference

**See detailed daily breakdowns in this file for**:

- Experiment 1.1-1.2: Baseline validation + ESM-C setup (Day 1)
- Experiment 2.1: Embedding signal analysis (Day 2)
- Experiment 3: Overfit test - architecture validation (Day 3)
- Experiment 4: Small dataset comparison (Day 4)
- Experiment 5: Medium dataset scaling (Day 5)
- Experiment 6.1-6.2: Gradient flow + FiLM analysis (Day 6)
- Decision framework (Day 7)

**Full implementation details** for Days 1-7 are provided below with:

- Specific code snippets
- Success criteria
- Debugging strategies
- Analysis scripts

---

## 📋 Implementation Checklist

Track your progress through the implementation. Check off items as you complete them!

### Phase 0: Scientific Validation

#### Day 1: ESM-C Setup + Integration ✅ COMPLETE

- [x] **1.1 ESM-C Infrastructure**

  - [x] from esm.sdk import client model = client(
        model="esmc-300m-2024-12", url="https://forge.evolutionaryscale.ai", token=token
        )
  - [x] Test ESM-C on single protein
  - [x] Extract embedding for 1 test pocket
  - [x] Verify embedding shape (960,) and statistics
  - [x] Create extraction script template

- [x] **1.2 Code Integration**
  - [x] Modify `dataset.py` - add esmc_path parameter
  - [x] Modify `dynamics.py` - add FiLM network
  - [x] Update all 8 dynamics() calls in codebase
  - [x] Create ESM-C config file
  - [x] Test imports and basic functionality

#### Day 2: Embedding Analysis ⏭️ IN PROGRESS

- [ ] **2.1 Signal Validation**
  - [ ] Extract embeddings for 100 test pockets
  - [ ] Compute embedding similarity matrix
  - [ ] Analyze correlation with binding affinity
  - [ ] Visualize with t-SNE/UMAP
  - [ ] Calculate mutual information I(ESM-C; Vina score)
  - [ ] Document findings (proceed if I > 0)

#### Day 3: Overfit Test (1 Sample)

- [ ] **3.1 Architecture Validation**
  - [ ] Extract ESM-C embedding for 1 training sample
  - [ ] Create config for 1-sample overfit test
  - [ ] Train ESM-C model on 1 sample (50 epochs)
  - [ ] Verify training loss reaches ~0
  - [ ] Check FiLM parameters change during training
  - [ ] Verify model generates valid molecule for that pocket
  - [ ] Debug any issues before proceeding

#### Day 4: Small Dataset (100 Samples)

- [ ] **4.1 Training Validation**
  - [ ] Extract ESM-C for 100 train samples
  - [ ] Create config for small dataset
  - [ ] Train ESM-C model (100 samples, 50 epochs)
  - [ ] Monitor training and validation loss
  - [ ] Check loss decreases appropriately
  - [ ] Generate 10 molecules per pocket
  - [ ] Compute validity, QED, SA metrics
  - [ ] Verify reasonable molecule quality

#### Day 5: Medium Dataset (1000 Samples)

- [ ] **5.1 Scaling Test**
  - [ ] Extract ESM-C for 1000 train samples
  - [ ] Create config for medium dataset
  - [ ] Train ESM-C model (1000 samples, 100 epochs)
  - [ ] Monitor loss curves for convergence
  - [ ] Early stopping if issues arise
  - [ ] Generate molecules for validation
  - [ ] Verify scaling behavior is healthy

#### Day 6: Gradient & Ablation Analysis

- [ ] **6.1 Gradient Flow Analysis**

  - [ ] Check gradients flowing to FiLM network
  - [ ] Verify gradients to ESM-C embedding (should be frozen)
  - [ ] Compute gradient norms across layers
  - [ ] Identify any gradient issues

- [ ] **6.2 FiLM Parameter Analysis**
  - [ ] Extract γ and β for all validation samples
  - [ ] Plot distribution (mean, std, percentiles)
  - [ ] Check if near identity (γ≈1, β≈0) → model ignoring ESM-C
  - [ ] Analyze per-channel modulation patterns
  - [ ] Visualize most modulated channels

#### Day 7: Go/No-Go Decision

- [ ] **7.1 Validation Review**
  - [ ] Compile all results from Days 2-6
  - [ ] Verify ESM-C embeddings contain signal
  - [ ] Confirm architecture trains successfully
  - [ ] Check FiLM network is active (not identity)
  - [ ] Review molecule quality metrics
  - [ ] Make Go/No-Go decision for full training
  - [ ] If GO: proceed to Phase 1
  - [ ] If issues: debug and iterate

---

### Phase 1: Full Training (If GO Decision)

#### Week 2: Full Training Launch

- [ ] **8.1 Data Preparation**

  - [ ] Extract ESM-C for all train data (~100k samples)
  - [ ] Extract ESM-C for all val data
  - [ ] Extract ESM-C for all test data
  - [ ] Verify all cached .npz files

- [ ] **8.2 Training Setup**
  - [ ] Create full training config
  - [ ] Set up logging (WandB/TensorBoard)
  - [ ] Launch ESM-C training
  - [ ] Monitor first 24 hours closely
  - [ ] Verify GPU utilization and training speed

#### Week 2-3: Training Monitoring

- [ ] **9.1 Training Progress**
  - [ ] Check loss curves daily
  - [ ] Validate every 20 epochs
  - [ ] Early stopping if needed
  - [ ] Save best checkpoint
  - [ ] Document training dynamics

#### Week 3: Evaluation

- [ ] **10.1 Quantitative Evaluation**

  - [ ] Generate 100 molecules per test pocket
  - [ ] Compute validity (%)
  - [ ] Compute connectivity (%)
  - [ ] Compute uniqueness (%)
  - [ ] Compute QED scores
  - [ ] Compute SA scores
  - [ ] Run Vina docking
  - [ ] Compute diversity metrics
  - [ ] Statistical significance tests

- [ ] **10.2 Qualitative Analysis**
  - [ ] Visualize example molecules
  - [ ] Case studies (best/worst pockets)
  - [ ] Success and failure mode analysis

#### Week 3-4: Deep Analysis

- [ ] **11.1 FiLM Analysis**

  - [ ] Extract all FiLM parameters
  - [ ] Visualize γ and β distributions
  - [ ] Analyze modulation patterns
  - [ ] Interpret modulated channels

- [ ] **11.2 Gradient Attribution**

  - [ ] Compute ESM-C dimension importance
  - [ ] Integrated gradients analysis
  - [ ] Identify critical dimensions

- [ ] **11.3 Embedding Space**

  - [ ] t-SNE visualization
  - [ ] Pocket clustering
  - [ ] Ligand-pocket relationships

- [ ] **11.4 Thesis Figures**
  - [ ] Architecture diagram
  - [ ] Results table
  - [ ] Loss curves comparison
  - [ ] Metric distributions
  - [ ] FiLM parameter plots
  - [ ] Case study visualizations
  - [ ] Ablation study results

---

### Continuous Tasks (Throughout)

- [ ] **Documentation**

  - [ ] Keep daily lab notebook
  - [ ] Document all experiments
  - [ ] Save all figures
  - [ ] Track hyperparameters
  - [ ] Note bugs and solutions

- [ ] **Code Quality**

  - [ ] Write unit tests
  - [ ] Ensure backward compatibility
  - [ ] Comment code thoroughly
  - [ ] Follow configuration patterns
  - [ ] Version control commits

- [ ] **Validation**
  - [ ] Test both baseline and ESM-C modes
  - [ ] Verify configs work
  - [ ] Check checkpoint saving/loading
  - [ ] Validate generated molecules

---

## Progress Summary

**Track completion rates**:

- Phase 0 (Days 1-7): `[x] 10/40` tasks completed (Day 1 ✅)
- Phase 1 (Weeks 2-4): `[ ] 0/28` tasks completed
- Continuous Tasks: `[ ] 0/15` tasks completed

**Total Progress**: `[x] 10/83` tasks completed (12%)

**Current Status**: Day 2 (Embedding Analysis) in progress

**Note**: Baseline training tasks removed - comparison will be done later using existing checkpoint.

---

_For complete day-by-day implementation details, see sections below..._
