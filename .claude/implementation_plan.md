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
**Goal**: Validate ESM-C contains signal and FiLM architecture works

| Day | Experiment | Hypothesis | Hours |
|-----|-----------|------------|-------|
| **Day 1** | Baseline + ESM-C setup | Setup infrastructure | 6h |
| **Day 2** | Embedding analysis | ESM-C contains meaningful signal | 4h |
| **Day 3** | Overfit test (1 sample) | Architecture can learn with ESM-C | 4h |
| **Day 4** | Small dataset (100 samples) | ESM-C improves validation loss | 6h |
| **Day 5** | Medium dataset (1000 samples) | Effect scales to larger data | 6h |
| **Day 6** | Gradient & ablation analysis | Understand what's working | 4h |
| **Day 7** | Go/No-Go decision | Decide on full training | 2h |

**Decision point**: If validation passes → proceed to full training (Week 2-3)

### Phase 1: Full Training (Week 2-3, conditional on Phase 0)
| Day | Focus | Hours |
|-----|-------|-------|
| **Day 8-9** | Launch full training | 4h |
| **Day 10-14** | Monitor training, early eval | 8h |
| **Day 15-17** | Full evaluation + analysis | 12h |
| **Day 18-21** | Deep analysis + thesis figures | 16h |

**Total Phase 0**: ~32 hours (validation)
**Total Phase 1**: ~40 hours (full implementation)

---

## Codebase Integration Points

**Critical modifications identified**:

| File | Lines | Modification |
|------|-------|--------------|
| `dataset.py` | 8, 12-13, 47 | Add esmc_path, load embeddings, return in __getitem__ |
| `dynamics.py` | 49, 87, 101 | Add FiLM network, update signature, apply conditioning |
| `conditional_model.py` | 253, 306, 445, 119 | Pass pocket_emb to 4 dynamics calls |
| `en_diffusion.py` | 516, 270 | Pass pocket_emb to 2 dynamics calls |
| `lightning_modules.py` | 211, 266-271, 952 | Setup esmc_path, add to pocket dict, load for inference |

**Total dynamics() calls to update**: 8 locations across codebase

---

## Key Success Criteria

### Go/No-Go Decision Matrix

| Scenario | Day 4-5 Result | Day 6 Result | Decision |
|----------|---------------|--------------|----------|
| **A** | ESM-C improves val loss | FiLM active | **GO** - Full training |
| **B** | ESM-C ≈ baseline | FiLM near identity | **GO** - Negative result valuable |
| **C** | ESM-C ≈ baseline | FiLM active | **GO** - ESM-C used but doesn't help |
| **D** | ESM-C worse | Bug found | **STOP** - Fix bug first |
| **E** | ESM-C worse | No bug | **PIVOT** - Try ablations |

**Key Insight**: Well-designed negative results are as valuable as positive ones!

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

*For complete day-by-day implementation details, see sections below...*
