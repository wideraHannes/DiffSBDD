# Master Thesis: ESM-C Conditioning for Structure-Based Drug Design

> **Research Question**: Can evolutionary context from protein language models (ESM-C) improve structure-based drug design through efficient adapter-based conditioning?

---

## 1. The Core Hypothesis

### What We're Testing

A **FiLM adapter** trained on ESM-C pocket embeddings can steer a frozen pretrained diffusion model toward:

- Better binding affinity (lower SMINA scores)
- Maintained chemical validity and connectivity
- Improved drug-like properties

### Key Innovation: FiLM-Only Fine-Tuning

```
Pretrained DiffSBDD (2M params)   →   FROZEN
                    +
FiLM Adapter (131K params)        →   TRAINED

Result: 6.5% new params, 100% of pretrained knowledge preserved
```

This is analogous to **LoRA for LLMs** or **adapters for vision models**.

---

## 2. The Text-to-Image Analogy

| Text-to-Image (Stable Diffusion) | Pocket-to-Ligand (This Thesis) |
|----------------------------------|--------------------------------|
| Text prompt | Protein pocket sequence |
| CLIP encoder | ESM-C encoder |
| 768-dim text features | 960-dim pocket features |
| Cross-attention / FiLM | **FiLM adapter** |
| U-Net (pretrained) | **EGNN (pretrained, frozen)** |
| Realistic image | Realistic ligand |

**What ESM-C Captures** (that one-hot doesn't):
- Evolutionary conservation
- Structural context
- Functional motifs
- Physicochemical patterns

---

## 3. Architecture

### Current Approach: FiLM-Only Fine-Tuning

```
                    PRETRAINED (frozen)
                    ┌─────────────────┐
Pocket coords ────►│                 │
Pocket one-hot ───►│      EGNN       │────► Predicted noise
Ligand coords ────►│   (2M params)   │
Ligand one-hot ───►│                 │
                    └────────▲────────┘
                             │
                    ┌────────┴────────┐
                    │  FiLM Adapter   │ ◄─── TRAINED (131K params)
                    │  h' = γ·h + β   │
                    └────────▲────────┘
                             │
                    ┌────────┴────────┐
                    │    ESM-C        │
                    │ (960-dim emb)   │
                    └─────────────────┘
                             ▲
                    Pocket sequence
```

### FiLM Network

```python
FiLM: Linear(960 → 128) → SiLU → Linear(128 → 64)
      Output: gamma (32-dim), beta (32-dim)
      Modulation: h' = gamma * h + beta
```

### Why FiLM-Only Works

1. **Pretrained model generates valid molecules** (connectivity solved)
2. **FiLM learns modulation** without changing core chemistry
3. **Identity init** ensures baseline behavior initially (γ=1, β=0)
4. **Fast iteration** (~131K params trains in hours, not days)

---

## 4. Implementation Timeline

### Phase 1: Full Training (Archived)

| Day | Focus | Outcome |
|-----|-------|---------|
| Days 1-2 | ESM-C setup | Working |
| Days 3-4 | Full training | **Blocked** (0% connectivity) |

**Lesson:** Full training from scratch has stability issues. Pivoted to fine-tuning.

### Phase 2: FiLM Fine-Tuning (Current)

| Day | Focus | Status |
|-----|-------|--------|
| **Day 5** | Implement FiLM-only training | **In Progress** |
| Day 6 | Evaluation & docking | Planned |
| Day 7 | HPC scaling & results | Planned |

### Key Implementation Tasks

1. Load pretrained checkpoint with `strict=False`
2. Initialize FiLM to identity (γ=1, β=0)
3. Freeze EGNN, train only FiLM
4. Thread `pocket_emb` through inference
5. Evaluate binding affinity improvement

---

## 5. Evaluation Metrics

### Primary Comparison

| Metric | Baseline | FiLM + ESM-C | Better |
|--------|----------|--------------|--------|
| SMINA (kcal/mol) | -X.XX | -X.XX | Lower |
| Validity | >90% | >90% | Same |
| Connectivity | >80% | >80% | Same |
| QED | 0.XX | 0.XX | Higher |

### Statistical Test

Paired t-test across N pockets:
- Generate 20 molecules per pocket (baseline vs ESM-C)
- Dock with SMINA
- Compare mean binding affinity per pocket

---

## 6. Expected Outcomes

### Scenario A: ESM-C Improves Binding

**Result:** Statistically significant improvement in SMINA scores (p < 0.05)

**Conclusion:** "Protein language models provide valuable evolutionary context that improves structure-based drug design when efficiently integrated via FiLM adapters."

### Scenario B: No Significant Improvement

**Result:** No statistical difference in binding affinity

**Conclusion:** "Local geometric interactions dominate ligand binding. Evolutionary context provides complementary but non-essential information for CrossDocked."

**Still valuable:** Negative results with proper analysis are publishable.

---

## 7. Thesis Structure (Draft)

| Chapter | Pages | Content |
|---------|-------|---------|
| 1. Introduction | 5-8 | Motivation, problem, contributions |
| 2. Background | 10-15 | SBDD, diffusion models, protein LMs, FiLM |
| 3. Method | 8-12 | Architecture, FiLM-only fine-tuning |
| 4. Experiments | 12-18 | Setup, results, statistical analysis |
| 5. Discussion | 5-8 | Findings, limitations, future work |
| 6. Conclusion | 2-3 | Summary |

**Total:** ~50-70 pages

---

## 8. Key Strengths

- **Novel:** First FiLM adapter approach for SBDD with protein LMs
- **Efficient:** 6.5% trainable params, hours not days to train
- **Low Risk:** Pretrained model works, we just add steering
- **Interpretable:** FiLM parameters show how ESM-C affects features
- **Reproducible:** Clear implementation, available checkpoint

---

## 9. References

1. Schneuing et al. (2024) - DiffSBDD - _Nature Computational Science_
2. Hayes et al. (2024) - ESM-C - "Simulating 500M years of evolution"
3. Ho et al. (2020) - DDPM
4. Perez et al. (2018) - FiLM conditioning
5. Rombach et al. (2022) - Stable Diffusion

---

_See `CODE_REFERENCE.md` for implementation details_
_See `plans/mellow-percolating-reef.md` for full implementation plan_
