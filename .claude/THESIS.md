# Master Thesis: ESM-C Conditioning for Structure-Based Drug Design

> **Research Question**: Can evolutionary context from protein language models (ESM-C) improve structure-based drug design by providing a global steering signal to the diffusion process?

---

## 1. The Core Hypothesis

### What We're Testing
A **single global pocket embedding** from ESM-C (960-dimensional) provides complementary information to local geometric features, steering ligand generation toward:
- More realistic molecular structures
- Better binding affinity
- Improved drug-like properties

### Null Hypothesis (What Might Not Work)
- **Local geometry dominance**: Binding determined purely by local atomic interactions
- **Context fragmentation**: ESM-C trained on full proteins may not capture pocket-specific info
- **Dataset limitations**: CrossDocked artificial complexes may lack evolutionary signal

**Key Insight**: Either outcome is valuable — proving or disproving contributes to understanding!

---

## 2. The Text-to-Image Analogy

| Text-to-Image (Stable Diffusion) | Pocket-to-Ligand (DiffSBDD + ESM-C) |
|----------------------------------|-------------------------------------|
| Text prompt | Protein pocket sequence |
| CLIP encoder | ESM-C encoder |
| 768-dim text features | 960-dim pocket features |
| Cross-attention / FiLM | FiLM modulation |
| U-Net denoising | EGNN denoising |
| Realistic image | Realistic ligand |

**What ESM-C Captures** (that one-hot doesn't):
- Evolutionary conservation (critical residues)
- Structural context (from full protein)
- Functional motifs (binding preferences)
- Physicochemical patterns (hydrophobicity, charge)

---

## 3. Architecture Overview

### Current DiffSBDD (Baseline)
```
Pocket: coords (N, 3) + one-hot AA (N, 20)  ← Limited!
Ligand: coords (M, 3) + one-hot atoms (M, types)
     ↓
  Encode → joint space (128-dim)
     ↓
  EGNN message passing
     ↓
  Predict noise ε
```

### Proposed (+ ESM-C Conditioning)
```
OFFLINE (once per pocket):
  Full protein → ESM-C → (N_total, 960)
  Extract pocket residues → Mean pool → (960,) global embedding
  Cache to disk

ONLINE (training/inference):
  Load cached embedding
     ↓
  FiLM: pocket_emb → MLP → [γ, β]  (each 128-dim)
     ↓
  h_ligand' = γ ⊙ h_ligand + β  ← STEERING!
     ↓
  EGNN (geometric + semantic)
     ↓
  Predict noise ε
```

### Why FiLM Works
- **Multiplicative modulation** stronger than additive
- Used in StyleGAN, FiLM-GAN, conditional models
- Interpretable: can analyze which features are modulated
- Efficient: applies uniformly to all ligand atoms

---

## 4. Implementation Timeline

### Phase 0: Scientific Validation (Week 1) — CURRENT

| Day | Experiment | Status |
|-----|------------|--------|
| Day 1 | ESM-C setup + integration | ✅ Complete |
| Day 2 | Embedding signal analysis | ✅ Complete |
| **Day 3** | Overfit test (1-5 samples) | 🔄 In Progress |
| Day 4 | Small dataset (100 samples) | ⏳ Pending |
| Day 5 | Medium dataset (1000 samples) | ⏳ Pending |
| Day 6 | Gradient & FiLM analysis | ⏳ Pending |
| Day 7 | Go/No-Go decision | ⏳ Pending |

### Phase 1: Full Training (Weeks 2-3)
- Launch full training on GPU cluster
- Monitor metrics, early stopping
- Full evaluation + thesis figures

### Go/No-Go Decision Matrix

| Training Result | FiLM Activity | Decision |
|-----------------|---------------|----------|
| Converges smoothly | Active & learning | **GO** |
| Converges | Near identity (γ≈1, β≈0) | GO (model may not need ESM-C) |
| Unstable | Gradient issues | STOP - Debug |
| Diverges | No clear bug | STOP - Review |

---

## 5. Evaluation Metrics

### Primary Metrics
| Metric | Direction | What It Measures |
|--------|-----------|------------------|
| Validity | ↑ | Chemically valid molecules |
| Connectivity | ↑ | Single connected component |
| QED | ↑ | Drug-likeness (0-1) |
| SA Score | ↓ | Synthetic accessibility (1-10) |
| Vina Score | ↓ | Docking affinity |
| Diversity | ↑ | Scaffold diversity |

### Analysis Metrics
- FiLM parameter distribution (γ, β)
- Gradient attribution (which ESM-C dims matter)
- t-SNE of pocket embeddings

---

## 6. Expected Outcomes

### Scenario A: ESM-C Improves Metrics ✅
| Metric | Baseline | +ESM-C |
|--------|----------|--------|
| Validity | 75% | **82%** |
| QED | 0.45 | **0.52** |
| SA | 3.2 | **2.8** |
| Vina | -7.5 | **-8.3** |

**Conclusion**: "Evolutionary context improves SBDD by capturing binding site characteristics beyond local geometry."

### Scenario B: No Improvement ⚠️
**Conclusion**: "Local geometric interactions dominate. Evolutionary context doesn't significantly influence ligand generation in CrossDocked."

**Still valuable**: Negative results publishable if well-analyzed!

---

## 7. Thesis Structure (Draft)

1. **Introduction** (5-8 pages): Motivation, problem, opportunity, contributions
2. **Background** (10-15 pages): SBDD, diffusion models, protein LMs, conditioning
3. **Method** (8-12 pages): ESM-C extraction, FiLM architecture, training
4. **Experiments** (12-18 pages): Setup, quantitative results, qualitative analysis, ablations
5. **Analysis** (8-12 pages): FiLM parameters, gradients, embeddings
6. **Discussion** (5-8 pages): Findings, limitations, future work
7. **Conclusion** (2-3 pages)

**Total**: ~60-80 pages

---

## 8. Key Strengths of This Approach

✅ **Novel**: First to use protein LM as global steering signal for SBDD  
✅ **Low Risk**: Baseline exists, simple implementation (~50 lines)  
✅ **Rapid**: 8 weeks from start to results  
✅ **Valuable Either Way**: Positive or negative results publishable  
✅ **Interpretable**: FiLM parameters show how conditioning works  
✅ **Extensible**: Can add per-residue embeddings later  

---

## 9. References

**Core Papers**:
1. Schneuing et al. (2024) - DiffSBDD - *Nature Computational Science*
2. Hayes et al. (2024) - ESM-C - "Simulating 500M years of evolution"
3. Ho et al. (2020) - DDPM
4. Perez et al. (2018) - FiLM conditioning
5. Rombach et al. (2022) - Stable Diffusion

---

*For detailed day-by-day implementation: see `implementation_plan.md`*  
*For code patterns and config: see `CODE_REFERENCE.md`*  
*For full verbose docs: see `archive/`*
