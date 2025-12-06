# DiffSBDD + ESM-C Thesis Project

> **Master Thesis**: Protein Context Conditioning for Structure-Based Drug Design
> **Core Idea**: Use ESM-C protein embeddings to steer ligand generation (like CLIP steers Stable Diffusion)

---

## 🎯 Current Status

**Phase**: Day 3 - Overfit Testing  
**Focus**: Validate architecture can learn on small data before full training  
**Blocking Issue**: 0% connectivity (molecules fragmented) at loss ~0.5

### Quick Commands
```bash
# Train overfit test (5 samples)
uv run python train.py --config thesis_work/experiments/day3_overfit/configs/day3_overfit_5sample.yml

# Generate molecules from checkpoint
uv run python generate_ligands.py <checkpoint> --pdbfile <pdb> --outdir <output>

# View wandb dashboard
# https://wandb.ai/johannes-widera-heinrich-heine-university-d-sseldorf/ligand-pocket-ddpm
```

### Key Metrics to Watch
| Metric | Good Value | Current |
|--------|-----------|---------|
| `loss/train` | < 0.2 for overfit | ~0.5 |
| Connectivity | > 80% | 0% ❌ |
| Validity | > 90% | 100% ✓ |

---

## 📁 Documentation Map

| Document | Purpose | When to Read |
|----------|---------|--------------|
| **CLAUDE.md** (this) | Entry point, current status | Start of each session |
| **THESIS.md** | Research question, architecture, timeline | Understanding the thesis |
| **CODE_REFERENCE.md** | Config flags, code patterns, integration points | When implementing |
| **implementation_plan.md** | Day-by-day tasks with checklists | During implementation |

### Daily Progress
- `thesis_work/daily_logs/INDEX.md` — Quick overview
- `thesis_work/daily_logs/2024-12-04_day3.md` — Today's work

### Experiments
- `thesis_work/experiments/day3_overfit/` — Current experiment

---

## 🧠 Core Concept (30-second summary)

```
Current DiffSBDD:
  Pocket (one-hot + coords) → EGNN → Ligand
  Problem: One-hot encoding loses evolutionary context

Proposed (+ ESM-C):  
  Pocket sequence → ESM-C → 960-dim embedding → FiLM conditioning
  FiLM modulates ligand features: h' = γ·h + β
  Result: Evolutionary context steers generation
```

**The Analogy**:
```
Text → CLIP → Stable Diffusion → Image
Pocket → ESM-C → DiffSBDD → Ligand
```

---

## 🔧 Key Code Locations

### Modified Files (ESM-C Integration)
| File | What Changed |
|------|--------------|
| `dataset.py` | Loads ESM-C embeddings from cache |
| `dynamics.py` | FiLM network (`self.pocket_film`) |
| `conditional_model.py` | Passes `pocket_emb` to dynamics (4 calls) |
| `en_diffusion.py` | Passes `pocket_emb` to dynamics (2 calls) |
| `lightning_modules.py` | Loads ESM-C path, handles inference |

### Config Flags
```yaml
# Enable ESM-C conditioning
esmc_conditioning: True
esmc_dim: 960
esmc_path: "path/to/embeddings.npz"

# Disable (baseline mode)
esmc_conditioning: False
esmc_path: null
```

### Entry Points
- `train.py` — Training
- `generate_ligands.py` — Inference
- `test.py` — Evaluation

---

## 🐛 Current Debugging Focus

### Problem: 0% Connectivity
**Symptom**: Generated molecules are chemically valid but fragmented (multiple disconnected pieces)

**Likely Causes**:
1. Loss not low enough (0.5 → need < 0.2)
2. Atom positions too spread out after denoising
3. Need more training epochs

**Diagnostic Steps**:
```python
# Check atom distances after generation
from scipy.spatial.distance import pdist
print(f"Distances: min={pdist(positions).min():.2f}, max={pdist(positions).max():.2f}")
# Bonds form at 1-2 Å. If min > 2 Å, atoms too far apart.
```

---

## 📚 Archive

Verbose documentation moved to `.claude/archive/`:
- `THESIS_PLAN.md` — Full 900-line thesis plan
- `06_GLOBAL_POCKET_CONDITIONING.md` — Architecture deep dive
- `CONFIGURATION_GUIDE.md` — Detailed config patterns
- `IN_DEPTH_EXPLANATION.md` — Code walkthrough
- `ARCHITECTURE_SUMMARY.txt` — Quick reference
- `s43588-024-00737-x.pdf` — Original DiffSBDD paper

---

## ⚡ Session Start Checklist

1. [ ] Check `thesis_work/daily_logs/INDEX.md` for yesterday's status
2. [ ] Review current experiment in `thesis_work/experiments/day3_overfit/`
3. [ ] Check wandb for training progress
4. [ ] Update this file's "Current Status" section when starting new work
