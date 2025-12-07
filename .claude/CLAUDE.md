# DiffSBDD + ESM-C Thesis Project

> **Master Thesis**: Protein Context Conditioning for Structure-Based Drug Design
> **Approach**: FiLM-only fine-tuning (131K params) on pretrained DiffSBDD (2M params frozen)

---

## Current Status

**Day 5**: FiLM Fine-Tuning Implementation
**Goal**: Train FiLM adapter to learn ESM-C → feature modulation

### Today's Tasks

```bash
# 1. Verify pretrained checkpoint works
uv run python generate_ligands.py checkpoints/crossdocked_fullatom_cond.ckpt \
    --pdbfile example/3rfm.pdb --outfile example/test.sdf --ref_ligand A:330 --n_samples 5

# 2. Implement FiLM identity init + freeze EGNN
# 3. Thread pocket_emb through inference
# 4. Run first FiLM-only training
```

---

## Core Concept

```
Pretrained Checkpoint              New FiLM Layer
────────────────────              ───────────────
EGNN (2M params)          +       FiLM (131K params)
   ↓ frozen                          ↓ trainable
Knows spatial chemistry            Learns: ESM-C → γ,β

Combined:  h' = γ(ESM-C) · h + β(ESM-C)
           └── FiLM modulates EGNN features based on pocket context
```

**Why this works:**
- Pretrained model generates valid, connected molecules
- FiLM learns pocket-specific modulation without changing core chemistry
- Identity init (γ=1, β=0) ensures baseline behavior initially

---

## Key Files to Modify

| File | Change | Status |
|------|--------|--------|
| `lightning_modules.py` | `load_pretrained_with_esmc()`, `_init_film_identity()` | TODO |
| `lightning_modules.py` | `configure_optimizers()` for FiLM-only | TODO |
| `lightning_modules.py` | `generate_ligands()` accept `pocket_emb` | TODO |
| `conditional_model.py` | Thread `pocket_emb` through sampling | TODO |
| `generate_ligands.py` | Add `--esmc_emb` argument | TODO |

**Already done:**
- `dynamics.py:55-61` — FiLM network defined
- `dynamics.py:119-131` — Forward pass handles `pocket_emb`

---

## Quick Reference

### Checkpoint Loading
```python
# Load pretrained + init FiLM to identity
model = LigandPocketDDPM.load_pretrained_with_esmc(
    "checkpoints/crossdocked_fullatom_cond.ckpt"
)
```

### FiLM Identity Init
```python
# gamma=1, beta=0 → h' = 1·h + 0 = h (no change)
film[-1].bias.data[:joint_nf] = 1.0   # gamma
film[-1].bias.data[joint_nf:] = 0.0   # beta
```

### FiLM-Only Training
```python
# Freeze EGNN, train only FiLM
for p in model.ddpm.parameters():
    p.requires_grad = False
for p in model.ddpm.dynamics.film_network.parameters():
    p.requires_grad = True
```

---

## Documentation

| Document | Purpose |
|----------|---------|
| `thesis_work/daily_logs/INDEX.md` | Daily progress |
| `thesis_work/experiments/day5_film_finetuning/` | Current experiment |
| `.claude/plans/mellow-percolating-reef.md` | Full implementation plan |

### Archive (old approach)
- `.claude/archive/` — Previous full-training documentation
- `thesis_work/experiments/_legacy/` — Days 1-4 experiments

---

## Session Checklist

1. [ ] Check `thesis_work/daily_logs/INDEX.md`
2. [ ] Review current task in plan: `.claude/plans/mellow-percolating-reef.md`
3. [ ] Update daily log when done
