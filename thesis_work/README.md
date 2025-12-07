# Thesis Work: ESM-C FiLM Conditioning for DiffSBDD

> **Master's Thesis**: Protein Context Conditioning for Structure-Based Drug Design
> **Approach**: FiLM-only fine-tuning of pretrained DiffSBDD
> **Start Date**: December 2024

---

## Quick Start

```bash
# Check current status
cat thesis_work/daily_logs/INDEX.md

# Run today's experiment
cd thesis_work/experiments/day5_film_finetuning
```

---

## Directory Structure

```
thesis_work/
├── README.md               # You are here
├── daily_logs/             # What happened each day
│   └── INDEX.md            # Start here!
│
├── experiments/            # Active experiments
│   ├── day5_film_finetuning/   # Current: FiLM-only training
│   └── _legacy/                # Archived: old approaches
│
├── configs/                # Training configs
├── results/                # Final figures & tables
│   ├── figures/
│   └── tables/
│
└── analysis/               # Analysis scripts
```

---

## Current Approach: FiLM Fine-Tuning

```
Pretrained DiffSBDD (2M params, frozen)
         │
         ▼
    ┌─────────┐
    │  EGNN   │ ← Learned spatial chemistry (frozen)
    └────┬────┘
         │
    ┌────▼────┐
    │  FiLM   │ ← ESM-C modulation (131K params, trainable)
    └────┬────┘
         │
         ▼
      Ligand
```

**Why this works:**
- Pretrained model already generates valid molecules
- FiLM learns: "Given this pocket context, adjust features like this"
- Clear attribution: any improvement = ESM-C value

---

## Progress

| Phase | Status | Description |
|-------|--------|-------------|
| Days 1-4 | Archived | Full training attempts (connectivity issues) |
| **Day 5** | **Current** | FiLM-only fine-tuning |
| Day 6 | Planned | Evaluation & docking |
| Day 7 | Planned | HPC scaling & results |

---

## Key Files

| File | Purpose |
|------|---------|
| `.claude/CLAUDE.md` | Claude instructions & quick reference |
| `thesis_work/daily_logs/INDEX.md` | Daily progress overview |
| `checkpoints/crossdocked_fullatom_cond.ckpt` | Pretrained baseline |
| `configs/film_finetuning.yml` | FiLM training config |
