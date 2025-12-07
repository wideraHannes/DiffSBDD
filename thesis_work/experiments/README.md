# Experiments

## Current Approach: FiLM-Only Fine-Tuning

| Experiment | Status | Description |
|------------|--------|-------------|
| **day5_film_finetuning** | Active | FiLM adapter training on pretrained model |

---

## Archived (Phase 1: Full Training)

Old experiments in `_legacy/` — these used full training which had connectivity issues:

| Experiment | What We Learned |
|------------|-----------------|
| `day2_embeddings` | ESM-C embeddings have meaningful signal |
| `day3_overfit` | Full training: 0% connectivity at loss 0.5 |
| `day4_baseline_validation` | Pretrained model generates valid molecules |

**Conclusion:** Full training blocked by connectivity. Pivoted to FiLM-only fine-tuning.

---

## Directory Structure

```
experiments/
├── day5_film_finetuning/     # Current
│   ├── configs/              # Training configs
│   ├── outputs/              # Checkpoints & logs
│   └── scripts/              # Helper scripts
│
└── _legacy/                   # Archived (days 1-4)
```

---

## Quick Start

```bash
# Run current experiment
cd thesis_work/experiments/day5_film_finetuning

# Check README for instructions
cat README.md
```
