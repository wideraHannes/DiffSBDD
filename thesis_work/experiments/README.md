# Experiments

**All experimental results organized by day/phase**

---

## 📁 Directory Structure

Each experiment follows this structure:

```
dayX_experiment_name/
├── README.md           # Experiment description and results
├── config.yml          # Configuration used
├── outputs/            # Raw outputs (logs, checkpoints, etc.)
├── analysis/           # Analysis notebooks/scripts
└── figures/            # Generated plots
```

---

## 🧪 Phase 0 Experiments

| Day | Experiment | Status | Directory |
|-----|------------|--------|-----------|
| 1 | ESM-C Setup & Integration | ✅ Complete | Tests in `esmc_integration/tests/` |
| 2 | Embedding Signal Analysis | 🔄 In Progress | `day2_embeddings/` |
| 3 | Overfit Test (1 sample) | ⏳ Pending | `day3_overfit/` |
| 4 | Small Dataset (100 samples) | ⏳ Pending | `day4_small_dataset/` |
| 5 | Medium Dataset (1000 samples) | ⏳ Pending | `day5_medium_dataset/` |
| 6 | Gradient & FiLM Analysis | ⏳ Pending | `day6_analysis/` |
| 7 | Go/No-Go Decision | ⏳ Pending | `day7_decision/` |

---

## 📊 Results Summary

**Day 1**: Integration tests passed ✅
- FiLM network active (Δloss = 44.15)
- All 8 dynamics() calls updated
- Backward compatibility maintained

**Day 2**: TBD

---

## 🔬 Creating New Experiments

```bash
# Create experiment directory
mkdir -p thesis_work/experiments/dayX_name/{outputs,analysis,figures}

# Document it
echo "# Day X: Experiment Name" > thesis_work/experiments/dayX_name/README.md
```

---

**See**: [Daily Logs](../daily_logs/) for detailed progress
