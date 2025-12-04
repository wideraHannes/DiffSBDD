# ESM-C Enhanced DiffSBDD - Master's Thesis

> **🎓 Quick Start Guide for Your Thesis Work**
>
> **READ THIS FIRST** when starting each session!

---

## 🚀 Starting a New Session?

### 1. Check Where You Left Off

```bash
# Read yesterday's work
cat thesis_work/daily_logs/INDEX.md

# Check current progress
cat thesis_work/documentation/progress_tracker.md
```

### 2. See Today's Tasks

Open the latest daily log in `thesis_work/daily_logs/` to see what's next.

**Current**: Day 2 - Embedding Analysis

---

## 📁 Project Structure

```
DiffSBDD/
│
├── thesis_work/                    # 👈 START HERE - Your main workspace
│   ├── daily_logs/                 # What happened each day
│   │   ├── INDEX.md               # Quick overview
│   │   ├── 2024-12-03_day1.md    # Day 1 summary
│   │   └── 2024-12-04_day2.md    # Today
│   │
│   ├── documentation/              # Plans and tracking
│   │   ├── progress_tracker.md   # Current status dashboard
│   │   ├── implementation_plan.md # Full Day 1-7 plan
│   │   └── session_summaries/    # Detailed notes
│   │
│   ├── experiments/                # Experimental results
│   │   ├── day1_setup/
│   │   ├── day2_embeddings/      # Current
│   │   └── ...
│   │
│   ├── analysis/                   # Analysis scripts/notebooks
│   ├── configs/                    # Experiment configs
│   └── results/                    # Final results
│
├── esmc_integration/               # ESM-C technical code
│   ├── extraction/                 # Embedding extraction
│   │   └── extract_esmc_embeddings.py
│   ├── tests/                      # All test files
│   ├── analysis/                   # ESM-C analysis tools
│   └── embeddings_cache/           # Cached .npz files
│
├── data/                           # Datasets
│   └── processed_crossdock_noH_full_temp/
│
├── equivariant_diffusion/          # Core DiffSBDD code (modified)
├── configs/                        # Original DiffSBDD configs
└── [other DiffSBDD files...]
```

---

## 📖 Key Documents

### Must Read First

1. **[Daily Log Index](thesis_work/daily_logs/INDEX.md)** - What you did each day
2. **[Progress Tracker](thesis_work/documentation/progress_tracker.md)** - Where you are now
3. **[Implementation Plan](thesis_work/documentation/implementation_plan.md)** - Full Days 1-7 plan

### Technical Documentation

- **[ESM-C Integration README](esmc_integration/README.md)** - Technical details
- **[Original Roadmap](roadmap.md)** - High-level thesis plan

---

## 🎯 Current Status (Day 2)

**Phase**: Phase 0 - Scientific Validation
**Progress**: 10/83 tasks (12%)
**Last Completed**: Day 1 (ESM-C Setup + Integration) ✅
**Current Focus**: Day 2 (Embedding Analysis) 🔄

### Today's Tasks

- [ ] Extract ESM-C embeddings for 100 test pockets
- [ ] Compute embedding similarity matrix
- [ ] Analyze correlation with binding affinity
- [ ] Visualize with t-SNE/UMAP
- [ ] Calculate mutual information

---

## 🔧 Quick Commands

### Extract Embeddings

```bash
python esmc_integration/extraction/extract_esmc_embeddings.py \
    --split test \
    --data_dir data/processed_crossdock_noH_full_temp \
    --max_samples 100 \
    --output_dir esmc_integration/embeddings_cache
```

### Run Tests

```bash
# Test FiLM integration
python esmc_integration/tests/test_esmc_integration.py

# Test full pipeline
python esmc_integration/tests/test_full_pipeline.py
```

### Check Dataset

```bash
# View dataset info
python -c "import numpy as np; data = np.load('data/processed_crossdock_noH_full_temp/train.npz'); print(data.files)"
```

---

## 📝 End of Session Checklist

Before you finish today:

- [ ] Update today's daily log: `thesis_work/daily_logs/2024-12-04_day2.md`
- [ ] Update progress tracker: `thesis_work/documentation/progress_tracker.md`
- [ ] Save any results to: `thesis_work/experiments/day2_embeddings/`
- [ ] Commit changes: `git add . && git commit -m "Day 2: [summary]"`

---

## 🆘 Need Help?

### Finding Things

- **"What did I do on Day X?"** → `thesis_work/daily_logs/`
- **"What's the full plan?"** → `thesis_work/documentation/implementation_plan.md`
- **"Where are my results?"** → `thesis_work/experiments/`
- **"How does ESM-C work?"** → `esmc_integration/README.md`

### Common Issues

**Can't find PDB files?**
```bash
find data/processed_crossdock_noH_full_temp -name "*.pdb" | head -5
```

**ESM-C API not working?**
```bash
# Check token
cat .env | grep ESM

# Test API
python -c "import esm; print('ESM installed:', esm.__version__)"
```

---

## 🎓 Thesis Phases Overview

```
┌─────────────────────────────────────────────────────────┐
│ Phase 0: Scientific Validation (Days 1-7)   [Current]  │
│ ├─ Day 1: Setup ✅                                      │
│ ├─ Day 2: Embedding Analysis 🔄                         │
│ ├─ Day 3-5: Training Validation                         │
│ └─ Day 6-7: Analysis & Decision                         │
├─────────────────────────────────────────────────────────┤
│ Phase 1: Full Training (Weeks 2-4)          [Pending]  │
└─────────────────────────────────────────────────────────┘
```

---

## 🔗 Quick Navigation

| Go To | Path |
|-------|------|
| 📅 Daily Logs | [thesis_work/daily_logs/INDEX.md](thesis_work/daily_logs/INDEX.md) |
| 📊 Progress | [thesis_work/documentation/progress_tracker.md](thesis_work/documentation/progress_tracker.md) |
| 📋 Plan | [thesis_work/documentation/implementation_plan.md](thesis_work/documentation/implementation_plan.md) |
| 🧪 Experiments | [thesis_work/experiments/](thesis_work/experiments/) |
| 💻 Code | [esmc_integration/](esmc_integration/) |

---

**Last Updated**: 2024-12-04
**Next Session**: Continue Day 2 - Extract embeddings and analyze signal

---

## 💡 Pro Tips

1. **Always read the daily log first** - Save yourself time figuring out what you did
2. **Update logs at the end of each session** - Future you will thank you
3. **One day = one focused task** - Don't try to do everything at once
4. **Document failures too** - Negative results are valuable
5. **Commit often** - Your thesis depends on this work

---

**🎯 Focus for Today**: Extract 100 embeddings and validate signal quality
