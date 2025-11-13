# .claude Directory - Documentation Index

This directory contains comprehensive documentation for ESM-C integration into DiffSBDD.

---

## 📖 Documentation Structure

### 🎯 **START HERE**

1. **`QUICK_REFERENCE.md`** ← Read this first!
   - Quick overview and current status
   - What to do right now
   - Key links and checklists

2. **`BASELINE_EVALUATION_GUIDE.md`** ← Do this FIRST!
   - How to establish baseline performance
   - Critical for proving ESM-C works
   - **Can run in parallel** with data re-processing
   - 4-8 hours total

3. **`NEXT_STEPS.md`** ← Detailed action plan
   - Step-by-step implementation guide
   - Code examples for every change
   - Timeline and checklist
   - **18 detailed steps** from baseline to evaluation

### 📋 Technical Documentation

4. **`ESM_C_INTEGRATION_STRATEGY.md`**
   - Complete technical strategy
   - Architecture design decisions
   - Data flow and preprocessing
   - Validation results (6/6 tests passed)
   - ~600 lines of detailed analysis

4. **`PLAN.md`**
   - Thesis plan and timeline
   - Phase breakdown (0-5)
   - Success criteria
   - Risk mitigation strategies
   - ~740 lines

5. **`ARCHITECTURE_DIAGRAMS.md`**
   - Codebase architecture overview
   - Module descriptions
   - Data flow diagrams
   - API reference

6. **`ARCHITECTURE_SUMMARY.txt`**
   - Condensed architecture notes
   - Quick reference for key components

---

## 🧪 Validation & Experiments

See `../experiments/` directory:

- **`validate_esmc_integration.py`** - Main validation suite (6 tests)
- **`analyze_pocket_representation.py`** - Data format deep dive
- **`example_residue_id_tracking.py`** - Code modification demo
- **`VALIDATION_SUMMARY.md`** - Complete validation report
- **`architecture_comparison.txt`** - Visual architecture comparison
- **`README.md`** - Experiments overview

---

## 📂 File Guide

| File | Purpose | Lines | Status |
|------|---------|-------|--------|
| `QUICK_REFERENCE.md` | Quick start guide | ~230 | ✅ Current |
| `BASELINE_EVALUATION_GUIDE.md` | Baseline evaluation how-to | ~300 | ✅ Current |
| `NEXT_STEPS.md` | Detailed action plan | ~1900 | ✅ Current |
| `ESM_C_INTEGRATION_STRATEGY.md` | Technical strategy | ~600 | ✅ Validated |
| `PLAN.md` | Thesis plan | ~740 | ✅ Current |
| `ARCHITECTURE_DIAGRAMS.md` | Codebase overview | ~500 | ✅ Reference |
| `ARCHITECTURE_SUMMARY.txt` | Quick architecture notes | ~100 | ✅ Reference |

---

## 🗺️ Reading Path by Role

### If you want to **implement right now**:
1. `QUICK_REFERENCE.md` (5 min read)
2. `BASELINE_EVALUATION_GUIDE.md` (10 min read)
3. `NEXT_STEPS.md` → Step 0.1 (verify baseline checkpoint)
4. `NEXT_STEPS.md` → Step 1 (modify process_crossdock.py, in parallel)

### If you want to **understand the strategy**:
1. `ESM_C_INTEGRATION_STRATEGY.md` (30 min read)
2. `experiments/VALIDATION_SUMMARY.md` (15 min read)
3. `experiments/architecture_comparison.txt` (10 min read)

### If you want to **plan the thesis**:
1. `PLAN.md` (20 min read)
2. `NEXT_STEPS.md` → Timeline Summary (5 min)
3. `QUICK_REFERENCE.md` → Success Criteria

### If you want to **understand the codebase**:
1. `ARCHITECTURE_DIAGRAMS.md` (full read)
2. `ARCHITECTURE_SUMMARY.txt` (quick reference)
3. `../CLAUDE.md` (project-level instructions)

---

## 🎯 Current Project Status

### ✅ Completed
- [x] Feasibility validation (6/6 tests passed)
- [x] Strategy documented
- [x] Code modifications identified
- [x] Timeline planned
- [x] Validation scripts created
- [x] Baseline evaluation guide created

### 🔄 In Progress
- [ ] **Phase 0A**: Baseline evaluation (← START HERE)
- [ ] **Phase 0B**: Data re-processing (← START HERE, PARALLEL)

### ⏳ Upcoming
- [ ] Phase 1: ESM-C pre-computation
- [ ] Phase 2: Model implementation
- [ ] Phase 3: Testing
- [ ] Phase 4: Full training
- [ ] Phase 5: Evaluation

---

## 🔑 Key Findings

### ✅ What Works
1. **Augmentation strategy** - Concat [one-hot + ESM-C] at encoder
2. **Decoder unchanged** - Outputs only one-hot (11-dim)
3. **Broadcasting** - Simple indexing from residues to atoms
4. **Storage** - Only ~76 GB (~0.74% of quota)
5. **Backward compatible** - Set `esmc_dim=0` reverts to baseline

### 🚨 Critical Blocker
- **Missing `pocket_residue_ids`** in current NPZ files
- Must re-process dataset before proceeding
- See `NEXT_STEPS.md` Step 1-4

---

## 📊 Validation Results Summary

All 6 tests **PASSED** ✅

1. ✅ Data structure inspection - Full-atom, 11-dim encoding
2. ✅ Broadcasting strategy - Works perfectly
3. ✅ Encoder/decoder dimensions - Compatible
4. ✅ Mock integration - No errors
5. ✅ Conditional mode - Decoder unused
6. ✅ Storage requirements - Acceptable

**Conclusion**: Integration is **FEASIBLE** and **RECOMMENDED**

---

## 🛠️ Quick Commands

```bash
# Validate integration
python experiments/validate_esmc_integration.py

# Analyze data format
python experiments/analyze_pocket_representation.py

# See code changes
python experiments/example_residue_id_tracking.py

# View architecture comparison
cat experiments/architecture_comparison.txt
```

---

## 📚 External References

- **Paper**: DiffSBDD (Nature Computational Science, 2024)
- **Model**: ESM-C / ESM-2 (Meta AI)
- **Dataset**: CrossDocked (~100K protein-ligand pairs)
- **Framework**: PyTorch Lightning

---

## 🆘 Getting Help

**For implementation questions**:
- See `NEXT_STEPS.md` - step-by-step guide
- See `experiments/example_residue_id_tracking.py` - code examples

**For technical questions**:
- See `ESM_C_INTEGRATION_STRATEGY.md` - full technical details
- See `experiments/VALIDATION_SUMMARY.md` - troubleshooting

**For planning questions**:
- See `PLAN.md` - thesis timeline
- See `QUICK_REFERENCE.md` - success criteria

**For codebase questions**:
- See `ARCHITECTURE_DIAGRAMS.md` - full architecture
- See `../CLAUDE.md` - project instructions

---

## 📈 Timeline at a Glance

| Week | Phase | What | Hours |
|------|-------|------|-------|
| 1 | Data | Re-process dataset | 8-16 |
| 1-2 | ESM-C | Pre-compute embeddings | 4-8 |
| 2 | Code | Implement augmentation | 8-16 |
| 3 | Test | Debug + small-scale | 12 |
| 3-5 | Train | Full training (HPC) | - |
| 5-6 | Eval | Metrics + comparison | 8-16 |

**Total**: 5-6 weeks, 40-68 hours hands-on

---

## 🎯 Next Action

**Right now**: Read `QUICK_REFERENCE.md` (5 minutes)

**Then**: Follow `NEXT_STEPS.md` Step 1 - Modify `process_crossdock.py`

**Estimated time to first milestone**: 2-4 hours (coding) + 8-16 hours (re-processing)

---

**Last Updated**: 2025-11-13
**Status**: Ready to proceed with Phase 0 (Data Re-processing)
