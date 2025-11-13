# ESM-C Integration - Quick Reference

**Status**: ✅ Validation Complete (6/6 tests passed)
**Next Action**: Modify `process_crossdock.py` to add residue ID tracking

---

## 📁 Key Documents

| Document | Purpose |
|----------|---------|
| `.claude/NEXT_STEPS.md` | **START HERE** - Detailed step-by-step guide |
| `.claude/ESM_C_INTEGRATION_STRATEGY.md` | Technical strategy and architecture |
| `.claude/PLAN.md` | Thesis plan and timeline |
| `experiments/VALIDATION_SUMMARY.md` | Validation results and troubleshooting |
| `experiments/README.md` | Validation scripts overview |

---

## 🚀 Quick Start

### Run Validation Tests
```bash
# Main validation suite
python experiments/validate_esmc_integration.py

# Analyze data format
python experiments/analyze_pocket_representation.py

# See code modifications needed
python experiments/example_residue_id_tracking.py
```

### Next Immediate Steps (Week 1)

**PARALLEL TASKS** - Can do both at the same time!

**Track A: Baseline Evaluation** (4-8 hours)
1. Verify baseline checkpoint exists
2. Generate molecules on test set
3. Compute all metrics (validity, QED, SA, etc.)
4. Document baseline numbers for comparison

**Track B: Data Re-processing** (8-16 hours)
1. Modify `process_crossdock.py` (lines 105-138)
   - See: `experiments/example_residue_id_tracking.py`
2. Test on single PDB
3. Re-process full dataset

**Why baseline first?**
- Establishes comparison target
- Can run while data re-processes
- Critical for proving ESM-C works!

---

## 🎯 Critical Finding

**Current NPZ files MISSING `pocket_residue_ids`**

Without this field, we cannot:
- Map atoms → residues
- Broadcast ESM-C embeddings to atoms
- Proceed with integration

**Solution**: Must re-process dataset with modified `process_crossdock.py`

---

## 📊 Architecture Summary

```
BASELINE:
  One-hot (11) → Encoder (11→22→128) → EGNN → Decoder (128→22→11)

ESM-C AUGMENTED:
  [One-hot (11) + ESM-C (960)] → Encoder (971→1942→128) → EGNN → Decoder (128→22→11)
                                                                            ↑
                                                    ESM-C NOT decoded (as intended!)
```

**Key insight**: Decoder outputs only one-hot (11-dim), NOT ESM-C (960-dim)

---

## ⏱️ Timeline

| Phase | Duration | Hands-on |
|-------|----------|----------|
| Data re-processing | Week 1 | 8-16h |
| ESM-C pre-computation | Week 1-2 | 4-8h |
| Implementation | Week 2 | 8-16h |
| Testing | Week 3 | 12h |
| Full training | Week 3-5 | (HPC) |
| Evaluation | Week 5-6 | 8-16h |

**Total**: 5-6 weeks, 40-68 hours hands-on

---

## 📝 Checklist (Current Phase)

### Phase 0A: Baseline Evaluation ← **START HERE** (PARALLEL)

- [ ] Verify baseline checkpoint
- [ ] Generate baseline molecules
- [ ] Compute metrics (validity, QED, SA, Vina)
- [ ] Create baseline report
- [ ] Document baseline numbers

### Phase 0B: Data Re-processing ← **ALSO START HERE** (PARALLEL)

- [ ] Modify `process_crossdock.py` (add residue ID tracking)
- [ ] Test on single PDB file
- [ ] Re-process train/val/test splits
- [ ] Validate re-processed data

### Phase 1: ESM-C Pre-computation (Next)

- [ ] Install fair-esm
- [ ] Write `scripts/precompute_esmc.py`
- [ ] Compute embeddings for all splits
- [ ] Validate ESM-C embeddings

---

## 🔧 Code Changes Required

**Total**: ~3 files, ~100 lines of code

1. **process_crossdock.py** (~20 lines)
   - Track residue IDs per atom

2. **dynamics.py** (~10 lines)
   - Add `esmc_dim` parameter
   - Augment encoder input

3. **dataset.py** (~10 lines)
   - Add `use_esmc` parameter
   - Load ESM-C embeddings

4. **lightning_modules.py** (~15 lines)
   - Pass ESM-C to dynamics
   - Extract from batch

5. **precompute_esmc.py** (NEW, ~150 lines)
   - Compute ESM-C embeddings
   - Broadcast to atoms

---

## 🎓 Success Criteria

### Minimum (Thesis-worthy)
- Model trains without crashes ✅
- Data processing works ✅
- Any metric improves >5%

### Good
- 2+ metrics improve (p<0.05)
- Better validity or QED
- Improved Vina scores

### Excellent
- All metrics improve
- Mean Vina shift >0.5 kcal/mol
- Novel high-affinity molecules

---

## 🆘 Troubleshooting

| Problem | Solution |
|---------|----------|
| NPZ missing `pocket_residue_ids` | Re-process with modified script |
| OOM during training | Reduce batch_size to 4 or 8 |
| NaN losses | Adjust `normalize_factors: [0.1, 1]` |
| Slow ESM-C computation | Use HPC with GPU, batch_size=32 |

See `experiments/VALIDATION_SUMMARY.md` for detailed troubleshooting.

---

## 📚 References

**Validation Evidence**:
- All tests: `experiments/validate_esmc_integration.py`
- Data analysis: `experiments/analyze_pocket_representation.py`
- Code examples: `experiments/example_residue_id_tracking.py`

**Architecture**:
- Text diagram: `experiments/architecture_comparison.txt`
- Technical details: `.claude/ESM_C_INTEGRATION_STRATEGY.md`

**Implementation Guide**:
- Step-by-step: `.claude/NEXT_STEPS.md`
- Thesis plan: `.claude/PLAN.md`

---

## 🎯 Current Actions (PARALLEL)

**Track A: Baseline Evaluation** (Start FIRST)

1. Check for baseline checkpoint: `checkpoints/crossdocked_fullatom_cond.ckpt`
2. Create `scripts/generate_baseline_test_set.py`
3. Run generation on test set
4. Create `scripts/evaluate_baseline.py`
5. Document baseline metrics

**Estimated time**: 4-8 hours

**Track B: Data Re-processing** (Can run in parallel)

1. Read `experiments/example_residue_id_tracking.py`
2. Open `process_crossdock.py`
3. Apply changes to lines 105-138
4. Test on single PDB
5. Re-process full dataset

**Estimated time**: 2-4 hours coding + 8-16 hours re-processing

---

**Why this order?**
- Baseline establishes target to beat
- Can run while waiting for data re-processing
- Provides early results for thesis

**Questions?** See `.claude/NEXT_STEPS.md` for detailed instructions on every step.
