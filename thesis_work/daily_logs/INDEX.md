# Daily Log Index

## Current Approach: FiLM-Only Fine-Tuning

**Key insight:** Train only FiLM adapter (131K params) on frozen pretrained DiffSBDD (2M params).

---

## Progress

### Phase 1: Full Training Attempts (Archived)

| Day | Focus | Outcome | Notes |
|-----|-------|---------|-------|
| Day 1 | ESM-C Setup | Done | Integration working |
| Day 2 | Embedding Analysis | Done | ESM-C embeddings look good |
| Day 3 | Overfit Test | Blocked | 0% connectivity issue |
| Day 4 | Baseline Validation | Partial | Pretrained model works |

**Conclusion:** Full training from scratch has connectivity issues. Pivot to FiLM-only fine-tuning.

---

### Phase 2: FiLM Fine-Tuning (Current)

| Day | Focus | Status | Link |
|-----|-------|--------|------|
| **Day 5** | FiLM-only training implementation | **In Progress** | [View](2024-12-07_day5.md) |
| Day 6 | Evaluation & docking | Planned | - |
| Day 7 | HPC scaling & results | Planned | - |

---

## Quick Commands

```bash
# Check pretrained baseline
uv run python generate_ligands.py checkpoints/crossdocked_fullatom_cond.ckpt \
    --pdbfile example/3rfm.pdb --outfile test.sdf --ref_ligand A:330 --n_samples 5

# View full plan
cat .claude/plans/mellow-percolating-reef.md
```

---

## Legend

- **In Progress** — Currently working on
- **Planned** — Next up
- **Done** — Completed
- **Blocked** — Has issues
