# ESM-C Development Directory

Organized workspace for ESM-C integration into DiffSBDD.

## Directory Structure

```
esmc_dev/
├── phase0_infrastructure/     # Day 1.1: ESM-C setup & testing
├── phase0_code_integration/   # Day 1.2: Code modifications
├── phase0_tests/              # Integration tests
├── embeddings_cache/          # Cached ESM-C embeddings
├── configs/                   # ESM-C experiment configs
└── notebooks/                 # Analysis notebooks
```

## Progress Tracking

- [x] Phase 0 Day 1.1: ESM-C Infrastructure ✅
- [ ] Phase 0 Day 1.2: Code Integration (in progress)
- [ ] Phase 0 Day 2: Embedding Analysis
- [ ] Phase 0 Day 3: Overfit Test
- [ ] Phase 0 Day 4-7: Validation experiments

## Quick Commands

```bash
# Extract embeddings for test set (small sample)
python esmc_dev/phase0_infrastructure/extract_esmc_embeddings.py --split test --max_samples 10

# Run integration tests
python esmc_dev/phase0_tests/test_integration.py

# Launch training with ESM-C
python train.py --config esmc_dev/configs/esmc_test.yml
```
