# Day 5: FiLM-Only Fine-Tuning

## Goal
Train only the FiLM adapter (~131K params) while keeping pretrained EGNN frozen (~2M params).

## Approach

```
1. Load pretrained checkpoint (strict=False)
2. Initialize FiLM to identity (γ=1, β=0)
3. Freeze all params except FiLM
4. Train on CrossDock with ESM-C embeddings
5. Evaluate binding affinity improvement
```

## Directory Structure

```
day5_film_finetuning/
├── README.md           # This file
├── configs/            # Training configs
│   └── film_finetuning.yml
├── outputs/            # Checkpoints & logs
└── scripts/            # Helper scripts
```

## Quick Commands

```bash
# Verify baseline works
uv run python generate_ligands.py checkpoints/crossdocked_fullatom_cond.ckpt \
    --pdbfile example/3rfm.pdb --outfile outputs/baseline_test.sdf \
    --ref_ligand A:330 --n_samples 5 --timesteps 100

# Train FiLM-only (after implementing changes)
uv run python train.py --config thesis_work/experiments/day5_film_finetuning/configs/film_finetuning.yml

# Generate with ESM-C (after implementing changes)
uv run python generate_ligands.py outputs/film_finetuned.ckpt \
    --pdbfile example/3rfm.pdb --outfile outputs/esmc_test.sdf \
    --ref_ligand A:330 --n_samples 20 --esmc_emb data/esmc_embeddings/3rfm.pt
```

## Implementation Checklist

- [ ] Verify pretrained checkpoint generates valid molecules
- [ ] Add `load_pretrained_with_esmc()` to lightning_modules.py
- [ ] Add `_init_film_identity()` method
- [ ] Modify `configure_optimizers()` for FiLM-only training
- [ ] Thread `pocket_emb` through inference pipeline
- [ ] Run first FiLM-only training (small subset)
- [ ] Compare baseline vs FiLM binding affinity

## Expected Results

| Metric | Baseline | FiLM + ESM-C |
|--------|----------|--------------|
| SMINA (kcal/mol) | -X.XX | -X.XX (better) |
| Validity | >90% | >90% |
| Connectivity | >80% | >80% |

## Notes

- FiLM network already exists in `dynamics.py:55-61`
- Forward pass already handles `pocket_emb` in `dynamics.py:119-131`
- Main work: checkpoint loading, optimizer config, inference pipeline
