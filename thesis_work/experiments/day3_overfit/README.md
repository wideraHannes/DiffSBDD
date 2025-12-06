# Day 3: Overfit Test Experiment

## Status: WORKING - Loss decreasing confirmed

## Quick Start (Recommended: 5-sample with wandb)

```bash
# Create dataset (5 train, 2 val samples)
uv run python thesis_work/experiments/day3_overfit/create_overfit_dataset.py --n_train 5 --n_val 2

# Run training with wandb logging
uv run python train.py --config thesis_work/experiments/day3_overfit/configs/day3_overfit_5sample.yml

# View loss curves
# https://wandb.ai/johannes-widera-heinrich-heine-university-d-sseldorf/ligand-pocket-ddpm
```

## Available Configs

| Config | Description | Command |
|--------|-------------|---------|
| `day3_overfit_5sample.yml` | **Recommended** - 5 samples, wandb, batch_size=5 | `uv run python train.py --config thesis_work/experiments/day3_overfit/configs/day3_overfit_5sample.yml` |
| `day3_baseline_1sample_no_esmc.yml` | 1 sample, no ESMC, baseline test | `uv run python train.py --config thesis_work/experiments/day3_overfit/configs/day3_baseline_1sample_no_esmc.yml` |
| `day3_aggressive_overfit.yml` | 1 sample, smaller model, lr=1e-2 | `uv run python train.py --config thesis_work/experiments/day3_overfit/configs/day3_aggressive_overfit.yml` |

## Key Metrics to Watch (in wandb)

| Metric | What it means | Expected for overfit |
|--------|---------------|---------------------|
| **`loss/train`** | Total diffusion loss | Should decrease toward 0 |
| `error_t_lig/train` | Ligand denoising error | Should decrease |
| `loss/val` | Validation loss | Higher (different molecules) |

## Dataset Creation

```bash
# Create custom dataset
uv run python thesis_work/experiments/day3_overfit/create_overfit_dataset.py \
  --n_train 5 \
  --n_val 2 \
  --output_dir thesis_work/experiments/day3_overfit/data_5sample

# Options:
#   --n_train N      Number of training samples (default: 1)
#   --n_val N        Number of validation samples (default: 1)
#   --train_start N  Starting index for training (default: 0)
#   --val_start N    Starting index for val (default: after train)
#   --source         Source split: train/val/test (default: test)
#   --esmc_dir       ESMC dir or "none" to skip
```

## Data Verified
- 1-sample: 31 ligand atoms, 369 pocket atoms
- 5-sample: 130 ligand atoms, 1705 pocket atoms
- Format matches original preprocessing (float64 masks)

## Results So Far

**aggressive-overfit (epoch 99)**:
- `loss/train: 0.72` (decreasing)
- `Connectivity/val: 0%` (still an issue)

## Files
```
day3_overfit/
├── README.md
├── create_overfit_dataset.py    # Multi-sample dataset creation
├── create_1sample_dataset.py    # Legacy 1-sample script
├── configs/
│   ├── day3_overfit_5sample.yml      # RECOMMENDED
│   ├── day3_baseline_1sample_no_esmc.yml
│   ├── day3_aggressive_overfit.yml
│   └── day3_overfit_wandb.yml
├── data_1sample/                # 1 sample dataset
├── data_5sample/                # 5 sample dataset
├── data_overfit/                # Default output for create_overfit_dataset.py
└── outputs/                     # Training logs
```
