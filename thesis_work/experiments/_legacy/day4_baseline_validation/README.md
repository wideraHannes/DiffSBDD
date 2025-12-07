# Day 4: Baseline Model Validation

## Goal

Validate that the **original DiffSBDD model** trains correctly on the full dataset before adding ESM-C modifications. This follows the Google Research Tuning Playbook: verify baseline works first.

## Why This Is Needed

Day 3 overfit test showed:

- Train loss stuck at ~0.5 (oscillating 0.3-0.5)
- 0% connectivity despite 100% validity
- Molecules fragmented (atoms too far apart)

**Hypothesis**: The overfit test (1-5 samples, 500 diffusion steps) is too hard. We need to see normal training dynamics first.

## Quick Start

```bash
# Run baseline validation (~2 hours on CPU)
uv run python train.py --config thesis_work/experiments/day4_baseline_validation/configs/baseline_validation.yml
```

## Expected Training Dynamics

| Epoch | Train Loss | Val Loss | Notes       |
| ----- | ---------- | -------- | ----------- |
| 1     | 2.0-3.0    | 2.5-4.0  | Random init |
| 10    | 1.0-1.5    | 1.5-2.0  | Learning    |
| 30    | 0.5-0.8    | 0.8-1.2  | Converging  |
| 50    | 0.4-0.6    | 0.6-0.9  | End of run  |

**Key**: Loss should **decrease steadily**, not oscillate wildly.

## Config Differences from Day 3

| Setting        | Day 3 Overfit | Day 4 Baseline        |
| -------------- | ------------- | --------------------- |
| Dataset        | 5 samples     | Full dataset (~100k)  |
| batch_size     | 1             | 4                     |
| n_epochs       | 1000          | 50                    |
| n_eval_samples | 1             | 2                     |
| Purpose        | Overfit test  | Verify training works |

## Success Criteria

✅ **Pass** if:

- Train loss decreases from ~2.0 to ~0.5 over 50 epochs
- Val loss follows similar trend (with gap)
- No NaN losses or crashes
- Some connectivity (>0%) by epoch 50

❌ **Fail** if:

- Loss oscillates without decreasing
- Loss explodes or goes NaN
- 0% connectivity after 50 epochs on full data

## Next Steps

**If baseline works**: Return to overfit test with insights about learning dynamics

**If baseline fails**: Debug the model/data before any thesis modifications

## Files

```
day4_baseline_validation/
├── README.md              # This file
├── configs/
│   └── baseline_validation.yml
└── outputs/               # Training logs (created during training)
```
