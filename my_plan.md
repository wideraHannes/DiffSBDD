# DiffSBDD Master's Thesis Plan

## 🎯 Project Overview

Structure-based drug design using diffusion models to generate drug-like molecules for protein pockets.

## 📁 Key Directories

```
├── checkpoints/crossdocked_fullatom_cond.ckpt    # Pre-trained model
├── data/processed_crossdock_noH_full_temp/       # Processed dataset
├── results/                                      # Generated molecules
└── analyze_results.py                           # Quality evaluation
```

## 🚀 Quick Start Commands

### Data Processing

_This step converts raw protein-ligand data into a format the AI model can understand. Think of it like preparing ingredients before cooking - we clean the data, extract important features, and organize everything properly._

```bash
# Process CrossDock dataset (debug: 100 train, 10 test)
python process_crossdock.py --basedir data --no_H
```

### Generate Molecules

_This is where the magic happens! The AI model looks at protein pockets and creates new drug-like molecules that could potentially bind to them. It's like having a chemist design new drugs, but much faster._

```bash
# Debug mode (fast)
uv run test.py checkpoints/crossdocked_fullatom_cond.ckpt \
    --test_dir data/processed_crossdock_noH_full_temp/test/ \
    --outdir results/debug_eval \
    --sanitize --batch_size 1 --n_samples 1 --timesteps 10

# Production mode
uv run test.py checkpoints/crossdocked_fullatom_cond.ckpt \
    --test_dir data/processed_crossdock_noH_full_temp/test/ \
    --outdir results/full_eval \
    --sanitize --batch_size 50 --n_samples 100
```

### Evaluate Quality

_After generating molecules, we need to check if they're actually good! This analyzes whether the molecules are chemically valid, drug-like, and have the right properties to be potential medicines._

```bash
# Analyze generated molecules
uv run analyze_results.py results/debug_eval
```

## 🎓 Thesis Development Plan

### Phase 1: Baseline Understanding ✅

_First, you need to understand how the existing system works. Like learning to drive with an instructor before going solo - you master the basics before making improvements._

- [x] Setup DiffSBDD environment
- [x] Process CrossDock dataset
- [x] Run pre-trained model evaluation
- [x] Achieve 100% validity, good drug-likeness

### Phase 2: Model Training

_Now you'll train your own AI model from scratch. This is like teaching a student chemistry - you feed it lots of examples until it learns the patterns of good drug design._

```bash
# Train custom model
python train.py --config configs/my_model.yml
```

### Phase 3: Research Contributions

_This is where you make your unique contribution to science! You'll improve the existing methods or create entirely new approaches. Think of it as inventing a better recipe._

1. **Novel Architecture**: Modify EGNN layers
2. **Better Conditioning**: Enhanced pocket features
3. **Evaluation Metrics**: New molecular properties
4. **Target Applications**: Specific protein families

### Phase 4: Experiments & Evaluation

_The final step is proving your improvements actually work better than existing methods. Like a taste test - you compare your new recipe against the old ones to show it's superior._

```python
experiments = {
    'baseline': 'crossdocked_fullatom_cond.ckpt',
    'my_model_v1': 'configs/thesis_v1.yml',
    'my_model_v2': 'configs/thesis_v2.yml'
}
```

## 📊 Quality Benchmarks

_These numbers tell you if your AI is doing a good job. Think of them like grades on a report card - they measure different aspects of how well the model creates drug-like molecules._

### Current Results (Debug Eval)

- **Validity**: 100% (10/10 molecules) - _All molecules are chemically possible_
- **Drug-likeness (QED)**: 0.532 ± 0.126 - _Molecules look like real drugs_
- **Lipinski Compliance**: 100% - _Follow pharmaceutical "rules of thumb"_
- **Molecular Weight**: 222.5 ± 105.5 Da - _Right size for drug molecules_

### Target Improvements

_Goals for your thesis - what you want to make better than the current system._

- [ ] Increase molecular diversity - _Create more varied molecule types_
- [ ] Improve binding affinity prediction - _Better at predicting which molecules stick to proteins_
- [ ] Faster generation speed - _Make the AI work quicker_
- [ ] Novel chemical scaffolds - _Discover completely new types of drug structures_

## 🔧 Development Setup

_These are like different "difficulty settings" for your AI. Start with easy/fast settings for testing, then use harder/slower settings for final results._

### Debug Parameters

```bash
--batch_size 1 --n_samples 1 --timesteps 10    # Ultra-fast (for testing)
--batch_size 10 --n_samples 10 --timesteps 50  # Medium (for development)
--batch_size 50 --n_samples 100                # Production (for final results)
```

### Training Config Template

_This file tells the AI how to learn. Like setting study hours, difficulty level, and learning pace for a student._

```yaml
model:
  n_layers: 9 # How deep/complex the AI brain is
  lig_max_radius: 2 # How far molecules can "see" their neighbors
  pocket_max_radius: 6 # How far proteins can "see" their neighbors

training:
  n_epochs: 3000 # How many times to study the entire dataset
  batch_size: 32 # How many examples to study at once
  lr: 1e-4 # How fast the AI learns (learning rate)
```

## 🎛️ Training & Configuration Insights

_Hard-learned lessons from getting the training pipeline working! These insights will save you hours of debugging._

### Environment Setup

```bash
# Use uv for dependency management (much faster than pip)
uv sync  # Installs all dependencies
uv run python train.py --config your_config.yml  # Run with managed environment
```

### Critical Configuration Fixes

#### 1. Dataset Compatibility Issue

**Problem**: Tensor dimension mismatch (14 vs 13 features)

```yaml
# ❌ WRONG - causes tensor errors
dataset: "crossdock"  # Expects 10 atom types

# ✅ CORRECT - matches processed data
dataset: "crossdock_full"  # Has 11 atom types including 'others'
```

#### 2. PyTorch Lightning 2.x Compatibility

The codebase was written for Lightning 1.x, needed these fixes:

```python
# ❌ OLD (v1.x): validation_epoch_end(self, outputs)
# ✅ NEW (v2.x): on_validation_epoch_end(self)

# ❌ OLD: configure_gradient_clipping(self, optimizer, optimizer_idx, ...)
# ✅ NEW: configure_gradient_clipping(self, optimizer, ...)
```

#### 3. CPU vs GPU Training Setup

```yaml
# For CPU training (development/testing)
egnn_params:
  device: "cpu"
gpus: 0
batch_size: 1-2  # Keep small for CPU

# For GPU training (production)
egnn_params:
  device: "cuda"
gpus: 1-4
batch_size: 16-32  # Can be larger with GPU
```

### Playground Configuration Template

_Perfect for quick testing and development:_

```yaml
# configs/my_playground.yml
run_name: "my-experiment-playground"
dataset: "crossdock_full" # IMPORTANT: Use full dataset
datadir: "data/processed_crossdock_noH_full_temp"

# Quick testing settings
n_epochs: 2
batch_size: 1
gpus: 0 # CPU only
egnn_params:
  device: "cpu"

# Disable wandb for local testing
wandb_params:
  mode: "disabled"

# Frequent evaluation for development
eval_epochs: 1
visualize_sample_epoch: 1
```

### Training Pipeline Success Checklist

✅ **Environment**: `uv sync` completes without errors  
✅ **Data**: Check dataset keys match expectations (`names` vs `receptors`)  
✅ **Config**: Use `crossdock_full` for 11-feature data  
✅ **Dependencies**: PyTorch Lightning 2.x compatibility fixes applied  
✅ **Resources**: Batch size appropriate for CPU/GPU  
✅ **Output**: Checkpoints saved to `logs/{run_name}/checkpoints/`

### Model Size & Performance

```
Model: ConditionalDDPM
├── Parameters: 4.8M trainable, 501 non-trainable
├── Size: ~19.3 MB
├── Training Speed: ~2.2 it/s on CPU, ~20+ it/s on GPU
└── Memory: ~1-2GB RAM for CPU training
```

### Checkpoint Management

Automatic checkpointing creates:

```
logs/{run_name}/checkpoints/
├── last.ckpt                    # Most recent state
├── best-model-epoch=XX.ckpt     # Best validation loss
└── last-v{N}.ckpt               # Version backups
```

### Common Debugging Commands

```bash
# Quick data inspection
python -c "
import torch
data = torch.load('data/processed_crossdock_noH_full_temp/train.npz', allow_pickle=True)
print('Keys:', list(data.keys()))
print('Shapes:', {k: v.shape for k,v in data.items() if hasattr(v, 'shape')})
"

# Check model architecture
uv run python -c "
from lightning_modules import LigandPocketDDPM
model = LigandPocketDDPM(dataset='crossdock_full', ...)
print(model)
"

# Monitor training progress
tail -f logs/{run_name}/train.log
```

### Performance Optimization Tips

1. **Start Small**: Use playground config first
2. **GPU Memory**: Reduce batch_size if CUDA OOM errors
3. **CPU Training**: Expect ~10x slower than GPU
4. **Data Loading**: Set `num_workers=0` for debugging
5. **Validation**: Disable expensive metrics during development

### Troubleshooting Guide

| Error                                | Solution                                 |
| ------------------------------------ | ---------------------------------------- |
| `KeyError: 'receptors'`              | Use `crossdock_full` dataset config      |
| `RuntimeError: tensor size mismatch` | Check atom feature dimensions (10 vs 11) |
| `TypeError: missing optimizer_idx`   | Update Lightning methods for v2.x        |
| `CUDA OOM`                           | Reduce batch_size or use CPU             |
| `ModuleNotFoundError`                | Run `uv sync` first                      |

## 📝 Next Actions

_Your roadmap to thesis success! These are concrete steps organized by when you need to do them._

1. **Immediate** (this week):

   - Analyze current baseline performance - _Understand what you have_
   - Review literature for improvements - _See what others have tried_
   - Design novel model architecture - _Plan your innovation_

2. **Short-term** (next month):

   - Implement model modifications - _Code your improvements_
   - Setup training pipeline - _Prepare the learning system_
   - Design evaluation metrics - _Decide how to measure success_

3. **Long-term** (thesis timeline):
   - Conduct experiments - _Test your ideas thoroughly_
   - Write thesis chapters - _Document your discoveries_
   - Prepare defense - _Get ready to present your work_

## 🎯 Success Metrics

_How you'll know if your thesis is successful - like having clear goals for a project._

- **Technical**: >90% validity, >0.6 QED, novel molecules - _Your AI must work really well_
- **Academic**: 3-5 key contributions, publishable results - _Your research must advance the field_
- **Timeline**: Complete by thesis deadline - _You must finish on time_

---

_This is your master's thesis command center. Update progress and results here._
