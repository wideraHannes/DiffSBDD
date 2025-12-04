# Experiment Configurations

**YAML configs for all experiments**

---

## 📁 Configuration Files

| Config | Experiment | Status |
|--------|------------|--------|
| `day3_overfit.yml` | 1-sample overfit test | ⏳ To be created |
| `day4_small_dataset.yml` | 100-sample training | ⏳ To be created |
| `day5_medium_dataset.yml` | 1000-sample training | ⏳ To be created |
| `full_training.yml` | Full dataset training | ⏳ To be created |

---

## 🔧 Configuration Template

```yaml
# Experiment name
run_name: "day3_overfit_esmc"

# Dataset
dataset: "crossdock_full"
datadir: "data/processed_crossdock_noH_full_temp"

# ESM-C configuration
esmc_config:
  enabled: true
  embeddings_path: "esmc_integration/embeddings_cache/train_esmc.npz"

# Training parameters
n_epochs: 50
batch_size: 1  # For overfit test
lr: 1.0e-3

# Model architecture
egnn_params:
  device: "cuda"
  joint_nf: 128
  hidden_nf: 256
  n_layers: 6
```

---

## 📝 Creating New Configs

```bash
# Copy template
cp configs/crossdock_fullatom_cond.yml thesis_work/configs/dayX_experiment.yml

# Edit for your experiment
vim thesis_work/configs/dayX_experiment.yml

# Run experiment
python train.py --config thesis_work/configs/dayX_experiment.yml
```

---

**See**: [Implementation Plan](../documentation/implementation_plan.md) for config details
