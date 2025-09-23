# 🗺️ ESM-C Enhanced DiffSBDD: Master's Thesis Roadmap

> **Goal**: Investigate the influence of ESM-C embeddings on conditional ligand generation

## 📋 Project Summary

You will implement and compare **three models** to understand how ESM-C protein embeddings can enhance structure-based drug design:

1. **Baseline**: Original DiffSBDD (residue graph only)
2. **Combined Signal**: DiffSBDD + ESM-C embeddings (inspired by SurfDock)
3. **Pure Embedding Signal**: DiffSBDD conditioned only on ESM-C embeddings

---

## 🚀 Phase 1: Foundation & Baseline [Weeks 1-3]

### ✅ **Already Completed**

- [x] Environment setup with `uv`
- [x] DiffSBDD training pipeline working
- [x] Dataset processing (CrossDock)
- [x] Baseline model training verified

### 🎯 **Week 1: Establish Baseline Performance**

#### 1.1 Train Baseline Model

```bash
# Create baseline configuration
cp configs/crossdock_fullatom_cond.yml configs/baseline_model.yml

# Modify for your baseline experiment
# configs/baseline_model.yml
run_name: "baseline-original-diffsbdd"
n_epochs: 100  # Increase for proper training
batch_size: 16  # Use GPU if available
gpus: 1  # Adjust based on your hardware

# Train baseline model
uv run python train.py --config configs/baseline_model.yml
```

#### 1.2 Evaluate Baseline Performance

```bash
# Test baseline model
uv run python test.py logs/baseline-original-diffsbdd/checkpoints/best-model.ckpt \
    --test_dir data/processed_crossdock_noH_full_temp/test/ \
    --outdir results/baseline_eval \
    --sanitize --batch_size 50 --n_samples 100

# Analyze results
uv run python analyze_results.py results/baseline_eval
```

#### 1.3 Document Baseline Metrics

```python
# Create results tracking file: results/baseline_metrics.json
{
    "model": "baseline_original_diffsbdd",
    "validity": 0.XX,
    "uniqueness": 0.XX,
    "novelty": 0.XX,
    "qed": 0.XX,
    "sa_score": 0.XX,
    "lipinski_compliance": 0.XX,
    "molecular_weight": XX.X,
    "binding_affinity": 0.XX  # If available
}
```

---

## 🧬 Phase 2: ESM-C Integration [Weeks 4-8]

### 🔬 **Week 2: ESM-C Setup & Understanding**

#### 2.1 Install ESM-C

```bash
# Add ESM-C to your environment
pip install fair-esm
# or if using conda:
conda install -c bioconda fair-esm

# Test ESM-C installation
python -c "
import torch
from esm import pretrained
model, alphabet = pretrained.esm2_t33_650M_UR50D()
print('ESM-C installed successfully!')
"
```

#### 2.2 Extract Protein Embeddings

Create `scripts/extract_esm_embeddings.py`:

```python
#!/usr/bin/env python3
"""
Extract ESM-C embeddings for all proteins in the dataset
"""
import torch
from esm import pretrained
import numpy as np
from pathlib import Path
import pickle
from Bio import SeqIO

def extract_embeddings_for_dataset(data_dir, output_dir):
    """Extract ESM-C embeddings for all proteins"""

    # Load ESM-C model
    model, alphabet = pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model.eval()  # disables dropout for deterministic results

    # Process each protein in the dataset
    protein_sequences = load_protein_sequences(data_dir)
    embeddings = {}

    for protein_id, sequence in protein_sequences.items():
        # Prepare data for ESM-C
        data = [("protein", sequence)]
        batch_labels, batch_strs, batch_tokens = batch_converter(data)

        # Extract embeddings
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[33], return_contacts=True)

        # Store per-residue embeddings
        embeddings[protein_id] = {
            'embeddings': results["representations"][33][0, 1:-1].numpy(),  # Remove start/end tokens
            'sequence': sequence,
            'length': len(sequence)
        }

        print(f"Extracted embeddings for {protein_id}: {embeddings[protein_id]['embeddings'].shape}")

    # Save embeddings
    output_path = Path(output_dir) / "esm_embeddings.pkl"
    with open(output_path, 'wb') as f:
        pickle.dump(embeddings, f)

    print(f"Saved {len(embeddings)} protein embeddings to {output_path}")

# Run extraction
if __name__ == "__main__":
    extract_embeddings_for_dataset(
        "data/processed_crossdock_noH_full_temp/",
        "data/esm_embeddings/"
    )
```

#### 2.3 Create ESM-C Dataset Loader

Create `dataset_esm.py`:

```python
import pickle
import torch
from dataset import ProcessedLigandPocketDataset

class ESMEnhancedDataset(ProcessedLigandPocketDataset):
    """Dataset that includes ESM-C embeddings"""

    def __init__(self, npz_path, esm_embeddings_path, center=True, transform=None):
        super().__init__(npz_path, center, transform)

        # Load ESM-C embeddings
        with open(esm_embeddings_path, 'rb') as f:
            self.esm_embeddings = pickle.load(f)

    def __getitem__(self, idx):
        data = super().__getitem__(idx)

        # Add ESM-C embeddings to the data
        protein_name = data['names']  # Adjust based on your naming scheme
        if protein_name in self.esm_embeddings:
            esm_features = self.esm_embeddings[protein_name]['embeddings']
            # Align with pocket residues (this needs careful implementation)
            data['esm_embeddings'] = torch.tensor(esm_features, dtype=torch.float32)
        else:
            # Fallback to zero embeddings if not found
            data['esm_embeddings'] = torch.zeros((data['pocket_coords'].shape[0], 1280))

        return data
```

### 🔧 **Week 3: Model Architecture Modifications**

#### 3.1 Create Enhanced Lightning Module

Create `lightning_modules_esm.py`:

```python
from lightning_modules import LigandPocketDDPM
import torch.nn as nn

class ESMEnhancedLigandPocketDDPM(LigandPocketDDPM):
    """Enhanced version with ESM-C embedding support"""

    def __init__(self, esm_embedding_dim=1280, fusion_method="concat", **kwargs):
        super().__init__(**kwargs)

        self.esm_embedding_dim = esm_embedding_dim
        self.fusion_method = fusion_method

        # ESM-C embedding processing layers
        if fusion_method == "concat":
            # Concatenate ESM features with residue features
            self.esm_projection = nn.Linear(esm_embedding_dim, self.residue_nf)
            self.fusion_layer = nn.Linear(self.residue_nf * 2, self.residue_nf)

        elif fusion_method == "add":
            # Add ESM features to residue features
            self.esm_projection = nn.Linear(esm_embedding_dim, self.residue_nf)

        elif fusion_method == "attention":
            # Use attention to fuse features
            self.esm_projection = nn.Linear(esm_embedding_dim, self.residue_nf)
            self.attention = nn.MultiheadAttention(self.residue_nf, num_heads=8)

    def process_pocket_features(self, pocket_data):
        """Process pocket with ESM-C embeddings"""

        # Original pocket processing
        pocket_features = pocket_data['x']  # Original residue features
        esm_features = pocket_data.get('esm_embeddings', None)

        if esm_features is not None:
            # Project ESM embeddings to match residue feature dimension
            esm_projected = self.esm_projection(esm_features)

            if self.fusion_method == "concat":
                # Concatenate and project back
                combined = torch.cat([pocket_features, esm_projected], dim=-1)
                pocket_features = self.fusion_layer(combined)

            elif self.fusion_method == "add":
                # Element-wise addition
                pocket_features = pocket_features + esm_projected

            elif self.fusion_method == "attention":
                # Attention-based fusion
                pocket_features, _ = self.attention(
                    pocket_features.unsqueeze(0),
                    esm_projected.unsqueeze(0),
                    esm_projected.unsqueeze(0)
                )
                pocket_features = pocket_features.squeeze(0)

        return pocket_features
```

### 🎯 **Week 4: Combined Signal Model Implementation**

#### 4.1 Create Combined Signal Configuration

```yaml
# configs/combined_signal_model.yml
run_name: "combined-signal-esm-diffsbdd"
dataset: "crossdock_full"
datadir: "data/processed_crossdock_noH_full_temp"

# ESM-C specific parameters
esm_config:
  embedding_dim: 1280
  fusion_method: "concat" # or "add", "attention"
  embeddings_path: "data/esm_embeddings/esm_embeddings.pkl"

# Training parameters
n_epochs: 100
batch_size: 8 # Reduced due to additional memory usage
lr: 1.0e-3

egnn_params:
  device: "cuda" # Use GPU for larger models
  joint_nf: 128
  hidden_nf: 256
  n_layers: 6
  # ... other parameters same as baseline
```

#### 4.2 Train Combined Signal Model

```bash
# Train the combined signal model
uv run python train_esm.py --config configs/combined_signal_model.yml
```

#### 4.3 Evaluate Combined Signal Model

```bash
# Test combined signal model
uv run python test_esm.py logs/combined-signal-esm-diffsbdd/checkpoints/best-model.ckpt \
    --test_dir data/processed_crossdock_noH_full_temp/test/ \
    --outdir results/combined_signal_eval \
    --esm_embeddings data/esm_embeddings/esm_embeddings.pkl \
    --sanitize --batch_size 50 --n_samples 100

# Analyze results
uv run python analyze_results.py results/combined_signal_eval
```

---

## 🧪 Phase 3: Pure Embedding Signal Model [Weeks 9-11]

### 🔬 **Week 5: Pure ESM-C Model Architecture**

#### 5.1 Create Pure Embedding Model

```python
class PureESMLigandPocketDDPM(LigandPocketDDPM):
    """DiffSBDD conditioned only on ESM-C embeddings"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Remove traditional pocket processing
        self.use_traditional_pocket = False

        # Enhanced ESM processing for pure embedding approach
        self.esm_encoder = nn.Sequential(
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, self.residue_nf)
        )

        # Positional encoding for ESM embeddings
        self.position_encoder = nn.Linear(3, self.residue_nf)  # For 3D coordinates

    def forward(self, ligand, pocket, return_info=False):
        """Forward pass using only ESM-C embeddings"""

        # Process ESM embeddings instead of traditional pocket features
        esm_features = self.esm_encoder(pocket['esm_embeddings'])

        # Add positional information
        if 'pos' in pocket:
            pos_features = self.position_encoder(pocket['pos'])
            pocket_features = esm_features + pos_features
        else:
            pocket_features = esm_features

        # Create modified pocket dict
        pocket_esm = {
            'x': pocket_features,
            'pos': pocket['pos'],
            'mask': pocket['mask'],
            'size': pocket['size']
        }

        # Use parent class forward with modified pocket
        return super().forward(ligand, pocket_esm, return_info)
```

#### 5.2 Pure Embedding Configuration

```yaml
# configs/pure_embedding_model.yml
run_name: "pure-embedding-esm-diffsbdd"
dataset: "crossdock_full"
datadir: "data/processed_crossdock_noH_full_temp"

# Pure ESM-C parameters
esm_config:
  embedding_dim: 1280
  use_only_esm: true
  enhanced_encoder: true
  embeddings_path: "data/esm_embeddings/esm_embeddings.pkl"

# Training parameters (might need adjustment)
n_epochs: 150 # Might need more epochs
batch_size: 8
lr: 1.0e-3

# Model architecture adjustments
egnn_params:
  # Might need architecture changes for pure embedding approach
  hidden_nf: 512 # Potentially larger hidden dimensions
  n_layers: 8 # Potentially more layers
```

### 🎯 **Week 6: Train and Evaluate Pure Embedding Model**

```bash
# Train pure embedding model
uv run python train_pure_esm.py --config configs/pure_embedding_model.yml

# Evaluate pure embedding model
uv run python test_pure_esm.py logs/pure-embedding-esm-diffsbdd/checkpoints/best-model.ckpt \
    --test_dir data/processed_crossdock_noH_full_temp/test/ \
    --outdir results/pure_embedding_eval \
    --esm_embeddings data/esm_embeddings/esm_embeddings.pkl \
    --sanitize --batch_size 50 --n_samples 100

# Analyze results
uv run python analyze_results.py results/pure_embedding_eval
```

---

## 📊 Phase 4: Comprehensive Evaluation [Weeks 12-16]

### 🔍 **Week 7-8: Advanced Evaluation Metrics**

#### 7.1 Implement Comprehensive Evaluation

Create `analysis/comprehensive_evaluation.py`:

```python
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, QED, rdMolDescriptors
import matplotlib.pyplot as plt
import seaborn as sns

class ComprehensiveEvaluator:
    """Comprehensive evaluation for all three models"""

    def __init__(self):
        self.metrics = [
            'validity', 'uniqueness', 'novelty', 'diversity',
            'qed', 'sa_score', 'molecular_weight', 'logp',
            'tpsa', 'num_rings', 'num_heteroatoms',
            'lipinski_violations', 'veber_violations'
        ]

    def evaluate_all_models(self, results_paths):
        """Evaluate all three models and compare"""

        results = {}
        for model_name, path in results_paths.items():
            results[model_name] = self.evaluate_single_model(path)

        # Create comparison DataFrame
        comparison_df = pd.DataFrame(results).T

        # Generate comparison plots
        self.plot_comparison(comparison_df)

        # Statistical significance tests
        self.statistical_tests(results)

        return comparison_df

    def plot_comparison(self, df):
        """Create comprehensive comparison plots"""

        # Validity and Quality Metrics
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # Plot each metric
        metrics_to_plot = ['validity', 'qed', 'sa_score', 'molecular_weight', 'diversity', 'novelty']

        for i, metric in enumerate(metrics_to_plot):
            ax = axes[i // 3, i % 3]
            df[metric].plot(kind='bar', ax=ax)
            ax.set_title(f'{metric.title()}')
            ax.set_ylabel('Score')

        plt.tight_layout()
        plt.savefig('results/model_comparison.png', dpi=300)

        # Molecular property distributions
        self.plot_molecular_distributions(df)
```

#### 7.2 Binding Affinity Evaluation (if possible)

```python
# If you have access to docking software
def evaluate_binding_affinity():
    """Evaluate binding affinity using molecular docking"""

    # Use AutoDock Vina or similar
    # This would require additional setup
    pass
```

### 📈 **Week 9: Results Analysis & Interpretation**

#### 9.1 Create Results Summary

```python
# results/create_summary_report.py
def generate_thesis_results():
    """Generate comprehensive results for thesis"""

    results_summary = {
        "baseline": {
            "description": "Original DiffSBDD with residue graph conditioning",
            "metrics": {...},
            "strengths": [...],
            "weaknesses": [...]
        },
        "combined_signal": {
            "description": "DiffSBDD + ESM-C embeddings (SurfDock inspired)",
            "metrics": {...},
            "improvements_over_baseline": {...},
            "strengths": [...],
            "weaknesses": [...]
        },
        "pure_embedding": {
            "description": "DiffSBDD conditioned only on ESM-C embeddings",
            "metrics": {...},
            "comparison_to_others": {...},
            "feasibility_analysis": "...",
            "strengths": [...],
            "weaknesses": [...]
        },
        "key_findings": [
            "ESM-C embeddings improve/don't improve molecular diversity by X%",
            "Combined signal approach shows X% improvement in QED scores",
            "Pure embedding approach is/isn't sufficient for conditional generation",
            "Computational efficiency comparison: ..."
        ]
    }
```

---

## 📝 Phase 5: Thesis Writing [Weeks 17-20]

### ✍️ **Week 10: Thesis Structure**

#### 10.1 Thesis Outline

```markdown
# Master's Thesis Structure

## 1. Introduction

- Structure-based drug design landscape
- Limitations of current approaches
- ESM-C protein embeddings potential
- Research questions and hypotheses

## 2. Related Work

- Diffusion models in molecular generation
- DiffSBDD original contribution
- Protein language models (ESM family)
- SurfDock and protein embedding applications

## 3. Methodology

- Baseline: Original DiffSBDD
- Combined Signal: Architecture modifications
- Pure Embedding: Conditioning strategy
- Dataset and evaluation metrics

## 4. Experiments

- Training procedures
- Evaluation protocols
- Computational requirements
- Statistical analysis methods

## 5. Results

- Quantitative comparisons
- Qualitative analysis
- Ablation studies
- Computational efficiency

## 6. Discussion

- Interpretation of results
- Implications for drug design
- Limitations and future work
- Broader impact

## 7. Conclusion

- Key contributions
- Answered research questions
- Future research directions
```

### 📊 **Week 11: Figures and Tables**

#### 11.1 Key Figures to Create

1. **Architecture Comparison**: Visual comparison of all three models
2. **Training Curves**: Loss curves for all models
3. **Molecular Property Distributions**: QED, SA, MW distributions
4. **Generated Molecule Examples**: Best examples from each model
5. **Performance Comparison**: Radar charts or bar plots
6. **Computational Efficiency**: Training time and memory usage

---

## 🛠️ Essential Development Tools

### 📁 **Project Structure**

```
DiffSBDD/
├── configs/
│   ├── baseline_model.yml
│   ├── combined_signal_model.yml
│   └── pure_embedding_model.yml
├── scripts/
│   ├── extract_esm_embeddings.py
│   └── run_all_experiments.py
├── models/
│   ├── lightning_modules_esm.py
│   └── dataset_esm.py
├── analysis/
│   ├── comprehensive_evaluation.py
│   └── thesis_plots.py
├── results/
│   ├── baseline_eval/
│   ├── combined_signal_eval/
│   ├── pure_embedding_eval/
│   └── comparison_report.html
└── thesis/
    ├── chapters/
    ├── figures/
    └── main.tex
```

### 🔧 **Useful Scripts**

#### Master Experiment Runner

```bash
# scripts/run_all_experiments.py
#!/usr/bin/env python3
"""Run all thesis experiments automatically"""

import subprocess
import logging
from pathlib import Path

def run_experiment(config_name, description):
    """Run a single experiment"""
    logging.info(f"Starting {description}")

    cmd = f"uv run python train.py --config configs/{config_name}"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    if result.returncode == 0:
        logging.info(f"✅ {description} completed successfully")
    else:
        logging.error(f"❌ {description} failed: {result.stderr}")

    return result.returncode == 0

# Run all experiments
experiments = [
    ("baseline_model.yml", "Baseline DiffSBDD"),
    ("combined_signal_model.yml", "Combined Signal ESM+DiffSBDD"),
    ("pure_embedding_model.yml", "Pure ESM Embedding Model")
]

for config, desc in experiments:
    run_experiment(config, desc)
```

### 📊 **Progress Tracking**

#### Experiment Tracker

```python
# Create experiments.json to track progress
{
    "baseline": {
        "status": "completed",
        "training_time": "2.5 hours",
        "best_epoch": 85,
        "final_loss": 0.245,
        "evaluation_completed": true,
        "results_path": "results/baseline_eval/"
    },
    "combined_signal": {
        "status": "in_progress",
        "current_epoch": 45,
        "estimated_completion": "2 hours",
        "evaluation_completed": false
    },
    "pure_embedding": {
        "status": "not_started",
        "dependencies": ["ESM embeddings extraction"]
    }
}
```

---

## 🎯 Success Criteria & Milestones

### ✅ **Phase Completion Checklist**

#### Phase 1: Foundation ✅

- [x] Training pipeline working
- [ ] Baseline model trained (>95% validity)
- [ ] Baseline evaluation complete
- [ ] Performance benchmarks established

#### Phase 2: ESM-C Integration

- [ ] ESM-C embeddings extracted for dataset
- [ ] Combined signal model architecture implemented
- [ ] Combined signal model trained
- [ ] Combined signal evaluation complete

#### Phase 3: Pure Embedding Model

- [ ] Pure embedding architecture designed
- [ ] Pure embedding model trained
- [ ] Pure embedding evaluation complete
- [ ] Feasibility analysis complete

#### Phase 4: Evaluation

- [ ] Comprehensive evaluation framework implemented
- [ ] All three models compared
- [ ] Statistical significance tests performed
- [ ] Key findings documented

#### Phase 5: Thesis

- [ ] Thesis outline complete
- [ ] All chapters drafted
- [ ] Figures and tables finalized
- [ ] Thesis defense prepared

### 🎯 **Expected Outcomes**

#### Technical Contributions

1. **Novel Application**: First systematic study of ESM-C embeddings in conditional molecular generation
2. **Architecture Innovations**: New fusion methods for protein embeddings in diffusion models
3. **Empirical Insights**: Understanding of when and how protein language models help drug design

#### Academic Deliverables

1. **Master's Thesis**: Complete thesis document
2. **Publication**: Conference/journal paper (if results are significant)
3. **Code Release**: Open-source implementation of ESM-enhanced DiffSBDD

### ⚠️ **Risk Mitigation**

#### Potential Issues & Solutions

| Risk                                 | Probability | Impact | Mitigation                                        |
| ------------------------------------ | ----------- | ------ | ------------------------------------------------- |
| ESM-C integration too complex        | Medium      | High   | Start with simple concatenation approach          |
| Pure embedding model doesn't work    | High        | Medium | Focus on combined signal if pure fails            |
| Computational resources insufficient | Medium      | High   | Use smaller models, request more compute          |
| Results show no improvement          | Medium      | High   | Negative results are still valuable contributions |

---

## 🚀 Getting Started Today

### Immediate Next Steps (This Week)

1. **Verify Baseline**: Ensure your baseline model is properly trained and evaluated

```bash
# Check if baseline training completed successfully
ls logs/*/checkpoints/
uv run python test.py [your_best_checkpoint] --test_dir data/processed_crossdock_noH_full_temp/test/ --outdir results/baseline_verification
```

2. **Setup ESM-C Environment**:

```bash
pip install fair-esm
python -c "from esm import pretrained; print('ESM ready!')"
```

3. **Create Project Structure**:

```bash
mkdir -p scripts models analysis results/baseline_eval results/combined_signal_eval results/pure_embedding_eval thesis
```

4. **Start ESM Embedding Extraction**:

```bash
# Create and run the embedding extraction script
python scripts/extract_esm_embeddings.py
```

### This Month's Goals

- [ ] Complete baseline evaluation
- [ ] Extract ESM-C embeddings for entire dataset
- [ ] Implement combined signal model architecture
- [ ] Start training combined signal model

---

**Remember**: This is a research project, so be prepared to adapt the roadmap based on your findings. Some approaches might work better than expected, others might need significant modifications. Document everything and don't be afraid of negative results - they're valuable contributions too!

Good luck with your thesis! 🎓✨
