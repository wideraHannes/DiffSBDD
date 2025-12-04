# Analysis Scripts and Notebooks

**Analysis tools for ESM-C embeddings and model evaluation**

---

## 📁 Structure

```
analysis/
├── day2_signal_analysis/       # Day 2: Embedding quality analysis
├── embedding_visualization/    # t-SNE, UMAP, etc.
├── film_analysis/              # FiLM parameter analysis
├── gradient_analysis/          # Gradient flow studies
└── utils/                      # Shared analysis utilities
```

---

## 📊 Analysis Types

### 1. Embedding Analysis (Day 2)

**Location**: `day2_signal_analysis/`

Scripts for analyzing ESM-C embedding quality:
- Similarity matrices
- Correlation with binding affinity
- t-SNE/UMAP visualization
- Mutual information calculation

### 2. Training Analysis (Days 3-5)

- Loss curves
- Validation metrics
- Overfitting checks

### 3. Model Analysis (Day 6)

- Gradient flow
- FiLM parameter distributions
- Ablation studies

---

## 🧮 Common Analysis Tasks

### Compute Embedding Similarity

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load embeddings
data = np.load('esmc_integration/embeddings_cache/test_esmc.npz')
embeddings = data['embeddings']

# Compute similarity
similarity = cosine_similarity(embeddings)
```

### Visualize with t-SNE

```python
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Reduce dimensions
tsne = TSNE(n_components=2, random_state=42)
coords = tsne.fit_transform(embeddings)

# Plot
plt.scatter(coords[:, 0], coords[:, 1])
plt.savefig('thesis_work/experiments/day2_embeddings/figures/tsne.png')
```

---

## 📓 Jupyter Notebooks

Create notebooks for interactive analysis:

```bash
# Start Jupyter in analysis directory
cd thesis_work/analysis
jupyter notebook
```

---

**See**: [Experiments](../experiments/) for results
