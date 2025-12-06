# In-Depth Explanation: How ESM-C Conditioning Works in DiffSBDD

> **Goal**: Add evolutionary and functional pocket context to ligand generation via FiLM conditioning

---

## Table of Contents

1. [Overview](#overview)
2. [Current DiffSBDD (Baseline)](#current-diffsbdd-baseline)
3. [The Problem](#the-problem)
4. [Proposed Solution: ESM-C + FiLM](#proposed-solution-esm-c--film)
5. [Step-by-Step Data Flow](#step-by-step-data-flow)
6. [How FiLM Steering Works](#how-film-steering-works)
7. [Visual Architecture](#visual-architecture)
8. [Concrete Example](#concrete-example)
9. [Training Dynamics](#training-dynamics)
10. [Why This is "Conditioning"](#why-this-is-conditioning)

---

## Overview

**What we're doing**: Adding a **single global 960-dimensional ESM-C embedding** per pocket that "steers" ligand generation via **FiLM (Feature-wise Linear Modulation)**.

**Where it happens**: In the denoising network (`EGNNDynamics`), right after encoding ligand features and before EGNN message passing.

**Effect**: Ligand features are scaled and shifted based on evolutionary/functional pocket context, guiding generation toward more realistic binders.

---

## Current DiffSBDD (Baseline)

### What Information is Used?

**Ligand Representation:**
- Coordinates: `(N_lig, 3)` - 3D positions
- Atom types: `(N_lig, atom_types)` - One-hot (C, N, O, S, ...)

**Pocket Representation:**
- Coordinates: `(N_pocket, 3)` - 3D positions
- Residue types: `(N_pocket, 20)` - One-hot amino acids

**Example one-hot encoding:**
```python
# ALA (Alanine)
[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

# VAL (Valine)
[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

# LEU (Leucine)
[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
```

### Current Forward Pass

**File**: `equivariant_diffusion/dynamics.py:87-186`

```python
def forward(self, xh_ligand_t, xh_pocket, t, mask_ligand, mask_pocket):
    """
    xh_ligand_t: (N_lig, 3 + atom_types) - Noisy ligand at timestep t
    xh_pocket: (N_pocket, 3 + 20) - Pocket (fixed)
    t: Diffusion timestep
    """

    # 1. Extract coordinates and features
    x_ligand = xh_ligand_t[:, :3]        # (N_lig, 3)
    h_ligand = xh_ligand_t[:, 3:]        # (N_lig, atom_types)

    x_pocket = xh_pocket[:, :3]          # (N_pocket, 3)
    h_pocket = xh_pocket[:, 3:]          # (N_pocket, 20)

    # 2. Encode to joint embedding space
    h_ligand = self.atom_encoder(h_ligand)      # (N_lig, 128)
    h_pocket = self.residue_encoder(h_pocket)   # (N_pocket, 128)

    # 3. Create unified graph
    x = torch.cat([x_ligand, x_pocket], dim=0)  # All coordinates
    h = torch.cat([h_ligand, h_pocket], dim=0)  # All features
    mask = torch.cat([mask_ligand, mask_pocket])

    # 4. Add timestep embedding
    h_time = t[mask]  # Broadcast timestep to all nodes
    h = torch.cat([h, h_time], dim=1)  # (N_total, 129)

    # 5. EGNN message passing
    #    Ligand atoms exchange messages with pocket atoms
    h_final, x_final = self.egnn(h, x, edges, mask=mask)

    # 6. Decode to predict noise
    h_ligand_final = h_final[:N_lig]
    h_pocket_final = h_final[N_lig:]

    epsilon_ligand = self.atom_decoder(h_ligand_final)
    epsilon_pocket = self.residue_decoder(h_pocket_final)

    return epsilon_ligand, epsilon_pocket
```

---

## The Problem

**What's missing?**

One-hot amino acid encoding captures only:
- ✅ Chemical identity (which amino acid)
- ❌ **Evolutionary conservation** (is this residue critical across species?)
- ❌ **Structural context** (what's the fold, domain, motif?)
- ❌ **Functional role** (is this an active site, binding site?)
- ❌ **Physicochemical environment** (hydrophobic pocket? Charged?)

**Example**: Both of these are "Leucine (LEU)":
```python
LEU in conserved hydrophobic binding pocket: [0,0,1,0,...]
LEU on protein surface (not binding):        [0,0,1,0,...]  # SAME!
```

But they have **very different binding significance**!

---

## Proposed Solution: ESM-C + FiLM

### What is ESM-C?

**ESM-C (Evolutionary Scale Modeling - C)** is a protein language model that:
- Trained on 500 million years of evolution
- Encodes sequences into rich embeddings
- Captures evolutionary, structural, and functional information

**Input**: Protein sequence `"MKTAYIAKQRQISFVKSHF..."`
**Output**: Per-residue embeddings `(N_residues, 960)`

Each 960-dimensional vector encodes:
- Evolutionary conservation patterns
- Structural propensities
- Functional motifs
- Physicochemical context

### What is FiLM?

**FiLM (Feature-wise Linear Modulation)** is a conditioning technique:

```
Input features: h ∈ ℝ^d
Conditioning signal: c ∈ ℝ^k

FiLM Network: c → [γ, β] where γ, β ∈ ℝ^d

Modulated features: h' = γ ⊙ h + β
```

Where:
- **γ (gamma)**: Scale parameters - amplify or suppress channels
- **β (beta)**: Shift parameters - add bias to channels
- **⊙**: Element-wise multiplication

**Used in**: StyleGAN, text-to-image models, visual reasoning

---

## Step-by-Step Data Flow

### Phase 1: Extract ESM-C Embeddings (OFFLINE)

**Script**: `scripts/extract_esmc_embeddings.py`

```python
from esm.models.esmc import ESMC
import torch
import numpy as np

# Load ESM-C model
model = ESMC.from_pretrained("esmc_600m").eval().cuda()

for pocket in dataset:
    # 1. Get FULL protein sequence (important for context!)
    full_sequence = "MKTAYIAKQRQISFVKSHFSRRNSKL..."  # All residues

    # 2. Run ESM-C on entire protein
    with torch.no_grad():
        embeddings = model.encode(full_sequence)
    # Shape: (N_total_residues, 960)

    # 3. Extract only pocket residues
    pocket_residue_ids = [45, 46, 47, 50, 51, 52, ...]  # Which residues
    pocket_embeddings = embeddings[pocket_residue_ids]
    # Shape: (N_pocket_residues, 960)

    # 4. Pool to single global vector
    global_pocket_emb = pocket_embeddings.mean(dim=0)
    # Shape: (960,)

    # 5. Save to cache
    cache[pocket_id] = global_pocket_emb.cpu().numpy()

# Save all pockets
np.savez('esmc_train.npz', **cache)
```

**Why mean pooling?**
- Simple and effective
- Captures overall pocket character
- Fixed dimensionality (no variable-length issues)

**Result**: Each pocket has a single 960-dim vector capturing its evolutionary/functional essence.

---

### Phase 2: Load During Training (ONLINE)

**File**: `dataset.py`

```python
class ProcessedLigandPocketDataset(Dataset):
    def __init__(self, data_path, esmc_path):
        self.data = np.load(data_path)  # Coordinates, one-hot
        self.esmc = np.load(esmc_path)  # ESM-C embeddings

    def __getitem__(self, idx):
        # Load as before
        sample = {
            'lig_coords': self.data['lig_coords'][idx],     # (N_lig, 3)
            'lig_one_hot': self.data['lig_one_hot'][idx],   # (N_lig, atom_types)
            'pocket_coords': self.data['pocket_coords'][idx],   # (N_pocket, 3)
            'pocket_one_hot': self.data['pocket_one_hot'][idx], # (N_pocket, 20)
            ...
        }

        # NEW: Load cached ESM-C embedding
        receptor_name = self.data['receptors'][idx]
        sample['pocket_emb'] = torch.from_numpy(
            self.esmc[receptor_name]
        ).float()  # (960,)

        return sample
```

**Data loader output** (per batch):
```python
batch = {
    'lig_coords': (N_lig_total, 3),
    'lig_one_hot': (N_lig_total, atom_types),
    'lig_mask': (N_lig_total,),  # Which molecule each atom belongs to
    'pocket_coords': (N_pocket_total, 3),
    'pocket_one_hot': (N_pocket_total, 20),
    'pocket_mask': (N_pocket_total,),
    'pocket_emb': (batch_size, 960),  # NEW! One per pocket
}
```

---

### Phase 3: FiLM Conditioning in Dynamics

**File**: `equivariant_diffusion/dynamics.py`

#### A. Add FiLM Network (in `__init__`)

```python
class EGNNDynamics(nn.Module):
    def __init__(self, atom_nf, residue_nf, joint_nf=128, ...):
        super().__init__()

        # Existing encoders
        self.atom_encoder = nn.Sequential(
            nn.Linear(atom_nf, 2 * atom_nf),
            nn.SiLU(),
            nn.Linear(2 * atom_nf, joint_nf)  # → 128
        )

        self.residue_encoder = nn.Sequential(
            nn.Linear(residue_nf, 2 * residue_nf),
            nn.SiLU(),
            nn.Linear(2 * residue_nf, joint_nf)  # → 128
        )

        # NEW: FiLM conditioning network
        self.pocket_film = nn.Sequential(
            nn.Linear(960, 960),           # Keep high capacity
            nn.SiLU(),
            nn.Linear(960, joint_nf * 2)   # → 256 (γ and β)
        )

        # Existing EGNN layers
        self.egnn = EGNN(...)

        # Existing decoders
        self.atom_decoder = nn.Sequential(...)
        self.residue_decoder = nn.Sequential(...)

        self.joint_nf = joint_nf
```

#### B. Apply FiLM (in `forward`)

```python
def forward(self, xh_ligand_t, xh_pocket, pocket_emb, t, mask_ligand, mask_pocket):
    """
    NEW parameter: pocket_emb (batch_size, 960)
    """

    # 1. Extract coordinates and features (same as before)
    x_ligand = xh_ligand_t[:, :3]
    h_ligand = xh_ligand_t[:, 3:]
    x_pocket = xh_pocket[:, :3]
    h_pocket = xh_pocket[:, 3:]

    # 2. Encode to joint space (same as before)
    h_ligand = self.atom_encoder(h_ligand)      # (N_lig_total, 128)
    h_pocket = self.residue_encoder(h_pocket)   # (N_pocket_total, 128)

    # 3. NEW: Apply FiLM conditioning from ESM-C
    #    This is where the "steering" happens!

    # Compute scale and shift from pocket embedding
    film_params = self.pocket_film(pocket_emb)  # (batch_size, 960) → (batch_size, 256)

    gamma = film_params[:, :self.joint_nf]      # (batch_size, 128) - scale
    beta = film_params[:, self.joint_nf:]       # (batch_size, 128) - shift

    # Expand to all ligand atoms based on batch membership
    # mask_ligand contains which molecule each atom belongs to
    gamma_expanded = gamma[mask_ligand]  # (N_lig_total, 128)
    beta_expanded = beta[mask_ligand]    # (N_lig_total, 128)

    # Apply FiLM: modulate ligand features
    h_ligand = gamma_expanded * h_ligand + beta_expanded
    #          ^^^^^^^^^^^^^^^^   ^^^^^^^^   ^^^^^^^^^^^^^
    #          Scale each         Original   Shift each
    #          channel            features   channel

    # 4. Continue as normal: unified graph, EGNN, decode
    x = torch.cat([x_ligand, x_pocket], dim=0)
    h = torch.cat([h_ligand, h_pocket], dim=0)
    mask = torch.cat([mask_ligand, mask_pocket])

    # Add timestep
    h_time = t[mask]
    h = torch.cat([h, h_time], dim=1)

    # EGNN message passing (now with "steered" ligand features!)
    h_final, x_final = self.egnn(h, x, edges, mask=mask)

    # Decode
    epsilon_ligand = self.atom_decoder(h_final[:N_lig_total])
    epsilon_pocket = self.residue_decoder(h_final[N_lig_total:])

    return epsilon_ligand, epsilon_pocket
```

---

## How FiLM Steering Works

### Mathematical View

**Before FiLM:**
```python
h_ligand[i] = [h₁, h₂, h₃, ..., h₁₂₈]  # Raw encoded features
```

**FiLM parameters from ESM-C:**
```python
gamma = [γ₁, γ₂, γ₃, ..., γ₁₂₈]  # Scale (typically near 1.0)
beta  = [β₁, β₂, β₃, ..., β₁₂₈]  # Shift (typically near 0.0)
```

**After FiLM:**
```python
h_ligand'[i] = [γ₁·h₁ + β₁, γ₂·h₂ + β₂, ..., γ₁₂₈·h₁₂₈ + β₁₂₈]
```

### What Each Parameter Does

**γ (Scale)**:
- `γ > 1`: Amplify this feature channel → "this feature is important for this pocket"
- `γ < 1`: Suppress this feature channel → "this feature is less relevant"
- `γ ≈ 1`: Keep feature as-is

**β (Shift)**:
- `β > 0`: Add positive bias → "shift activations upward"
- `β < 0`: Add negative bias → "shift activations downward"
- `β ≈ 0`: No bias

### Example: Hydrophobic Pocket

**Pocket character** (encoded in ESM-C):
- High hydrophobic conservation
- Typically binds aromatic rings
- Deep binding pocket

**FiLM learns**:
```python
# Channels related to hydrophobicity
gamma[20:30] = [1.5, 1.8, 1.6, ...]  # Amplify!
beta[20:30] = [0.2, 0.3, 0.1, ...]   # Positive shift

# Channels related to polarity
gamma[50:60] = [0.5, 0.3, 0.4, ...]  # Suppress!
beta[50:60] = [-0.1, -0.2, 0.0, ...]  # Negative shift

# Channels related to size
gamma[80:90] = [1.2, 1.3, 1.4, ...]  # Prefer larger
beta[80:90] = [0.1, 0.2, 0.1, ...]
```

**Result**: Ligand features are "tuned" to favor hydrophobic, aromatic, larger molecules → better fit for this pocket!

---

## Visual Architecture

### Full Forward Pass with ESM-C

```
┌─────────────────────────────────────────────────────────────┐
│                    DENOISING STEP t                         │
└─────────────────────────────────────────────────────────────┘

INPUTS:
┌──────────────────┐  ┌──────────────────┐  ┌─────────────┐
│ Noisy Ligand zt  │  │ Pocket (fixed)   │  │ ESM-C (960) │
│   (N_lig, 3+F)   │  │ (N_pocket, 3+20) │  │ cached      │
└──────────────────┘  └──────────────────┘  └─────────────┘
        │                      │                     │
        ↓                      ↓                     │
┌──────────────────┐  ┌──────────────────┐          │
│ Atom Encoder     │  │ Residue Encoder  │          │
│ F → 128          │  │ 20 → 128         │          │
└──────────────────┘  └──────────────────┘          │
        │                      │                     │
        ↓                      │                     ↓
   h_ligand                    │           ┌──────────────────┐
   (N_lig, 128)                │           │  FiLM Network    │
        │                      │           │  960 → 256       │
        │                      │           └──────────────────┘
        │                      │                     │
        │                      │           ┌─────────┴─────────┐
        │                      │           ↓                   ↓
        │                      │      γ (128)             β (128)
        │                      │       SCALE              SHIFT
        │                      │           │                   │
        │                      │           └─────────┬─────────┘
        │                      │                     ↓
        │                      │         ┌─────────────────────┐
        │                      │         │ Broadcast to atoms  │
        │                      │         │ via batch mask      │
        │                      │         └─────────────────────┘
        │                      │                     │
        │                      │           ┌─────────┴─────────┐
        │                      │           ↓                   ↓
        │                      │     γ_expanded          β_expanded
        │                      │     (N_lig, 128)        (N_lig, 128)
        │                      │           │                   │
        └──────────────────────┴───────────┴───────────────────┘
                               │
                               ↓
                    h_ligand' = γ·h_ligand + β  ← STEERING!
                               │
                    ┌──────────┴──────────┐
                    ↓                     ↓
             h_ligand' (steered)    h_pocket (unchanged)
                    │                     │
                    └──────────┬──────────┘
                               ↓
                    ┌─────────────────────┐
                    │  Unified Graph      │
                    │  x = [x_lig; x_poc] │
                    │  h = [h_lig'; h_poc]│
                    └─────────────────────┘
                               │
                               ↓
                    ┌─────────────────────┐
                    │  EGNN Layers        │
                    │  (Message Passing)  │
                    │  • Ligand-Ligand    │
                    │  • Ligand-Pocket    │
                    │  • Pocket-Pocket    │
                    └─────────────────────┘
                               │
                               ↓
                    ┌─────────────────────┐
                    │  Decoders           │
                    │  128 → F            │
                    └─────────────────────┘
                               │
                               ↓
                    Predicted noise ε
                               │
                               ↓
                        Update: zt → zt-1
```

---

## Concrete Example

### Scenario: Hydrophobic Binding Pocket

**Pocket**: Kinase ATP binding site
- Deep hydrophobic pocket
- Conserved aromatic residues (PHE, TRP)
- Typically binds purine-like scaffolds

**ESM-C embedding** (960-dim vector) captures:
- High conservation of key residues
- Hydrophobic microenvironment
- Preference for flat, aromatic molecules

### Step 1: Initial Ligand Features

```python
# After encoding, before FiLM
h_ligand[atom_42] = [
    0.5,   # Channel 0: polarity-related
    -0.3,  # Channel 1: size-related
    0.8,   # Channel 2: hydrophobicity-related
    0.1,   # Channel 3: aromaticity-related
    ...    # 124 more channels
]
```

### Step 2: FiLM Parameters from ESM-C

```python
# FiLM network learns from ESM-C
gamma = [
    0.5,   # Channel 0: suppress polarity (hydrophobic pocket!)
    1.2,   # Channel 1: keep size
    1.8,   # Channel 2: amplify hydrophobicity ✓
    1.5,   # Channel 3: amplify aromaticity ✓
    ...
]

beta = [
    -0.2,  # Channel 0: shift polarity down
    0.0,   # Channel 1: no shift
    0.3,   # Channel 2: shift hydrophobicity up ✓
    0.2,   # Channel 3: shift aromaticity up ✓
    ...
]
```

### Step 3: After FiLM

```python
# Apply: h' = γ·h + β
h_ligand'[atom_42] = [
    0.5 * 0.5 - 0.2 = 0.05,    # Polarity suppressed ✓
    1.2 * (-0.3) + 0.0 = -0.36, # Size kept
    1.8 * 0.8 + 0.3 = 1.74,     # Hydrophobicity amplified! ✓
    1.5 * 0.1 + 0.2 = 0.35,     # Aromaticity amplified! ✓
    ...
]
```

### Step 4: EGNN Message Passing

Now when EGNN does message passing:
```python
# Ligand atom 42 receives messages from pocket
message = MLP(h_ligand'[42], h_pocket[j], distance)

# h_ligand'[42] now emphasizes hydrophobicity and aromaticity
# → Messages will favor hydrophobic/aromatic interactions
# → Denoising step will move toward hydrophobic/aromatic molecule
```

### Result

Over T denoising steps:
- Initial noise → gradually shaped
- FiLM guides at each step
- Final molecule: hydrophobic, aromatic, flat (good fit for kinase!)

---

## Training Dynamics

### What Gets Trained?

**Trainable (updated by backprop):**
```python
✅ self.pocket_film  # FiLM network weights
✅ self.atom_encoder  # Ligand encoder
✅ self.residue_encoder  # Pocket encoder
✅ self.egnn  # Message passing layers
✅ self.atom_decoder  # Output decoder
```

**Fixed (not trained):**
```python
❌ ESM-C embeddings  # Pre-computed, cached
❌ Pocket coordinates  # Fixed during generation
```

### Gradient Flow

```
Loss = ||ε_pred - ε_true||²
   │
   ↓ ∂L/∂ε_pred
Decoder
   │
   ↓ ∂L/∂h_final
EGNN
   │
   ↓ ∂L/∂h_ligand'
FiLM: h' = γ·h + β
   │
   ├→ ∂L/∂γ → Update FiLM network ✓
   ├→ ∂L/∂β → Update FiLM network ✓
   └→ ∂L/∂h → Update encoder ✓
```

**Key**: Gradients flow through FiLM, teaching it which features to amplify/suppress based on loss!

### Learning Process

**Epoch 1**:
- γ ≈ 1.0, β ≈ 0.0 (near identity)
- FiLM has minimal effect
- Model relies on geometric conditioning

**Epoch 50**:
- γ starts deviating from 1.0
- β starts showing patterns
- FiLM learns basic pocket preferences

**Epoch 200**:
- γ shows clear amplification/suppression patterns
- β learns pocket-specific biases
- Model generates better molecules for specific pocket types

**Epoch 500** (converged):
- γ and β are well-tuned
- ESM-C conditioning strongly influences generation
- Molecules better match pocket character

---

## Why This is "Conditioning"

### Comparison with Text-to-Image

| Aspect | Text-to-Image (Stable Diffusion) | Pocket-to-Ligand (DiffSBDD + ESM-C) |
|--------|----------------------------------|-------------------------------------|
| **Input** | Text prompt: "a cat on a beach" | Pocket sequence + structure |
| **Encoder** | CLIP (trained on text-image pairs) | ESM-C (trained on evolution) |
| **Embedding** | 768-dim text features | 960-dim pocket features |
| **Conditioning** | Cross-attention / FiLM | FiLM modulation |
| **Denoising** | U-Net (handles spatial) | EGNN (handles 3D geometry) |
| **Output** | Realistic image with cats, beach | Realistic ligand for pocket |
| **Mechanism** | Text features modulate image features | Pocket features modulate ligand features |

### Key Parallels

1. **Semantic Guidance**:
   - Text: "Make it look like a cat" → guides image generation
   - ESM-C: "This is a hydrophobic pocket" → guides ligand generation

2. **Iterative Refinement**:
   - Both use diffusion: noise → signal over T steps
   - Conditioning applied at every step

3. **Modularity**:
   - Can change text prompt without retraining image model
   - Can use different ESM-C embeddings without retraining diffusion

4. **Feature Modulation**:
   - Both use learned transformations to incorporate conditioning
   - FiLM: γ and β learned to translate semantic info into feature space

---

## Summary: The Complete Picture

### What We Add

**One line of input**:
```python
pocket_emb: (960,)  # Pre-computed ESM-C embedding
```

**One network**:
```python
self.pocket_film = MLP(960 → 256)  # Converts ESM-C to γ and β
```

**One operation**:
```python
h_ligand = gamma * h_ligand + beta  # Feature-wise modulation
```

### What We Get

**Ligand generation now considers**:
- ✅ Geometric constraints (EGNN) - prevents clashes, respects distances
- ✅ **Evolutionary context (ESM-C)** - prefers realistic binding patterns
- ✅ **Functional information (ESM-C)** - matches pocket character
- ✅ **Physicochemical properties (ESM-C)** - appropriate hydrophobicity, charge

### Why It Should Work

1. **ESM-C captures real biological signal** - trained on evolution
2. **FiLM is proven conditioning technique** - works in many domains
3. **Clean separation** - geometry (EGNN) + semantics (ESM-C)
4. **Differentiable** - gradients teach FiLM what to amplify/suppress
5. **Low risk** - if unhelpful, model learns γ≈1, β≈0 (identity)

### The Key Insight

**Before**: Model only knows "this is Leucine"
**After**: Model knows "this is a highly conserved Leucine in a hydrophobic binding pocket that typically binds aromatic compounds"

→ Better molecules! 🎯

---

## References

**Main Papers**:
- DiffSBDD: Schneuing et al., Nature Computational Science 2024
- ESM-C: Hayes et al., bioRxiv 2024
- FiLM: Perez et al., "FiLM: Visual Reasoning with a General Conditioning Layer", AAAI 2018
- Stable Diffusion: Rombach et al., "High-Resolution Image Synthesis with Latent Diffusion Models", CVPR 2022

**Code**:
- DiffSBDD: https://github.com/arneschneuing/DiffSBDD
- ESM: https://github.com/evolutionaryscale/esm

---

**Last Updated**: December 2024
