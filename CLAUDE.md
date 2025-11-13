# DiffSBDD Codebase Architecture & Overview

## Executive Summary

**DiffSBDD** is an **Equivariant Diffusion Model for Structure-Based Drug Design**. It generates novel small-molecule ligands that bind to protein pockets. The model is equivariant to rotations and translations (SE(3) equivariance), meaning it respects 3D geometric symmetries.

**Core Task**: Given a protein binding pocket, generate 3D ligand molecules that:
- Are spatially feasible (valid bond distances)
- Are chemically valid (valid valencies)
- Have desirable drug properties (QED, SA Score)
- Bind favorably to the target pocket

---

## 1. Overall Architecture

### High-Level Flow

```
Data (Ligand + Pocket) 
    ↓
Training/Inference
    ├─ [TRAINING] LigandPocketDDPM (PyTorch Lightning)
    │   ├─ Loads data via ProcessedLigandPocketDataset
    │   ├─ Processes through EnVariationalDiffusion or ConditionalDDPM
    │   ├─ EGNNDynamics computes denoising updates
    │   └─ Outputs loss for optimization
    │
    └─ [INFERENCE] generate_ligands() or inpaint()
        ├─ Loads checkpoint
        ├─ Samples from DDPM
        ├─ Builds RDKit molecules
        └─ Evaluates quality (metrics, docking)
```

### Key Design Patterns

1. **PyTorch Lightning Integration**: Training orchestration via `LigandPocketDDPM(pl.LightningModule)`
2. **Equivariance**: 3D geometry respects rotation/translation via EGNN
3. **Conditional vs Joint**: Two training modes
   - **Conditional ("pocket_conditioning")**: Pocket is fixed, only generate ligand
   - **Joint ("joint")**: Generate both ligand and pocket coordinates together
4. **Diffusion Approach**: Noise → Signal via score matching (denoising)

---

## 2. Key Modules & Responsibilities

### 2.1 `lightning_modules.py` (1079 lines)

**Core Class**: `LigandPocketDDPM(pl.LightningModule)`

**Responsibilities**:
- Training pipeline orchestration (PyTorch Lightning)
- Model initialization and checkpointing
- Data loading and batch processing
- Sampling and inference
- Evaluation metrics computation
- Loss computation and backpropagation

**Key Methods**:
- `__init__()`: Initializes DDPM model, metrics, data transforms
- `setup()`: Loads train/val/test datasets
- `forward()`: Computes variational loss (VLB objective)
- `training_step()`: Single training iteration
- `validation_step()`: Evaluation on validation set
- `generate_ligands()`: Main inference method (lines 897-1036)
  - Takes PDB file + pocket definition
  - Samples ligand coordinates and atom types
  - Converts to RDKit molecules
  - Applies molecule filtering/optimization
- `sample_and_save_given_pocket()`: Generate and visualize samples
- `sample_chain_and_save()`: Visualize diffusion trajectory
- `prepare_pocket()`: Extracts and encodes pocket residues

**Virtual Nodes Feature**:
- Pads ligands to fixed size with dummy "Ne" atoms
- Enables fixed tensor shapes during diffusion

### 2.2 `equivariant_diffusion/` Directory

#### 2.2.1 `en_diffusion.py` (1190 lines)

**Main Classes**:

1. **`EnVariationalDiffusion(nn.Module)`** - Joint diffusion model
   - Diffuses both ligand AND pocket coordinates
   - Used when `mode="joint"`
   - Implements variational lower bound (VLB) objective
   
2. **`DistributionNodes`** (line 958)
   - Models ligand size distribution P(N|ligand size)
   - Learned from training data histogram
   - Used to sample `num_nodes_lig` during generation

**Key Methods**:
- `forward()`: Computes loss and KL divergence
  - Takes ligand and pocket as dictionaries with keys: `x` (coords), `one_hot` (atom types), `mask`, `size`
  - Returns loss terms for all components
- `kl_prior_with_pocket()`: KL divergence between q(z_T|x) and prior N(0,1)
- `sample()`: Generates samples from prior through reverse diffusion
- `sample_given_pocket()`: Conditional generation (fixed pocket)
- `sample_p_zs_given_zt()`: One denoising step in reverse process
- `gamma()`: Noise schedule (learned or predefined)
- `sigma()`: Noise standard deviation at timestep t

**Noise Schedule Options**:
- `learned`: Network learns γ_t
- `polynomial_2`: Predefined polynomial schedule
- `cosine`: Cosine annealing
- Others: `linear`, `sqrt`

#### 2.2.2 `conditional_model.py` (31796 bytes)

**Main Class**: `ConditionalDDPM(EnVariationalDiffusion)`

**Key Difference from Joint**:
- Pocket coordinates are FIXED during diffusion
- Only ligand is sampled
- Used when `mode="pocket_conditioning"`
- Simpler KL divergence (no pocket contribution)

**Key Methods**:
- `kl_prior()`: Only computes KL for ligand
- `log_pxh_given_z0_without_constants()`: Likelihood computation
- `sample_given_pocket()`: Conditional sampling given pocket

#### 2.2.3 `dynamics.py` (8723 bytes)

**Main Class**: `EGNNDynamics(nn.Module)`

**Purpose**: The denoising network that predicts ε_t given z_t

**Architecture**:
```
Input: [x_atoms, h_atoms, x_residues, h_residues, t]
  ↓
Atom/Residue Encoders (MLPs)
  ↓
EGNN (Equivariant Graph Neural Network)
  - Updates coordinates equivariantly
  - Updates features invariantly
  - Uses edge features (optional embedding)
  ↓
Atom/Residue Decoders (MLPs)
  ↓
Output: [eps_x_atoms, eps_h_atoms, eps_x_residues, eps_h_residues]
```

**Key Features**:
- Encodes atoms/residues from one-hot to joint feature space
- EGNN preserves SE(3) equivariance
- Can use edge cutoffs to limit interaction distances
- Edge embedding dimension customizable
- Two modes: `egnn_dynamics` (recommended), `gnn_dynamics`

**Critical Note**: Currently has placeholder for ESM-C embeddings (see comments in code)

#### 2.2.4 `egnn_new.py` (14847 bytes)

**Main Classes**:
- `EGNN`: Equivariant Graph Neural Network
- `GNN`: Non-equivariant GNN variant

**EGNN Design**:
- Splits features into invariant (node features h) and equivariant (coordinates x)
- Message passing uses edge vectors to compute invariant aggregations
- Updates coordinates by predicting coordinate deltas
- Multi-layer with attention and normalization options

---

### 2.3 `dataset.py` (71 lines)

**Class**: `ProcessedLigandPocketDataset(Dataset)`

**Data Format** (NPZ files):
Each NPZ file contains concatenated data for all molecules:
- `lig_coords`: Ligand atom coordinates
- `lig_one_hot`: One-hot encoded ligand atom types
- `lig_mask`: Batch indices for ligands
- `pocket_coords`: Pocket atom/residue coordinates
- `pocket_one_hot`: One-hot encoded pocket residues/atoms
- `pocket_mask`: Batch indices for pockets
- `names`: Molecule identifiers
- `receptors`: Receptor names

**Processing**:
- Loads NPZ as dictionary of tensors
- Splits by mask boundaries
- Centers ligand+pocket around origin
- Supports custom transforms (e.g., virtual nodes)

**Collate Function**:
- Concatenates multiple samples
- Creates batch masks (sample indices)
- Handles mask tensors specially for torch_scatter

---

### 2.4 `analysis/` Directory

#### 2.4.1 `metrics.py` (8995 bytes)

**Key Classes**:

1. **`CategoricalDistribution`**
   - Tracks atom/residue type distributions
   - Computes KL divergence from generated molecules
   
2. **`BasicMolecularMetrics`**
   - `compute_validity()`: Checks if molecules are chemically valid
   - `compute_connectivity()`: Checks if molecules are connected
   - `compute_uniqueness()`: Deduplicates by SMILES
   - `compute_novelty()`: Checks if molecules are novel vs training set
   
3. **`MoleculeProperties`** (referenced but not fully shown)
   - Likely computes QED, SA Score, LogP, Lipinski rules

#### 2.4.2 `molecule_builder.py` (269 lines)

**Key Functions**:

1. **`get_bond_order()`**
   - Given distance between atoms, predicts bond type
   - Uses bond length lookup tables from `constants.py`
   - Returns 0 (no bond), 1 (single), 2 (double), 3 (triple)
   
2. **`make_mol_openbabel()`** (lines 69-108)
   - Uses OpenBabel to infer bonds from 3D coordinates
   - More robust for complex molecules
   
3. **`make_mol_edm()`** (lines 111-153)
   - Uses bond length tables (EDM approach)
   - Computes pairwise distances
   - Assigns bond types based on tables

4. **`build_molecule()`** (lines 156-175)
   - Main entry point
   - Calls OpenBabel by default (more robust)
   - Returns RDKit molecule with conformer
   
5. **`process_molecule()`** (lines 178-230)
   - Sanitizes (fixes valencies)
   - Adds hydrogens
   - Keeps only largest fragment
   - Relaxes structure with UFF force field
   
6. **`filter_rd_mol()`**
   - Filters out molecules with 3-3 ring intersections

#### 2.4.3 `visualization.py` (16203 bytes)

- Writes XYZ files for molecular visualization
- Creates PNG images with atom colors by type
- Integrates with WandB for logging

#### 2.4.4 `docking.py` (7011 bytes)

- `smina_score()`: Molecular docking scoring
- Integrates with external docking software

---

### 2.5 `utils.py` (235 lines)

**Key Utilities**:

1. **`Queue`**: Fixed-size queue for tracking gradient norms
2. **`get_grad_norm()`**: Computes gradient norm for clipping
3. **`write_xyz_file()`**: Writes molecules to XYZ format
4. **`write_sdf_file()`**: Writes molecules to SDF format
5. **`residues_to_atoms()`**: Converts CA atoms to all-atom representation
6. **`get_pocket_from_ligand()`**: Defines pocket by distance cutoff from reference ligand
7. **`batch_to_list()`**: Splits batch tensor by mask
8. **`num_nodes_to_batch_mask()`**: Creates batch mask from node counts
9. **`calc_rmsd()`**: RMSD with unknown atom correspondence (graph isomorphism)
10. **`AppendVirtualNodes`**: Transform class to pad molecules with dummy atoms

---

### 2.6 `constants.py` (16196 bytes)

**Contains**:
- `FLOAT_TYPE`, `INT_TYPE`: Data types
- Bond parameters (lengths, margins)
- `allowed_bonds`: Valency rules per atom type
- `bonds1/2/3`: Single/double/triple bond length tables
- `covalent_radii`: Van der Waals radii
- Backbone geometry constants (N-CA distance, angles)
- **`dataset_params`**: Dictionary with dataset-specific info:
  - `atom_encoder`/`atom_decoder`: Ligand atom types
  - `aa_encoder`/`aa_decoder`: Amino acid types
  - `atom_hist`/`aa_hist`: Empirical distributions
  - `lennard_jones_rm`: LJ parameters for atoms

---

### 2.7 Configuration System (`configs/`)

**Format**: YAML files

**Example structure** (`crossdock_fullatom_cond.yml`):
```yaml
run_name: "SE3-cond-full"
logdir: "logs"
wandb_params:
  entity: "..."
  group: "crossdock"
  mode: "online"

mode: "pocket_conditioning"  # or "joint"
pocket_representation: "full-atom"  # or "CA"
virtual_nodes: False

batch_size: 16
lr: 1.0e-3
n_epochs: 1000
gpus: 4
clip_grad: True

egnn_params:
  joint_nf: 128
  hidden_nf: 256
  n_layers: 6
  attention: True
  edge_cutoff_pocket: 5.0
  edge_cutoff_interaction: 5.0

diffusion_params:
  diffusion_steps: 500
  diffusion_noise_schedule: "polynomial_2"
  diffusion_loss_type: "l2"  # or "vlb"
  normalize_factors: [1, 4]  # [x, h]

eval_params:
  n_eval_samples: 100
  eval_batch_size: 100
```

**Loading**: `train.py` merges CLI args + config YAML via `merge_args_and_yaml()`

---

## 3. Training Flow

### Overview
```
train.py
  ├─ Parse config YAML
  ├─ Create LigandPocketDDPM
  │   ├─ Initialize EGNNDynamics (denoising network)
  │   ├─ Initialize EnVariationalDiffusion or ConditionalDDPM
  │   ├─ Create metrics (validity, uniqueness, novelty)
  │   └─ Create data loaders
  ├─ Setup WandB logger
  ├─ Create PyTorch Lightning Trainer
  └─ trainer.fit(model)
```

### Training Step (`training_step()`)

1. **Forward Pass** (lines 274-356):
   - Extract ligand and pocket from batch
   - Call `self.ddpm()` which returns:
     - `delta_log_px`: Log-det-Jacobian correction
     - `error_t_lig`: Ligand denoising error
     - `error_t_pocket`: Pocket denoising error
     - `SNR_weight`: Signal-to-noise ratio weight
     - Various loss components (loss_0, KL_prior, etc.)

2. **Loss Computation**:
   - **L2 loss** (training mode):
     ```
     loss_t = 0.5 * (error_t_lig / denom_lig + error_t_pocket / denom_pocket)
     loss_0 = (loss_0_x_ligand + loss_0_x_pocket + loss_0_h)
     nll = loss_t + loss_0 + kl_prior
     ```
   - **VLB loss** (evaluation mode):
     ```
     loss_t = -T * 0.5 * SNR_weight * (error_t_lig + error_t_pocket)
     (with additional terms)
     ```

3. **Auxiliary Loss** (optional):
   - Lennard-Jones potential penalty to discourage clashes
   - Weighted by schedule during early timesteps

4. **Gradient Clipping** (optional):
   - Track gradient norms in Queue
   - Set max norm to 150% mean + 2*stdev

### Validation & Evaluation

- Every `eval_epochs`: Sample molecules and compute metrics
  - Validity, connectivity, uniqueness, novelty
  - Atom type distributions
  - DFT metrics (optional)

---

## 4. Inference / Generation Flow

### Entry Points

1. **`generate_ligands.py`** (lines 897-1036 in lightning_modules.py)
   - Most common entry point
   - Takes PDB file + pocket definition
   - Returns RDKit molecules

2. **`inpaint.py`** (8952 bytes)
   - Partial ligand redesign
   - Fix some atoms, regenerate rest
   - Uses joint diffusion with masks

3. **`optimize.py`** (9554 bytes)
   - Evolutionary algorithm
   - Optimize molecules for properties (QED, SA Score)
   - Uses inpainting steps

### Generation Process (high-level)

```python
# 1. Load model
model = LigandPocketDDPM.load_from_checkpoint(checkpoint)

# 2. Define pocket
residues = get_pocket_from_ligand(pdb, ref_ligand)
pocket = prepare_pocket(residues, repeats=n_samples)

# 3. Sample ligand size
num_nodes_lig = ddpm.size_distribution.sample_conditional(
    n1=None, n2=pocket["size"]
)

# 4. Reverse diffusion (conditional)
xh_lig, xh_pocket, lig_mask, pocket_mask = ddpm.sample_given_pocket(
    pocket, num_nodes_lig
)

# 5. Build molecules
for coords, atom_types in batch_data:
    mol = build_molecule(coords, atom_types, dataset_info)
    mol = process_molecule(mol, sanitize=True, largest_frag=True)
    molecules.append(mol)

# 6. Write SDF
write_sdf_file(output_file, molecules)
```

### `sample_given_pocket()` (from en_diffusion.py)

**Steps**:
1. Sample z_0 from standard normal (respecting center-of-mass constraint)
2. For t = T, T-1, ..., 1:
   - Predict ε_t using dynamics network
   - Compute posterior mean and variance
   - Sample z_{t-1}
   - Remove mean to enforce center-of-mass = 0
3. Return z_0 (final sample)

---

## 5. Data Processing

### Input Format (NPZ)

- Processed data stored in NPZ format
- One file per split: `train.npz`, `val.npz`, `test.npz`

### Data Preparation Scripts

1. **`process_crossdock.py`** (18348 bytes)
   - Processes CrossDocked dataset
   - Extracts pockets and ligands from PDB files
   - Computes distance-based pocket boundaries

2. **`process_bindingmoad.py`** (25281 bytes)
   - Processes Binding MOAD dataset
   - Similar workflow

### NPZ Structure

After loading, each sample contains:
```python
{
    'lig_coords': torch.Tensor (n_atoms, 3),
    'lig_one_hot': torch.Tensor (n_atoms, atom_nf),
    'lig_mask': torch.Tensor (n_atoms,),
    'pocket_coords': torch.Tensor (n_residues, 3) or (n_atoms, 3),
    'pocket_one_hot': torch.Tensor (n_residues, aa_nf) or (n_atoms, atom_nf),
    'pocket_mask': torch.Tensor (n_residues,) or (n_atoms,),
    'names': str,
    'receptors': str,
}
```

### Centering

Data is centered around the mean of ligand + pocket:
```python
mean = (lig_coords.sum(0) + pocket_coords.sum(0)) / (len(lig) + len(pocket))
lig_coords -= mean
pocket_coords -= mean
```

---

## 6. Model Architecture

### EGNN (Equivariant Graph Neural Network)

**Equivariance Property**: If we rotate/translate input coordinates, outputs transform the same way.

**Architecture Concept**:
```
Node embeddings h_i and coordinates x_i

Message passing round:
  For each edge (i, j):
    - Compute edge features (invariant): r_ij = ||x_i - x_j||
    - MLP(h_i, h_j, r_ij) → edge scalar message m_ij
    
  For each node i:
    - Aggregate invariant features: agg_j(m_ij)
    - Update h_i' = h_i + agg_j(m_ij)
    - Compute coordinate update: Δx_i = sum_j (m_ij / r_ij) * (x_i - x_j)
    - x_i' = x_i + Δx_i
```

**Key Properties**:
- h updates are invariant (coordinates don't appear directly)
- x updates scale with edge direction (equivariant)
- Respects SE(3) symmetry

### Diffusion Model

**Parametrization**: Noise prediction (ε-prediction)

**Forward Process** (noising):
```
z_t = α_t * z_0 + σ_t * ε,  where ε ~ N(0, I)
α_t = sqrt(1 - σ_t²)
```

**Noise Schedule**:
- Controlled by γ_t = log(α_t²) - log(σ_t²)
- Monotonic: γ_0 ≈ 0 (clean), γ_T ≈ -∞ (noise)

**Reverse Process** (denoising):
```
z_{t-1} ~ p(z_{t-1} | z_t) = N(μ_t, σ_{t|t-1}²)
μ_t depends on ε_t prediction from network
```

**Loss** (training):
```
L = ||ε_t - ε_θ(z_t, t)||²
```

---

## 7. Key Abstractions & Their Relationships

### Data Flow Diagram

```
Raw PDB files
    ↓
process_crossdock.py / process_bindingmoad.py
    ↓
NPZ files (processed_crossdock_noH_full/)
    ↓
ProcessedLigandPocketDataset
    ↓
DataLoader (batching via collate_fn)
    ↓
LigandPocketDDPM.forward()
    ├─ Ligand dict: {x, one_hot, size, mask}
    ├─ Pocket dict: {x, one_hot, size, mask}
    ↓
EnVariationalDiffusion or ConditionalDDPM
    ├─ Add noise (q process)
    ├─ Predict noise with EGNNDynamics
    ├─ Compute loss
    ↓
Training loss → backprop → optimizer step
```

### Class Hierarchy

```
pl.LightningModule
    └─ LigandPocketDDPM
            ├─ uses: EnVariationalDiffusion (joint mode)
            │         └─ uses: EGNNDynamics
            │         └─ uses: PredefinedNoiseSchedule / GammaNetwork
            │         └─ uses: DistributionNodes
            │
            ├─ uses: ConditionalDDPM (conditional mode)
            │         └─ (same dependencies as above)
            │
            ├─ uses: BasicMolecularMetrics
            ├─ uses: MoleculeProperties
            ├─ uses: CategoricalDistribution
            └─ uses: AppendVirtualNodes (if virtual_nodes=True)

nn.Module
    ├─ EnVariationalDiffusion
    │   ├─ EGNNDynamics
    │   ├─ GammaNetwork or PredefinedNoiseSchedule
    │   └─ DistributionNodes
    │
    ├─ ConditionalDDPM (extends EnVariationalDiffusion)
    │   └─ (same as above)
    │
    ├─ EGNNDynamics
    │   ├─ EGNN
    │   └─ MLPs (encoders/decoders)
    │
    └─ EGNN
        └─ GNN (message passing layers)
```

---

## 8. Important Implementation Details

### Virtual Nodes

**Purpose**: Make ligand size fixed during training

**Implementation** (`utils.AppendVirtualNodes`):
1. Adds dummy "Ne" (Neon) atoms to ligand
2. Pads up to `max_num_nodes` from histogram
3. Marks with `num_virtual_atoms` field
4. Virtual atom loss is zeroed out
5. Allows fixed tensor shapes, better batching

### Mask Tensors

**Format**: Batch indices for scatter operations

Example: If batch has [mol1: 5 atoms, mol2: 3 atoms, mol3: 4 atoms]:
```
mask = [0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2]
```

Used with `torch_scatter` for safe aggregation across batch samples.

### Center-of-Mass Constraint

**Why**: Diffusion in translational-equivariant space

**Implementation**:
```python
# After each denoising step:
x = remove_mean_batch(x, mask)
# This ensures: sum(x * mask) == 0 for each batch element
```

### Pocket Representation

Two options:
1. **CA only**: Use Cα atoms only (simpler, ~20 residue types)
2. **Full-atom**: All pocket atoms (more detailed, ~10 atom types)

### Loss Type

1. **L2** (default): Simple MSE loss
   ```
   loss = ||ε_pred - ε_true||²
   ```
   Faster training, good for large-scale experiments.

2. **VLB**: Variational lower bound on likelihood
   ```
   -log p(x) <= L_VLB
   ```
   More principled but slower.

---

## 9. Configuration Key Parameters

### Model Hyperparameters

- `joint_nf`: Shared feature dimension for atoms/residues (128)
- `hidden_nf`: EGNN hidden dimension (256)
- `n_layers`: EGNN depth (6)
- `attention`: Enable attention in EGNN (True/False)
- `tanh`: Use tanh activation (True/False)
- `edge_cutoff_*`: Distance cutoffs for interactions (e.g., 5.0 Å)

### Diffusion Hyperparameters

- `diffusion_steps`: T (number of timesteps, typically 500)
- `diffusion_noise_schedule`: Type of schedule (polynomial_2, cosine, etc.)
- `diffusion_loss_type`: L2 or VLB
- `normalize_factors`: [norm_x, norm_h] for data normalization

### Training Hyperparameters

- `batch_size`: Batch size (16-128)
- `lr`: Learning rate (1e-3 to 1e-4)
- `n_epochs`: Total epochs (500-1000)
- `clip_grad`: Gradient clipping enabled
- `augment_rotation`: Random rotation augmentation (False in current configs)

### Evaluation Parameters

- `eval_epochs`: Compute metrics every N epochs
- `n_eval_samples`: Number of molecules to sample for eval
- `visualize_sample_epoch`: Save visualization every N epochs
- `keep_frames`: Number of diffusion frames to keep for visualization

---

## 10. Common Workflows & Entry Points

### Workflow 1: Training from Scratch

```bash
# 1. Prepare data
python process_crossdock.py /path/to/crossdocked --no_H

# 2. Train
python train.py --config configs/crossdock_fullatom_cond.yml

# 3. Monitor with WandB
# Logs go to logs/SE3-cond-full/
```

### Workflow 2: Generate Ligands

```bash
# 1. Load checkpoint
checkpoint = "checkpoints/crossdocked_fullatom_cond.ckpt"

# 2. Generate (via generate_ligands.py)
python generate_ligands.py $checkpoint \
    --pdbfile example/3rfm.pdb \
    --outfile example/3rfm_mol.sdf \
    --ref_ligand A:330 \
    --n_samples 20

# OR with residue list
python generate_ligands.py $checkpoint \
    --pdbfile example/3rfm.pdb \
    --outfile example/3rfm_mol.sdf \
    --resi_list A:1 A:2 A:3 ... \
    --n_samples 20
```

### Workflow 3: Substructure Inpainting

```bash
python inpaint.py $checkpoint \
    --pdbfile example/5ndu.pdb \
    --outfile example/5ndu_linked.sdf \
    --ref_ligand example/5ndu_C_8V2.sdf \
    --fix_atoms example/fragments.sdf \
    --add_n_nodes 10 \
    --center ligand
```

### Workflow 4: Molecular Optimization

```bash
python optimize.py \
    --checkpoint $checkpoint \
    --pdbfile example/5ndu.pdb \
    --outfile output.sdf \
    --ref_ligand example/5ndu_C_8V2.sdf \
    --objective sa \  # or 'qed'
    --population_size 100 \
    --evolution_steps 10 \
    --top_k 10 \
    --timesteps 100
```

---

## 11. Debugging & Common Issues

### Virtual Nodes Comment in dynamics.py
```python
# Line 38: "das würde ich durch das embedding aus ESM-C ersetzen residue nf = 960"
```
Translation: "I would replace this with the embedding from ESM-C with residue_nf = 960"

This suggests potential integration with protein language models (ESM-C) for better pocket representations.

### Key Variables to Monitor

1. **Gradient norm**: Tracked in `gradnorm_queue`
2. **Loss components**:
   - `error_t_lig`: Ligand coordinate error
   - `error_t_pocket`: Pocket coordinate error
   - `loss_0`: Reconstruction loss at t=0
   - `kl_prior`: KL divergence of prior
3. **Sampling metrics**:
   - Validity: % chemically valid
   - Connectivity: % connected molecules
   - Uniqueness: % unique (by SMILES)
   - Novelty: % not in training set

### Common Failure Modes

1. **Invalid molecules**: Check bond prediction in `molecule_builder.py`
2. **Clashing atoms**: Lennard-Jones auxiliary loss helps
3. **Low validity**: May need more training or better data
4. **Mode collapse**: Monitor uniqueness metric

---

## 12. Summary Table: Key Files

| File | Lines | Purpose |
|------|-------|---------|
| `lightning_modules.py` | 1079 | PyTorch Lightning training & inference |
| `equivariant_diffusion/en_diffusion.py` | 1190 | Diffusion model core |
| `equivariant_diffusion/conditional_model.py` | ~500 | Conditional variant |
| `equivariant_diffusion/dynamics.py` | 278 | EGNN-based denoising network |
| `equivariant_diffusion/egnn_new.py` | 511 | Graph neural network |
| `dataset.py` | 71 | Data loading & batching |
| `analysis/metrics.py` | 245 | Molecular metrics |
| `analysis/molecule_builder.py` | 269 | Bond prediction & RDKit conversion |
| `analysis/visualization.py` | 452 | XYZ & PNG visualization |
| `utils.py` | 235 | Helper functions |
| `constants.py` | 463 | Bond tables & dataset params |
| `configs/*.yml` | ~1.5KB each | Training configurations |
| `train.py` | 139 | Training entry point |
| `generate_ligands.py` | 61 | Inference entry point |
| `inpaint.py` | 298 | Inpainting entry point |
| `optimize.py` | 338 | Optimization entry point |

---

## 13. Paper Reference

**Citation**:
```bibtex
@article{schneuing2024diffsbdd,
   title={Structure-based drug design with equivariant diffusion models},
   author={Schneuing, Arne and Harris, Charles and Du, Yuanqi and ...},
   journal={Nature Computational Science},
   year={2024},
   volume={4}, number={12}, pages={899-909}
   doi={10.1038/s43588-024-00737-x}
}
```

Key concepts from paper:
- SE(3)-equivariant diffusion
- Two modes: joint (pocket + ligand) and conditional (ligand only)
- Variational lower bound loss for likelihood estimation
- Pocket representation: CA atoms or full-atom
- Evaluation: validity, connectivity, uniqueness, novelty, docking scores

---

## Quick Reference: API

### Sampling a Single Molecule

```python
import torch
from lightning_modules import LigandPocketDDPM

# Load model
model = LigandPocketDDPM.load_from_checkpoint(checkpoint)
model = model.to('cuda')
model.eval()

# Generate
molecules = model.generate_ligands(
    pdb_file="example.pdb",
    n_samples=10,
    ref_ligand="A:330",  # Define pocket by reference
)

# Access molecules
for mol in molecules:
    print(mol.GetNumAtoms(), "atoms")
    smi = Chem.MolToSmiles(mol)
    print(f"SMILES: {smi}")
```

### Training Custom Model

```bash
# Create config (e.g., configs/my_config.yml)
# Edit: datadir, run_name, egnn_params, diffusion_params

python train.py --config configs/my_config.yml
```

### Resuming Training

```bash
python train.py --config configs/my_config.yml \
    --resume logs/my_run_name/checkpoints/last.ckpt
```

---

**End of Architecture Overview**
