# DiffSBDD Architecture Diagrams & Visual Guides

## 1. High-Level System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DiffSBDD System                                   │
└─────────────────────────────────────────────────────────────────────────────┘

                            TRAINING PATH
                            
┌──────────────┐     ┌──────────────────┐     ┌────────────────────────────┐
│   PDB Files  │────>│  Data Processors │────>│   Processed Data (NPZ)     │
│ + SMILES     │     │ (process_cross   │     │ train.npz, val.npz, etc.  │
└──────────────┘     │  process_bindingmoad) └────────────────────────────┘
                     └──────────────────┘                    │
                                                             ↓
                     ┌────────────────────────────┐  ┌──────────────────────┐
                     │ ProcessedLigandPocket      │<─┤ DataLoader           │
                     │ Dataset                    │  │ (batch, collate_fn)  │
                     └────────────────────────────┘  └──────────────────────┘
                                                            │
                                                            ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                    LigandPocketDDPM (PyTorch Lightning)                     │
│  ┌──────────────────┐  ┌─────────────────┐  ┌─────────────────────────┐   │
│  │ EnVariational    │  │ ConditionalDDPM │  │ Metrics & Visualization │   │
│  │ Diffusion        │  │ (for conditional)   │ - BasicMolecularMetrics │   │
│  │ (for joint)      │  │                 │  │ - MoleculeProperties    │   │
│  └──────────────────┘  └─────────────────┘  │ - CategoricalDistrib    │   │
│         ↓                       ↓             └─────────────────────────┘   │
│  ┌──────────────────────────────────────┐                                   │
│  │    EGNNDynamics (Denoising Network)  │                                   │
│  │  ┌────────────────────────────────┐  │                                   │
│  │  │ Atom/Residue Encoders (MLPs)   │  │                                   │
│  │  ├────────────────────────────────┤  │                                   │
│  │  │ EGNN (Graph Neural Network)    │  │  (EGNNDynamics predicts          │
│  │  │ - Message passing               │  │   noise to remove at each        │
│  │  │ - Equivariant coordinate update │  │   diffusion step)               │
│  │  ├────────────────────────────────┤  │                                   │
│  │  │ Atom/Residue Decoders (MLPs)   │  │                                   │
│  │  └────────────────────────────────┘  │                                   │
│  └──────────────────────────────────────┘                                   │
│                                                                              │
│  Loss Computation & Metrics                                                │
│  ├─ L2 loss: ||ε_pred - ε_true||²                                          │
│  ├─ VLB (variational lower bound)                                          │
│  ├─ Auxiliary Lennard-Jones penalty                                        │
│  └─ Evaluation metrics on validation set                                   │
└─────────────────────────────────────────────────────────────────────────────┘
         │
         ↓
    Checkpoint saved to WandB
    
    
                            INFERENCE PATH
                            
┌─────────────────────────┐
│   Checkpoint (.ckpt)    │
└─────────────────────────┘
         │
         ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ generate_ligands() / inpaint() / optimize()                                 │
│  ├─ Load model from checkpoint                                              │
│  ├─ Extract pocket from PDB (or use residue list)                          │
│  ├─ Sample ligand size from DistributionNodes                              │
│  └─ Run reverse diffusion: z_T → z_0                                       │
└─────────────────────────────────────────────────────────────────────────────┘
         │
         ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ Post-Processing: molecule_builder.py                                        │
│  ├─ Extract coordinates and atom types from z_0                            │
│  ├─ Infer bonds via OpenBabel (robust)                                     │
│  ├─ Create RDKit Mol objects with conformer                                │
│  └─ Apply molecule_builder filters                                         │
└─────────────────────────────────────────────────────────────────────────────┘
         │
         ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ Filtering & Validation: process_molecule()                                 │
│  ├─ Sanitize (fix valencies)                                               │
│  ├─ Remove hydrogens (if needed)                                           │
│  ├─ Keep largest fragment                                                  │
│  ├─ UFF force field relaxation                                             │
│  └─ Filter 3-3 ring intersections                                          │
└─────────────────────────────────────────────────────────────────────────────┘
         │
         ↓
┌──────────────────────────────────┐
│   Output: RDKit Molecules        │
│   (SDF file or for evaluation)   │
└──────────────────────────────────┘
```

---

## 2. Diffusion Model Forward & Reverse Process

```
FORWARD PROCESS (Training Data → Noise)
─────────────────────────────────────

Clean molecule x_0
      │
      ├─→ Add small noise → x_1 (still mostly signal)
      │
      ├─→ Add more noise → x_2
      │
      ├─→ ...
      │
      └─→ Add lots of noise → x_T (mostly noise)

Mathematical:
  z_t = α_t * x_0 + σ_t * ε,  where ε ~ N(0, I)
  α_t = sqrt(1 - σ_t²)
  
  Noise schedule γ_t = log(α_t²/σ_t²)
  γ_0 > γ_1 > ... > γ_T  (monotonic decreasing)


REVERSE PROCESS (Noise → Clean Molecule)
─────────────────────────────────────

Start with noise z_T ~ N(0, I)
      │
      ├─→ Denoise step 1: EGNNDynamics predicts ε_T
      ├─→ Sample z_{T-1} from p(z_{T-1}|z_T)
      │
      ├─→ Denoise step 2: EGNNDynamics predicts ε_{T-1}
      ├─→ Sample z_{T-2} from p(z_{T-2}|z_{T-1})
      │
      ├─→ ...
      │
      └─→ Denoise step T: EGNNDynamics predicts ε_1
          Sample z_0 from p(z_0|z_1) → Final molecule


TRAINING OBJECTIVE
─────────────────

For each training step:
  1. Sample clean molecule x_0 from dataset
  2. Sample random timestep t ∈ {1, ..., T}
  3. Sample noise ε ~ N(0, I)
  4. Compute z_t = α_t * x_0 + σ_t * ε
  5. Run EGNNDynamics: ε̂ = ε_θ(z_t, t)
  6. Compute loss: L = ||ε̂ - ε||²  (or VLB variant)
  7. Backprop & update θ


SAMPLING AT INFERENCE
─────────────────────

For each sample:
  1. Sample z_T ~ N(0, I) with center-of-mass = 0
  2. For t = T, T-1, ..., 1:
     a. Run EGNNDynamics to predict ε_t(z_t)
     b. Compute mean of p(z_{t-1}|z_t) using ε_t
     c. Sample z_{t-1} from p(z_{t-1}|z_t)
     d. Enforce center-of-mass = 0
  3. Extract z_0:
     x_0 = (z_0 - σ_0*ε̂) / α_0  (coordinate)
     h_0 = softmax(z_0[3:])      (atom types)
  4. Build RDKit molecule from x_0 and argmax(h_0)
```

---

## 3. EGNNDynamics Architecture

```
INPUT REPRESENTATION
────────────────────
Concatenated graph with:
  - Ligand atoms: coordinates x_i, one-hot h_i
  - Pocket residues: coordinates x_j, one-hot h_j
  - Batch mask indicating which atoms belong to which sample
  - Timestep t (diffusion step being reversed)


ARCHITECTURE FLOW
─────────────────

Input: [Ligand + Pocket atoms/residues, t]
  │
  ├─ Ligand atoms x_atoms ──→ Atom Encoder MLP ──→ h_atoms' (joint_nf)
  │
  ├─ Pocket residues x_residues ──→ Residue Encoder MLP ──→ h_residues' (joint_nf)
  │
  ├─ Concatenate time embedding:
  │   h_atoms'' = [h_atoms', t_embed]       (dims: joint_nf + 1)
  │   h_residues'' = [h_residues', t_embed] (dims: joint_nf + 1)
  │
  ├─ Pass through EGNN with multiple layers
  │   ├─ Layer 1:  h, x ──→ h_new, x_new (equivariant)
  │   ├─ Layer 2:  h_new, x_new ──→ h_new2, x_new2
  │   └─ Layer k:  ...
  │
  ├─ Atom Decoder MLP:  h_final[atoms] ──→ ε_x_atoms, ε_h_atoms
  │
  ├─ Residue Decoder MLP:  h_final[residues] ──→ ε_x_residues, ε_h_residues
  │
  └─ Output: [ε_x_atoms, ε_h_atoms, ε_x_residues, ε_h_residues]

Output goes back to diffusion model for reverse step


KEY PROPERTIES
──────────────
✓ Coordinates x are EQUIVARIANT:
  - If input rotates, output coordinates rotate the same way
  - If input translates, output coordinates translate the same way
  
✓ Features h are INVARIANT:
  - Don't depend on absolute coordinate values, only relative geometry
  
✓ Respects SE(3) symmetry - crucial for 3D generation

✓ Supports optional features:
  - Edge cutoffs (limit distance-based interactions)
  - Edge embeddings (bond type information)
  - Attention mechanisms
  - Various activation functions (SiLU, Tanh, etc.)
```

---

## 4. Data Representation & Batching

```
SINGLE SAMPLE (NPZ DICTIONARY)
──────────────────────────────

{
  'lig_coords':      shape (N_atoms, 3),          float32  # Ligand coordinates
  'lig_one_hot':     shape (N_atoms, 10),         float32  # Ligand atom types
  'lig_mask':        shape (N_atoms,),            int64    # Atom indices
  
  'pocket_coords':   shape (N_residues, 3),       float32  # Pocket coords
  'pocket_one_hot':  shape (N_residues, 20),      float32  # Residue types
  'pocket_mask':     shape (N_residues,),         int64    # Residue indices
  
  'names':           shape (),                    str      # Mol identifier
  'receptors':       shape (),                    str      # Protein name
}


BATCH ASSEMBLY (via collate_fn)
───────────────────────────────

Input: [sample_1, sample_2, sample_3]
       sample_1: 5 atoms,  3 residues
       sample_2: 4 atoms,  2 residues
       sample_3: 6 atoms,  4 residues

After collate_fn:

{
  'lig_coords':      shape (15, 3)  # Concatenated: 5+4+6
  'lig_one_hot':     shape (15, 10)
  'lig_mask':        [0,0,0,0,0, 1,1,1,1, 2,2,2,2,2,2]  ← batch indices
  
  'pocket_coords':   shape (9, 3)   # Concatenated: 3+2+4
  'pocket_one_hot':  shape (9, 20)
  'pocket_mask':     [0,0,0, 1,1, 2,2,2,2]  ← batch indices
  
  'names':           [str1, str2, str3]
  'receptors':       [str1, str2, str3]
}

WHY MASKS?
──────────
torch_scatter operations need batch indices to aggregate per-sample

Example: scatter_mean(data, mask, dim=0) 
  Computes mean for each unique mask value separately:
  - mask=0: mean(data[0:5])
  - mask=1: mean(data[5:9])
  - mask=2: mean(data[9:15])
```

---

## 5. Loss Computation Pipeline

```
FORWARD PASS (Lines 274-356 in lightning_modules.py)
──────────────────────────────────────────────────

ligand = {
  'x': lig_coords,    # (n_atoms, 3)
  'one_hot': lig_one_hot,  # (n_atoms, atom_nf)
  'size': num_lig_atoms,   # (batch_size,)
  'mask': lig_mask     # (n_atoms,) batch indices
}

pocket = {
  'x': pocket_coords,  # (n_residues, 3)
  'one_hot': pocket_one_hot,  # (n_residues, residue_nf)
  'size': num_pocket_nodes,   # (batch_size,)
  'mask': pocket_mask  # (n_residues,) batch indices
}

Call: nll, info = self.ddpm(ligand, pocket, return_info=True)
  │
  ├─ Inside ddpm.forward():
  │   1. Sample t ~ Uniform(1, T)
  │   2. Sample ε ~ N(0, I) with center-of-mass = 0
  │   3. Compute z_t = α_t*[x,h] + σ_t*ε
  │   4. Call dynamics(z_t, t) → ε_pred
  │   5. Compute various loss terms:
  │      - error_t_lig = ||ε_pred[atoms] - ε[atoms]||²
  │      - error_t_pocket = ||ε_pred[residues] - ε[residues]||²
  │      - loss_0_x_ligand, loss_0_x_pocket, loss_0_h (reconstruction)
  │      - kl_prior (KL divergence of prior)
  │      - SNR_weight (signal-to-noise ratio)
  │
  └─ Returns: (nll, info_dict)


LOSS ASSEMBLY (L2 Mode - Training)
─────────────────────────────────

denom_lig = 3 * actual_ligand_size + atom_nf * ligand_size
denom_pocket = (3 + residue_nf) * pocket_size

loss_t = 0.5 * (error_t_lig / denom_lig + error_t_pocket / denom_pocket)
  ↑ Normalized per atom/residue

loss_0 = loss_0_x_ligand / (3 * actual_ligand_size) +
         loss_0_x_pocket / (3 * pocket_size) +
         loss_0_h

nll = loss_t + loss_0 + kl_prior
  ↑ Total negative log-likelihood for training


AUXILIARY LOSS (Optional)
─────────────────────────

When auxiliary_loss=True:
  
  x_lig_hat = xh_lig_hat[:, :3]
  h_lig_hat = xh_lig_hat[:, 3:]
  
  # Lennard-Jones potential (penalizes clashes)
  lj_potential = compute_lj_potential(x_lig_hat, h_lig_hat, mask)
  
  # Weight schedule: strong early, weak at t=0
  weight = auxiliary_weight_schedule(t)
  
  weighted_lj = weight * lj_potential
  
  nll = nll + weighted_lj  ← adds clash penalty


GRADIENT CLIPPING (Optional)
────────────────────────────

if clip_grad:
  recent_grad_norms = [norm_batch_1, norm_batch_2, ...]
  
  max_grad_norm = 1.5 * mean(recent_norms) + 2 * std(recent_norms)
  
  torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
  
  ← Prevents exploding gradients during training
```

---

## 6. Inference / Sampling Process

```
STEP 1: LOAD CHECKPOINT & MODEL
────────────────────────────────

model = LigandPocketDDPM.load_from_checkpoint(
    checkpoint_path,
    map_location=device
)
model = model.to(device)
model.eval()  ← Set to evaluation mode (no dropout, etc.)


STEP 2: DEFINE POCKET
──────────────────────

Option A: From residue list
  pocket_residues = [
    pdb_struct['A'][(' ', 1, ' ')],
    pdb_struct['A'][(' ', 2, ' ')],
    ...
  ]

Option B: From reference ligand in PDB
  pocket_residues = get_pocket_from_ligand(
    pdb_struct, 
    ref_ligand='A:330',  # chain:residue_number
    dist_cutoff=8.0
  )
  → All residues within 8Å of ligand atoms

Option C: From external SDF file
  pocket_residues = get_pocket_from_ligand(
    pdb_struct,
    ref_ligand='path/to/ref.sdf'  ← External coords
  )


STEP 3: PREPARE POCKET
───────────────────────

pocket = model.prepare_pocket(
    pocket_residues,
    repeats=n_samples  ← Repeat for batch generation
)

Result:
  pocket = {
    'x': torch.Tensor (n_residues * n_samples, 3),
    'one_hot': torch.Tensor (n_residues * n_samples, residue_nf),
    'size': torch.Tensor (n_samples,),  ← Residues per sample
    'mask': torch.Tensor (n_residues * n_samples,),
  }


STEP 4: SAMPLE LIGAND SIZE
───────────────────────────

num_nodes_lig = model.ddpm.size_distribution.sample_conditional(
    n1=None,  ← No conditioning on input ligand size
    n2=pocket['size']  ← Condition on pocket size
)

Sample from learned P(N_ligand | N_pocket) distribution


STEP 5: RUN REVERSE DIFFUSION
───────────────────────────────

(Inside ddpm.sample_given_pocket)

Initialize:
  z_T ~ N(0, I) with center-of-mass = 0
  batch_mask setup

For t in range(T, 0, -1):  # T=500 typically
  
  1. Predict noise:
     eps_pred = dynamics(z_t, t, masks)
  
  2. Compute denoising parameters:
     alpha_t, sigma_t, sigma_t_given_prev
  
  3. Compute posterior mean:
     mu = (z_t - sigma_t * eps_pred / sigma_t) / alpha_t
  
  4. Sample next state:
     z_{t-1} ~ N(mu, sigma_{t|t-1}²)
  
  5. Enforce constraints:
     z_{t-1} = remove_mean_batch(z_{t-1}, mask)

Result: z_0 contains [coordinates, features] of generated ligand


STEP 6: EXTRACT GENERATED MOLECULES
────────────────────────────────────

x = z_0[:, :3]        # Coordinates
h = z_0[:, 3:]        # Feature logits

atom_type = argmax(h)  ← One-hot → single index

Split by lig_mask:
  molecules_batch = [
    (coords_1, types_1),
    (coords_2, types_2),
    ...
  ]


STEP 7: BUILD RDKit MOLECULES
──────────────────────────────

For each (coords, atom_types):
  
  1. Call build_molecule(coords, atom_types, dataset_info)
     ├─ Use OpenBabel to infer bonds from coordinates
     ├─ Create RDKit RWMol
     ├─ Add all atoms & bonds
     └─ Add 3D conformer
  
  2. Call process_molecule():
     ├─ Sanitize (fix valencies)
     ├─ Keep largest connected fragment
     ├─ UFF force field optimization (optional)
     ├─ Filter invalid rings
     └─ Return RDKit Mol or None


STEP 8: OUTPUT
───────────────

molecules = [valid_mol_1, valid_mol_2, ...]

Write to SDF:
  write_sdf_file('output.sdf', molecules)

Or use for further analysis:
  - Compute metrics (validity, uniqueness, novelty, docking)
  - Visualize structures
  - Prepare for experimental validation
```

---

## 7. Configuration Hierarchy

```
YAML Config File
  ├─ Paths
  │  ├─ run_name: "experiment_id"
  │  ├─ logdir: "logs/"
  │  └─ datadir: "data/processed_..."
  │
  ├─ Dataset & Model Selection
  │  ├─ dataset: "crossdock" or "bindingmoad"
  │  ├─ mode: "pocket_conditioning" or "joint"
  │  ├─ pocket_representation: "CA" or "full-atom"
  │  └─ virtual_nodes: true/false
  │
  ├─ Training Basics
  │  ├─ batch_size: 16 (adjust per GPU memory)
  │  ├─ lr: 1.0e-3 (learning rate)
  │  ├─ n_epochs: 1000
  │  ├─ gpus: 4 (number of GPUs)
  │  ├─ num_workers: 4 (data loading workers)
  │  ├─ clip_grad: true (gradient clipping)
  │  └─ accumulate_grad_batches: 1
  │
  ├─ EGNN Architecture (egnn_params)
  │  ├─ device: "cuda"
  │  ├─ joint_nf: 128  (shared feature dim)
  │  ├─ hidden_nf: 256  (EGNN hidden layers)
  │  ├─ n_layers: 6  (EGNN depth)
  │  ├─ attention: true
  │  ├─ tanh: true
  │  ├─ norm_constant: 1
  │  ├─ inv_sublayers: 1
  │  ├─ sin_embedding: false
  │  ├─ aggregation_method: "sum"
  │  ├─ normalization_factor: 100
  │  ├─ edge_cutoff_ligand: null  (no cutoff within ligand)
  │  ├─ edge_cutoff_pocket: 5.0  (5 Å interactions)
  │  └─ edge_cutoff_interaction: 5.0  (5 Å pocket-ligand)
  │
  ├─ Diffusion Model (diffusion_params)
  │  ├─ diffusion_steps: 500  (T value)
  │  ├─ diffusion_noise_schedule: "polynomial_2"
  │  │  (options: learned, polynomial_2, cosine, linear, sqrt)
  │  ├─ diffusion_noise_precision: 5.0e-4
  │  ├─ diffusion_loss_type: "l2"  (or "vlb")
  │  └─ normalize_factors: [1, 4]  (norm_x, norm_h)
  │
  ├─ Auxiliary Loss (loss_params) [optional]
  │  ├─ max_weight: 0.001
  │  ├─ schedule: "linear"  (weight decay schedule)
  │  └─ clamp_lj: 3.0  (max LJ potential value)
  │
  ├─ Evaluation (eval_params)
  │  ├─ n_eval_samples: 100  (sample count for metrics)
  │  ├─ eval_batch_size: 100
  │  ├─ smiles_file: "path/to/train_smiles.npy"  (for novelty)
  │  ├─ n_visualize_samples: 5
  │  └─ keep_frames: 100  (diffusion trajectory frames)
  │
  ├─ Evaluation Schedule
  │  ├─ eval_epochs: 50  (compute metrics every N epochs)
  │  ├─ visualize_sample_epoch: 50  (save samples every N)
  │  └─ visualize_chain_epoch: 50  (save trajectory every N)
  │
  ├─ WandB Logging (wandb_params)
  │  ├─ mode: "online"  (offline, online, disabled)
  │  ├─ entity: "your_username"
  │  └─ group: "crossdock"  (experiment group)
  │
  └─ Other Flags
     ├─ augment_noise: 0  (gaussian noise augmentation)
     ├─ augment_rotation: false  (random rotation aug)
     ├─ auxiliary_loss: false  (LJ penalty)
     ├─ enable_progress_bar: true
     └─ num_sanity_val_steps: 0
```

---

## 8. Key Classes Dependency Graph

```
pytorch_lightning.LightningModule
    ↑
    │
LigandPocketDDPM ◄───────────────────────────┐
    │                                         │
    ├─→ EnVariationalDiffusion ◄─────────────┼──────┐
    │   (or ConditionalDDPM)                 │      │
    │       │                                 │      │
    │       ├─→ EGNNDynamics ◄───────────────┤───┐  │
    │       │       │                        │   │  │
    │       │       └─→ EGNN (GNN layers)    │   │  │
    │       │                                │   │  │
    │       ├─→ GammaNetwork                 │   │  │
    │       │   (or PredefinedNoiseSchedule) │   │  │
    │       │                                │   │  │
    │       └─→ DistributionNodes ◄─────────┤───┼──┐
    │                                        │   │  │
    ├─→ ProcessedLigandPocketDataset ◄─────┘   │  │
    │   (dataset loading & batching)           │  │
    │                                          │  │
    ├─→ BasicMolecularMetrics ◄────────────────┘  │
    │   (validity, connectivity, etc.)             │
    │                                              │
    ├─→ MoleculeProperties                        │
    │   (QED, SA Score, LogP)                     │
    │                                              │
    ├─→ CategoricalDistribution ◄────────────────┘
    │   (atom type distribution tracking)
    │
    └─→ AppendVirtualNodes (optional transform)


Other key modules NOT in LigandPocketDDPM:

build_molecule() ◄─── molecule_builder.py
  ├─→ OpenBabel (bond inference)
  └─→ RDKit (molecule object creation)

process_molecule()
  └─→ RDKit (sanitization, filtering, relaxation)

generate_ligands (entry point)
  └─→ LigandPocketDDPM.generate_ligands()

inpaint (entry point)
  └─→ EnVariationalDiffusion.inpaint()

optimize (entry point)
  └─→ Evolutionary algorithm with inpainting
```

---

## 9. Noise Schedule Visualization

```
TIMESTEP vs NOISE LEVEL
────────────────────────

σ_t² (variance added)
  ↑
  │                                    ═════════════════  z_T (mostly noise)
  │                            ═════════
  │                      ═════════
  │                ═════════
  │          ═════════
  │    ═════════
  │  ═════════  
  │═══════════════════════════════════════════════────→ t (timestep)
  0                                                    T

α_t² (signal retained)
  ↑
  │═══════════════════════════════════════════════─────  x_0 (mostly signal)
  │    ═════════
  │        ═════════
  │            ═════════
  │                ═════════
  │                    ═════════
  │                        ═════════════════════════════  z_T
  │═══════════════════════════════════════════════─────
  0                                                    T


TRAINING: Minimize noise prediction error at all timesteps
INFERENCE: Start from noise, progressively denoise

High t: Denoising is easier (more noise, high SNR)
Low t: Denoising is harder (less noise, low SNR)
```

---

## 10. Memory & Computation Flow

```
TRAINING MEMORY USAGE
─────────────────────

Batch_size = 16 samples

Per sample:
  - Ligand: 10-40 atoms  → 30-120 coordinates (120-480 floats)
  - Pocket: 50-200 atoms/residues → 150-600 features (150-600 floats)

Batch (16 samples):
  ≈ (30-120 + 150-600) * 16 * 4 bytes
  ≈ 6-37 MB raw data

Model parameters:
  EGNN 6 layers, hidden_nf=256: ≈ 5-10 million params
  = 20-40 MB weights + 20-40 MB gradients

Total typical GPU memory: 8-16 GB (RTX 3090 / A100)

Gradient accumulation: Can use smaller batch_size, accumulate_grad_batches=4


INFERENCE MEMORY USAGE
──────────────────────

Model: 20-40 MB
Single sample inference: ≈ 100 MB (allocations during denoising)
Can generate 100+ samples in batch with 8 GB VRAM


COMPUTATION TIME
─────────────────

Training (RTX 3090):
  - 1 epoch ≈ 5-10 minutes (depends on dataset size)
  - 1000 epochs ≈ 5-10 days
  
Inference (RTX 3090):
  - 500 denoising steps × 16 samples: ≈ 2-5 minutes per batch
  - 20 samples = ≈ 30 seconds
```

---

**End of Architecture Diagrams**
