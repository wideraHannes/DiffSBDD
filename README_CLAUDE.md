# DiffSBDD Documentation for Claude Code

This directory contains comprehensive documentation about the DiffSBDD codebase structure and architecture.

## Documentation Files

### 1. **CLAUDE.md** (908 lines, 25KB)
The primary, comprehensive architecture guide covering:
- Executive summary of what DiffSBDD does
- Overall architecture patterns
- Detailed breakdown of all key modules (lightning_modules, diffusion models, denoising network, EGNN)
- Data processing pipeline
- Training flow with loss computation
- Inference/generation flow
- Configuration system
- Important implementation details (virtual nodes, masks, constraints)
- Common workflows and entry points
- Debugging checklist
- Quick API reference

**Best for:** Complete understanding of the codebase, understanding how components fit together, implementation details

---

### 2. **ARCHITECTURE_SUMMARY.txt** (345 lines, 14KB)
A quick reference guide with:
- What is DiffSBDD in plain English
- Main components breakdown (6 sections)
- Data flow diagrams (training & inference)
- Key training concepts
- Configuration system overview
- Quick start commands
- Debugging checklist
- Important implementation notes
- File size reference
- Paper reference

**Best for:** Quick lookups, understanding the big picture, finding specific information, debugging

---

### 3. **ARCHITECTURE_DIAGRAMS.md** (744 lines, 29KB)
Visual guides and detailed flowcharts:
- High-level system architecture (training & inference paths)
- Diffusion model forward & reverse process
- EGNNDynamics architecture flow
- Data representation & batching
- Loss computation pipeline
- Complete inference/sampling process (8 steps)
- Configuration hierarchy
- Class dependency graph
- Noise schedule visualization
- Memory & computation notes

**Best for:** Visual learners, understanding data flow, understanding process pipelines, referencing specific algorithms

---

## Quick Navigation

### I want to understand...

- **What the model does** → Read: CLAUDE.md Section 1 + ARCHITECTURE_SUMMARY.txt "WHAT IS DiffSBDD?"
- **How to train the model** → Read: CLAUDE.md Section 3 + ARCHITECTURE_DIAGRAMS.md Section 5
- **How to generate molecules** → Read: CLAUDE.md Section 4 + ARCHITECTURE_DIAGRAMS.md Section 6
- **The data format** → Read: CLAUDE.md Section 5 + ARCHITECTURE_DIAGRAMS.md Section 4
- **Each module's responsibility** → Read: CLAUDE.md Section 2
- **The diffusion process** → Read: ARCHITECTURE_DIAGRAMS.md Section 2
- **EGNN architecture** → Read: ARCHITECTURE_DIAGRAMS.md Section 3
- **How to configure training** → Read: CLAUDE.md Section 7 + ARCHITECTURE_DIAGRAMS.md Section 9
- **How to debug issues** → Read: ARCHITECTURE_SUMMARY.txt "DEBUGGING CHECKLIST"
- **Quick commands** → Read: ARCHITECTURE_SUMMARY.txt "QUICK START COMMANDS"
- **File structure** → Read: ARCHITECTURE_SUMMARY.txt "FILE SIZE REFERENCE"

---

## File Organization

```
DiffSBDD/
├─ README_CLAUDE.md (this file - navigation guide)
├─ CLAUDE.md (comprehensive architecture guide - 908 lines)
├─ ARCHITECTURE_SUMMARY.txt (quick reference - 345 lines)
├─ ARCHITECTURE_DIAGRAMS.md (visual guides - 744 lines)
│
├─ lightning_modules.py (1079 lines - training engine)
├─ train.py (139 lines - training entry point)
├─ dataset.py (71 lines - data loading)
├─ utils.py (235 lines - utilities)
├─ constants.py (463 lines - bond tables, dataset params)
│
├─ equivariant_diffusion/
│  ├─ en_diffusion.py (1190 lines - diffusion model core)
│  ├─ conditional_model.py (~500 lines - conditional variant)
│  ├─ dynamics.py (278 lines - EGNN-based denoising network)
│  └─ egnn_new.py (511 lines - equivariant graph neural network)
│
├─ analysis/
│  ├─ metrics.py (245 lines - evaluation metrics)
│  ├─ molecule_builder.py (269 lines - bond prediction, RDKit)
│  ├─ visualization.py (452 lines - visualization)
│  ├─ docking.py (247 lines - docking scores)
│  └─ SA_Score/ (synthetic accessibility scoring)
│
├─ configs/ (8 YAML configuration files)
│  ├─ crossdock_fullatom_cond.yml (conditional full-atom)
│  ├─ crossdock_fullatom_joint.yml (joint full-atom)
│  ├─ crossdock_ca_cond.yml (conditional CA-only)
│  ├─ crossdock_ca_joint.yml (joint CA-only)
│  ├─ moad_*.yml (similar variants for Binding MOAD)
│  └─ paths.py (path configuration)
│
├─ generate_ligands.py (61 lines - inference entry point)
├─ inpaint.py (298 lines - substructure inpainting)
├─ optimize.py (338 lines - molecular optimization)
│
├─ process_crossdock.py (25KB - data preparation for CrossDock)
├─ process_bindingmoad.py (25KB - data preparation for MOAD)
│
├─ checkpoints/ (pre-trained model checkpoints)
├─ data/ (processed training data)
├─ logs/ (training logs and checkpoints)
└─ results/ (inference results)
```

---

## Key Concepts Glossary

### Equivariance
The model respects rotations and translations. If you rotate the input protein and ligand, the output rotates the same way. This is crucial for 3D molecular generation.

### Diffusion
A process that learns to remove noise from data. Training: clean molecule → noisy molecule. Inference: noise → clean molecule.

### EGNN (Equivariant Graph Neural Network)
A neural network that processes molecular graphs while maintaining equivariance to 3D rotations/translations.

### Conditional Generation
Generating ligands given a fixed pocket. Only the ligand is sampled; pocket coordinates stay fixed.

### Joint Generation
Generating both ligand and pocket together. More complex but can handle pocket flexibility.

### Virtual Nodes
Dummy atoms added to ligands to make them all the same size. Enables fixed tensor shapes for efficient batching.

### Batch Masks
Indices indicating which atoms belong to which molecule in a batch. Essential for torch_scatter operations.

### NPZ Files
NumPy archive files containing processed training data (coordinates, atom types, masks).

---

## Architecture Overview (One Sentence Each)

- **LigandPocketDDPM**: PyTorch Lightning module orchestrating training and inference
- **EnVariationalDiffusion**: Core diffusion model implementing forward/reverse processes
- **ConditionalDDPM**: Variant of diffusion model for conditional generation (pocket fixed)
- **EGNNDynamics**: Denoising network that predicts noise to remove at each step
- **EGNN**: Equivariant graph neural network preserving 3D symmetries
- **ProcessedLigandPocketDataset**: Loads and batches pre-processed NPZ files
- **BasicMolecularMetrics**: Evaluates generated molecules (validity, uniqueness, novelty)
- **MoleculeProperties**: Computes drug-relevant properties (QED, SA Score, etc.)
- **molecule_builder.py**: Converts generated coordinates/atoms to RDKit molecules with bonds
- **DistributionNodes**: Models the distribution of ligand sizes given pocket size

---

## Training vs Inference

### Training
1. Load data from NPZ files
2. For each batch:
   - Sample timestep t
   - Add noise to create z_t
   - Predict noise with EGNNDynamics
   - Compute loss
   - Backpropagate
3. Save checkpoint

### Inference
1. Load checkpoint
2. Define pocket (from PDB, residue list, or SDF)
3. Sample ligand size
4. Initialize noise z_T
5. Loop: denoise with EGNNDynamics T→1
6. Extract coordinates and atom types
7. Build RDKit molecules with inferred bonds
8. Filter and return

---

## Common Questions

**Q: What's the difference between CA and full-atom?**
A: CA uses only Cα atoms (~20 types), full-atom uses all atoms (~10 types). CA is simpler, full-atom is more detailed.

**Q: What's the difference between joint and conditional?**
A: Joint: both ligand and pocket are generated. Conditional: pocket fixed, only ligand generated.

**Q: Why do we need masks?**
A: Masks indicate which atoms belong to which sample in a batch. They enable safe per-sample aggregation.

**Q: What are virtual nodes?**
A: Dummy atoms padding ligands to a fixed size. They improve training efficiency by enabling static tensor shapes.

**Q: How long does training take?**
A: ~5-10 minutes per epoch on RTX 3090, ~1000 epochs = 5-10 days total.

**Q: How long does inference take?**
A: ~2-5 minutes for 16 samples (500 denoising steps) on RTX 3090.

**Q: What's OpenBabel for?**
A: Inferring bonds from 3D coordinates. More robust than direct distance tables.

**Q: What's UFF relaxation?**
A: Force field optimization to fix unrealistic 3D structures (clashes, bad angles).

**Q: Can I use CPU?**
A: Yes, but very slow (~10x slower than GPU). GPU strongly recommended.

---

## For Different Use Cases

### I'm a researcher wanting to use pre-trained models
- Start with: ARCHITECTURE_SUMMARY.txt "QUICK START COMMANDS"
- Then read: CLAUDE.md Section 4 "Inference / Generation Flow"
- Refer to: generate_ligands.py for examples

### I'm a student wanting to understand the model
- Start with: CLAUDE.md Section 1 "Overall Architecture"
- Continue with: ARCHITECTURE_DIAGRAMS.md
- Deep dive: CLAUDE.md Sections 2-6

### I'm implementing a feature or fixing a bug
- Check: CLAUDE.md "Key Abstractions & Their Relationships"
- Understand data flow: ARCHITECTURE_DIAGRAMS.md "Data Representation & Batching"
- Locate the relevant module in: CLAUDE.md Section 2 "Key Modules"
- Debug using: ARCHITECTURE_SUMMARY.txt "DEBUGGING CHECKLIST"

### I'm tuning hyperparameters
- Read: CLAUDE.md Section 7 "Configuration System"
- Check: ARCHITECTURE_DIAGRAMS.md Section 9 "Configuration Hierarchy"
- Monitor: ARCHITECTURE_SUMMARY.txt "KEY TRAINING CONCEPTS"

---

## Related Resources

### Paper
Schneuing et al., "Structure-based drug design with equivariant diffusion models"
Nature Computational Science, 2024
https://doi.org/10.1038/s43588-024-00737-x

### Original Repository
https://github.com/arneschneuing/DiffSBDD

### Key Libraries
- PyTorch & PyTorch Lightning (training framework)
- RDKit (molecular operations)
- OpenBabel (bond inference)
- torch_scatter (efficient aggregation)
- BioPython (protein parsing)

---

## Version Info

- Documentation created: November 13, 2025
- Codebase analyzed: DiffSBDD (Zenodo)
- Total lines analyzed: ~6000 core logic lines
- Documentation size: ~2000 lines across 3 files

---

## How to Use These Docs

1. **First time?** Read ARCHITECTURE_SUMMARY.txt top to bottom
2. **Need details?** Open CLAUDE.md and search for your topic
3. **Visual learner?** Jump to ARCHITECTURE_DIAGRAMS.md
4. **Stuck debugging?** Use ARCHITECTURE_SUMMARY.txt "DEBUGGING CHECKLIST"
5. **Want to understand a module?** Find it in CLAUDE.md Section 2

---

**Last Updated**: November 13, 2025  
**Documentation Completeness**: Comprehensive coverage of all major components, data flows, and training/inference processes

