# ESM-C Integration into DiffSBDD: Lean Thesis Plan

**Author**: Hannes Widera
**Supervisor**: Senior AI Scientist (20 years molecular modeling)
**Goal**: Fast prototype to validate if ESM-C protein embeddings improve ligand generation quality

---

## Philosophy: Fast Validation, Not Broad Exploration

**Core Hypothesis**: Rich ESM-C protein embeddings (960-dim) will outperform one-hot amino acid encodings (20-dim) by capturing evolutionary/structural context.

**Success Metric**: ANY statistically significant improvement in validity, QED, SA score, or Vina docking score.

**Timeline**: 6-8 weeks to thesis-ready results

---

## Phase 0: Reality Check (Week 1)

### 0.1 Use Existing Baseline - NO RETRAINING

**Critical Insight from Paper**:
- Authors already trained and published model weights (Zenodo: `10.5281/zenodo.8183747`)
- CrossDocked full-atom conditional model exists: `crossdocked_fullatom_cond.ckpt` (you have this!)
- Binding MOAD model also available

**Action**:
✅ **Use pre-trained checkpoint as baseline** - saves 2-3 weeks of training time
- Evaluate baseline on test set: `python generate_ligands.py checkpoints/crossdocked_fullatom_cond.ckpt`
- Document baseline metrics from existing checkpoint
- No need to retrain - focus energy on ESM-C integration

**Baseline Metrics to Record** (from test set evaluation):
- Validity (% chemically valid molecules)
- Uniqueness (% unique SMILES)
- QED (drug-likeness)
- SA Score (synthetic accessibility)
- Vina docking score distribution
- Inference time per molecule

### 0.2 Computational Resources Assessment

**HPC at HHU**:
- PBS job system
- Storage: 10TB project space, 15TB scratch
- **TODO**: Find GPU specs (email HPC support)
- No internet access (use mirrored repos)

**Estimated ESM-C Model Requirements**:
- Training: ~10-14 days on 4 GPUs (based on baseline)
- Pre-compute embeddings: ~4-8 hours (100K pockets, batch ESM-C inference)
- Inference: Negligible overhead if embeddings pre-computed

**Action Items**:
1. ⏳ SSH into HPC, check GPU availability: `qsub -I -l select=1:ngpus=1 -l walltime=01:00:00`
2. ⏳ Verify dataset location: `ls -lh data/processed_crossdock_noH_full/`
3. ⏳ Profile baseline inference speed on HPC
4. ⏳ Install ESM-C locally: `uv pip install fair-esm`

---

## Phase 1: ESM-C Integration Design (Week 1-2)

### 1.1 Critical Architecture Analysis

**Current Pocket Encoding** (from `dynamics.py:38-43`):
```python
# One-hot encoding: 20 amino acid types
residue_nf = 20  # Amino acid one-hot dimension

self.residue_encoder = nn.Sequential(
    nn.Linear(20, 40),      # 20 → 40
    act_fn,
    nn.Linear(40, 128)      # 40 → 128 (joint_nf)
)
```

**ESM-C Replacement**:
```python
# ESM-C embeddings: 960-dim per residue
residue_nf = 960

self.residue_encoder = nn.Sequential(
    nn.Linear(960, 1920),   # 960 → 1920 (or just 960 → 128 directly?)
    act_fn,
    nn.Linear(1920, 128)    # → joint_nf
)
```

### 1.2 Design Decisions (Challenged)

**DECISION 1: Frozen ESM-C (CONFIRMED)**
- ✅ **Correct choice**: ESM-C pre-trained on millions of proteins
- No fine-tuning needed (saves memory + training time)
- Use as feature extractor only

**DECISION 2: Pre-compute embeddings (CONFIRMED)**
- ✅ **Critical for speed**: Pre-compute ALL training/val/test embeddings offline
- Storage: ~76GB for CrossDock (acceptable with 10TB quota)
- Training is same speed as baseline (no ESM-C forward pass)

**DECISION 3: Replace one-hot entirely (CHALLENGED)**

**PROBLEM**: Decoder loss expects `residue_nf` output
```python
# dynamics.py:45-49
self.residue_decoder = nn.Sequential(
    nn.Linear(128, 2 * residue_nf),
    act_fn,
    nn.Linear(2 * residue_nf, residue_nf)  # Must output 960-dim now!
)
```

**Issue**: In conditional mode (`pocket_conditioning`), pocket is FIXED during generation. Why decode pocket at all?

**SOLUTION**: Check if pocket decoder is even used in conditional mode!

**Action**:
⏳ Analyze `conditional_model.py` and `en_diffusion.py` - does decoder loss apply to fixed pocket?

**HYPOTHESIS**: In conditional mode, pocket decoder may be a no-op. If true, we can:
- Replace one-hot → ESM-C in encoder
- Keep decoder output dimension at 960 (but it won't be trained)
- OR: Make decoder output dummy values (since pocket is fixed)

### 1.3 Simplified Architecture (RECOMMENDED)

**Option A: Minimal Change (FASTEST TO TEST)**
```python
class EGNNDynamics(nn.Module):
    def __init__(self, atom_nf, residue_nf=960, use_esmc=True, ...):
        if use_esmc:
            # Direct projection from ESM-C to joint space
            self.residue_encoder = nn.Linear(960, joint_nf)  # Single layer!
            self.residue_decoder = nn.Linear(joint_nf, 960)  # Dummy output
        else:
            # Original one-hot encoding
            self.residue_encoder = nn.Sequential(...)
```

**Why this works**:
- ESM-C already has rich 960-dim features
- No need for 960→1920→128 bottleneck
- Simpler = fewer parameters = faster convergence
- Decoder unused in conditional mode anyway

**Option B: Two-stream (IF Option A fails)**
- Keep one-hot (20-dim) for decoding
- Add ESM-C (960-dim) as auxiliary input
- Concatenate before EGNN: `[one_hot_encoded, esmc_encoded]`

**DECISION**: Start with Option A, fall back to B only if needed.

---

## Phase 2: Fast Prototype Implementation (Week 2-3)

### 2.1 Pre-compute ESM-C Embeddings

**New Script**: `scripts/precompute_esmc.py`

**Workflow**:
```python
import torch
import numpy as np
from esm import pretrained

# Load ESM-C model (check latest version!)
model, alphabet = pretrained.esm2_t33_650M_UR50D()  # Or ESM-C if available
model = model.eval().cuda()

# Load CrossDock data
data = np.load('data/processed_crossdock_noH_full/train.npz', allow_pickle=True)
pocket_one_hot = data['pocket_one_hot']
pocket_masks = data['pocket_mask']

# Extract sequences from one-hot
from constants import dataset_params
aa_decoder = dataset_params['crossdock']['aa_decoder']
sequences = one_hot_to_sequences(pocket_one_hot, aa_decoder)

# Batch ESM-C inference
embeddings = []
for seq_batch in tqdm(batch(sequences, batch_size=32)):
    with torch.no_grad():
        results = model(seq_batch, repr_layers=[33])  # Last layer
        emb = results['representations'][33]  # (batch, len, 960)
        embeddings.append(emb.cpu())

# Save augmented NPZ
np.savez('data/processed_crossdock_esmc/train.npz',
         **data,  # Keep all original fields
         pocket_esmc=embeddings)  # Add ESM-C embeddings
```

**Challenges to Handle**:
1. **Sequence extraction**: Pocket is atoms, not residues
   - If `pocket_representation='full-atom'`: Map atoms → residues → sequence
   - Need residue ID mapping (from PDB processing)
2. **Variable lengths**: ESM-C outputs per-residue embeddings
   - Align ESM-C output with pocket coordinates by residue ID
3. **Missing residues**: Some PDB structures have gaps
   - Use zero-padding or learned dummy embedding

**Action**:
⏳ Check how `process_crossdock.py` handles residues
⏳ Verify pocket representation (CA-only or full-atom)
⏳ Design sequence extraction strategy

**Estimated Time**: 4-8 hours for full CrossDock dataset on GPU

### 2.2 Modify Data Loader

**File**: `dataset.py`

**Changes**:
```python
class ProcessedLigandPocketDataset(Dataset):
    def __init__(self, data_path, use_esmc=False):
        self.data = np.load(data_path, allow_pickle=True)
        self.use_esmc = use_esmc

    def __getitem__(self, idx):
        sample = {
            'lig_coords': ...,
            'lig_one_hot': ...,
            'pocket_coords': ...,
        }

        if self.use_esmc:
            sample['pocket_h'] = self.data['pocket_esmc'][idx]  # 960-dim
        else:
            sample['pocket_h'] = self.data['pocket_one_hot'][idx]  # 20-dim

        return sample
```

**Gotcha**: Ensure batch collation handles variable-dim pocket features!

### 2.3 Modify Model Architecture

**File**: `equivariant_diffusion/dynamics.py`

**Changes** (Option A - Minimal):
```python
class EGNNDynamics(nn.Module):
    def __init__(self, atom_nf, residue_nf, use_esmc=False, ...):
        self.use_esmc = use_esmc

        # Ligand encoder (unchanged)
        self.atom_encoder = nn.Sequential(
            nn.Linear(atom_nf, 2 * atom_nf),
            act_fn,
            nn.Linear(2 * atom_nf, joint_nf)
        )

        # Pocket encoder (CHANGED)
        if use_esmc:
            # Direct projection: 960 → 128
            self.residue_encoder = nn.Linear(960, joint_nf)
        else:
            # Original: 20 → 40 → 128
            self.residue_encoder = nn.Sequential(
                nn.Linear(residue_nf, 2 * residue_nf),
                act_fn,
                nn.Linear(2 * residue_nf, joint_nf)
            )

        # Decoder (may be unused in conditional mode)
        decoder_out_dim = 960 if use_esmc else residue_nf
        self.residue_decoder = nn.Sequential(
            nn.Linear(joint_nf, 2 * decoder_out_dim),
            act_fn,
            nn.Linear(2 * decoder_out_dim, decoder_out_dim)
        )

    def forward(self, xh_atoms, xh_residues, t, mask_atoms, mask_residues):
        # Extract coords and features
        x_residues = xh_residues[:, :3]
        h_residues = xh_residues[:, 3:]  # Now 960-dim if use_esmc=True

        # Encode (works for both 20-dim and 960-dim)
        h_residues = self.residue_encoder(h_residues)

        # Rest unchanged...
```

**File**: `lightning_modules.py`

**Changes**:
```python
class LigandPocketDDPM(pl.LightningModule):
    def __init__(self, ..., use_esmc=False):
        self.use_esmc = use_esmc

        # Initialize dynamics with correct residue_nf
        residue_nf = 960 if use_esmc else len(self.dataset_info['aa_encoder'])

        self.dynamics = EGNNDynamics(
            atom_nf=len(self.dataset_info['atom_encoder']),
            residue_nf=residue_nf,
            use_esmc=use_esmc,
            ...
        )
```

### 2.4 Create ESM-C Config

**File**: `configs/crossdock_fullatom_cond_esmc.yml`

```yaml
run_name: "ESM-C-fast-prototype"
logdir: "logs"
wandb_params:
  mode: "online"
  entity: "your-wandb-username"
  group: "crossdock-esmc"

dataset: "crossdock"
datadir: "data/processed_crossdock_esmc"  # NEW: ESM-C augmented data
use_esmc: True  # NEW FLAG

mode: "pocket_conditioning"
pocket_representation: "full-atom"
virtual_nodes: False

# Reduced batch size (ESM-C embeddings larger)
batch_size: 8  # Reduced from 16
lr: 1.0e-3
n_epochs: 500  # Half of baseline (early stopping if no improvement)
num_workers: 0
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
  diffusion_loss_type: "l2"
  normalize_factors: [1, 4]

eval_epochs: 25  # More frequent eval (catch overfitting early)
visualize_sample_epoch: 50
eval_params:
  n_eval_samples: 100
  eval_batch_size: 50
```

### 2.5 Inference with ESM-C (On-the-fly)

**Challenge**: At inference time, we don't have pre-computed ESM-C for new pockets.

**Solution**: Compute ESM-C on-the-fly in `prepare_pocket()`

**File**: `lightning_modules.py`

```python
def prepare_pocket(self, pdb_file, residues):
    # Extract pocket structure
    pocket_coords, pocket_one_hot = extract_pocket_from_pdb(pdb_file, residues)

    if self.use_esmc:
        # Compute ESM-C embeddings on-the-fly
        sequence = one_hot_to_sequence(pocket_one_hot)
        pocket_h = self.esmc_model(sequence)  # (n_residues, 960)
    else:
        pocket_h = pocket_one_hot

    return {'coords': pocket_coords, 'h': pocket_h}
```

**Optimization**: Cache ESM-C model in `__init__` to avoid reloading.

---

## Phase 3: Training & Evaluation (Week 3-5)

### 3.1 Training Protocol

**Step 1**: Small-scale test (1000 samples, 10 epochs)
```bash
# Create debug config with subset
python train.py --config configs/crossdock_fullatom_cond_esmc.yml
```
- Verify no NaN losses
- Check memory usage (reduce batch size if needed)
- Ensure ESM-C embeddings load correctly

**Step 2**: Full training (100K samples, 500 epochs)
```bash
# Submit to HPC
qsub -A $PROJECT .claude/hpc/train_esmc.pbs
```

**Early Stopping Strategy**:
- Monitor validation loss every 25 epochs
- If no improvement for 100 epochs → stop
- Target: 300-500 epochs (10-12 days)

**Potential Issues**:

| Issue | Symptom | Fix |
|-------|---------|-----|
| Memory overflow | OOM error | Reduce `batch_size` to 4 |
| NaN losses | Loss = NaN after few epochs | Adjust `normalize_factors` to [0.1, 1] |
| Slow convergence | Val loss plateau early | Increase LR to 2e-3 with warmup |
| Decoder dimension mismatch | Shape error in decoder | Verify `residue_nf` passed correctly |

### 3.2 Evaluation Metrics (Comparison to Baseline)

**Molecular Quality** (from paper, Fig 2):
- ✅ Validity: % RDKit-valid molecules (baseline: ~72%)
- ✅ Uniqueness: % unique SMILES (baseline: ~95%)
- ✅ Novelty: % not in training set (baseline: ~98%)
- ✅ QED: Drug-likeness score 0-1 (baseline: ~0.45)
- ✅ SA Score: Synthetic accessibility 1-10 (baseline: ~3.2)

**Binding Affinity**:
- ✅ Vina score: Predicted binding free energy (baseline: mean ~-9.17 on CrossDock)
- Distribution analysis: ESM-C should improve tail (better top-10% binders)

**Similarity**:
- Tanimoto similarity to reference ligand (expect similar or slightly different)

**Inference Speed**:
- Time per molecule (ESM-C should be ~same as baseline if pre-computed)

**Statistical Tests**:
- Wilcoxon signed-rank test for paired comparisons
- p < 0.05 for significance

### 3.3 Evaluation Script

```bash
# Generate 100 molecules per test pocket
python generate_ligands.py checkpoints/esmc_model.ckpt \
    --test_set data/processed_crossdock_esmc/test.npz \
    --n_samples 100 \
    --outdir results/esmc/

# Baseline comparison (use existing checkpoint!)
python generate_ligands.py checkpoints/crossdocked_fullatom_cond.ckpt \
    --test_set data/processed_crossdock_noH_full/test.npz \
    --n_samples 100 \
    --outdir results/baseline/

# Compute metrics
python analyze_results.py \
    --baseline results/baseline/ \
    --esmc results/esmc/ \
    --output comparison.csv
```

### 3.4 Success Criteria

**Minimum Success (Thesis-worthy)**:
- Model trains without crashes
- ANY metric improves by >5% (e.g., validity 72% → 77%)
- Negative results are publishable if methodology is sound

**Good Success**:
- 2+ metrics improve significantly (p < 0.05)
- Top-10% Vina scores better than baseline
- Qualitative examples show improved binding geometry

**Excellent Success**:
- All metrics improve
- Vina scores: mean shifts by >0.5 kcal/mol
- Generated molecules are chemically diverse AND high-affinity

---

## Phase 4: Analysis & Thesis Writing (Week 5-8)

### 4.1 Core Thesis Structure (Lean)

**1. Introduction** (3-4 pages)
- Drug discovery challenges
- Why protein context matters
- ESM-C vs one-hot encoding hypothesis

**2. Background** (5-6 pages)
- Diffusion models (brief, cite DiffSBDD paper)
- Equivariant GNNs (1 page, reference original)
- Protein language models (ESM-2 → ESM-C)
- **Key insight from paper**: "SE(3)-equivariant, not E(3)" (reflection-sensitive)

**3. Methods** (6-8 pages)
- DiffSBDD baseline architecture
- ESM-C integration (Option A design)
- Pre-computation pipeline
- Training protocol
- Evaluation metrics

**4. Results** (8-10 pages)
- **Table 1**: Baseline vs ESM-C metrics
- **Table 2**: Statistical significance tests
- **Figure 1**: Vina score distributions
- **Figure 2**: 3-5 qualitative examples (good vs bad)
- **Figure 3**: Training curves (loss convergence)

**5. Discussion** (4-5 pages)
- What worked / didn't work
- Why ESM-C helps (or doesn't)
- Limitations (data quality, diffusion model capacity)
- Future work (multi-task, active learning, ESM-C fine-tuning)

**6. Conclusion** (1-2 pages)

**Total**: 30-40 pages

### 4.2 Key Figures to Prepare

**Figure 1**: Architecture diagram
- Show residue encoder change (20-dim → 960-dim)
- Highlight where ESM-C enters EGNN

**Figure 2**: Violin plots (like paper Fig 2a)
- Vina score difference (gen - ref)
- Tanimoto similarity
- Side-by-side: Baseline vs ESM-C

**Figure 3**: Case study
- Pick 2-3 pockets where ESM-C excels
- Show: reference ligand, baseline output, ESM-C output
- Annotate binding interactions (H-bonds, hydrophobic)

**Figure 4**: Failure analysis
- Where does ESM-C underperform?
- Hypothesis: Small pockets, low sequence diversity?

### 4.3 Discussion Points (From Paper Insights)

**Insight 1**: Paper shows CrossDocked has "unrealistic protein-ligand interactions" (page 4)
- This means baseline Vina scores are inflated
- ESM-C may perform better on **Binding MOAD** (real experimental structures)
- **Recommendation**: If time permits, evaluate on Binding MOAD test set

**Insight 2**: Paper uses **inpainting** and **iterative optimization** (not just de novo)
- ESM-C might shine more in these tasks
- **Low-hanging fruit**: Test scaffold hopping with ESM-C pocket encoding

**Insight 3**: Paper's "resampling" trick (r=10) for inpainting (Extended Data Fig 1g)
- Harmonizes generated region with fixed context
- Could this help ESM-C? (More information to harmonize with)

**Insight 4**: SE(3) vs E(3) equivariance (reflection sensitivity)
- This is about chirality (R/S enantiomers)
- ESM-C doesn't change this (it's a scalar feature)
- Mention in thesis: "ESM-C is rotation/translation-invariant"

---

## Phase 5: Risk Mitigation

### 5.1 What if ESM-C doesn't improve results?

**Hypothesis 1**: ESM-C embeddings are too high-dimensional
- **Test**: PCA to reduce 960 → 256 dims, retrain
- **Timeline**: +1 week

**Hypothesis 2**: Encoder architecture is wrong
- **Test**: Switch to Option B (concatenate one-hot + ESM-C)
- **Timeline**: +1 week

**Hypothesis 3**: CrossDocked data quality is limiting factor
- **Test**: Train on Binding MOAD (40K samples, smaller but higher quality)
- **Timeline**: +2 weeks

**Hypothesis 4**: Conditional mode doesn't benefit from ESM-C
- **Test**: Try joint mode (diffuse pocket + ligand together)
- **Why**: Joint mode might use pocket context more
- **Timeline**: +2 weeks

**Thesis Angle for Negative Results**:
- "We rigorously tested ESM-C integration and found that..."
- "One-hot encoding is surprisingly effective, likely because..."
- "Computational analysis reveals pocket diversity in CrossDocked is low, limiting ESM-C's advantage"
- Still publishable if methodology is solid

### 5.2 Compute Resource Failure

**Scenario**: HPC has 2-week queue or limited GPU hours

**Backup Plan**:
- Use **Binding MOAD** instead (40K samples, 3x smaller)
- Reduce epochs to 200 (5-7 days training)
- Colab Pro with A100 (~$50/month) as last resort

### 5.3 ESM-C API Changes

**Scenario**: ESM-C is different from ESM-2, API incompatible

**Backup Plan**:
- Use ESM-2 (published, stable, 1280-dim)
- Adjust `residue_encoder` input dimension
- Thesis contribution still valid

---

## Appendix: Commands Cheatsheet

### Data Preprocessing
```bash
# Pre-compute ESM-C embeddings
python scripts/precompute_esmc.py \
    --input data/processed_crossdock_noH_full/ \
    --output data/processed_crossdock_esmc/ \
    --model esm2_t33_650M_UR50D
```

### Training
```bash
# Debug run (small subset)
python train.py --config configs/crossdock_fullatom_cond_esmc_debug.yml

# Full training (HPC)
qsub .claude/hpc/train_esmc.pbs
```

### Inference
```bash
# Generate test molecules
python generate_ligands.py checkpoints/esmc_model.ckpt \
    --pdbfile example/3rfm.pdb \
    --ref_ligand A:330 \
    --n_samples 100 \
    --outfile test.sdf
```

### Evaluation
```bash
# Compute metrics
python analyze_results.py \
    --checkpoint checkpoints/esmc_model.ckpt \
    --test_set data/processed_crossdock_esmc/test.npz \
    --baseline checkpoints/crossdocked_fullatom_cond.ckpt \
    --output metrics.json

# Docking (if Vina available)
smina -r pocket.pdb -l generated.sdf --score_only
```

---

## Timeline Summary (6-8 weeks)

| Week | Phase | Deliverable |
|------|-------|-------------|
| 1 | Setup + Design | HPC access, baseline eval, ESM-C local test |
| 2 | Data Prep | Pre-computed ESM-C embeddings for full dataset |
| 3 | Implementation | Modified codebase, debug training run |
| 4-5 | Training | ESM-C model trained, checkpoints saved |
| 5-6 | Evaluation | Metrics computed, statistical tests, plots |
| 6-7 | Analysis | Case studies, failure analysis, figures |
| 7-8 | Writing | Draft thesis chapters, proofread, submit |

**Critical Path**:
1. Pre-compute embeddings (blocks training)
2. Training (blocks evaluation)
3. Evaluation (blocks analysis)

**Parallelizable**:
- Thesis intro/background (write during training)
- ESM-C ablations (if first model fails, try variations)

---

## Key Questions to Resolve This Week

1. **Pocket representation**: Is CrossDock full-atom or CA-only?
   - Affects sequence extraction from one-hot
   - Check: `configs/crossdock_fullatom_cond.yml` → `pocket_representation`

2. **Decoder usage in conditional mode**: Is pocket decoder trained?
   - Read `conditional_model.py:kl_prior()`
   - If pocket is fixed, decoder might be no-op

3. **HPC GPU specs**: What GPUs are available?
   - Email HPC support or check wiki
   - V100 vs A100 affects batch size

4. **ESM-C availability**: Is ESM-C released or use ESM-2?
   - Check `fair-esm` GitHub for latest models
   - ESM-2 is stable fallback

5. **Dataset format**: How are residues stored in NPZ?
   - Is there a residue ID field?
   - Or do we infer from pocket_one_hot?

---

## Next Immediate Steps (This Week)

**Day 1-2**:
1. ✅ Review this plan, discuss with advisor
2. ⏳ Access HPC, check GPU specs
3. ⏳ Verify baseline checkpoint works: `python generate_ligands.py checkpoints/crossdocked_fullatom_cond.ckpt --pdbfile example/3rfm.pdb --ref_ligand A:330 --n_samples 10`
4. ⏳ Install ESM locally: `uv pip install fair-esm`
5. ⏳ Test ESM-2/ESM-C inference on example sequence

**Day 3-5**:
1. ⏳ Analyze `conditional_model.py` - confirm decoder behavior
2. ⏳ Read `process_crossdock.py` - understand residue extraction
3. ⏳ Write `scripts/precompute_esmc.py` (draft version)
4. ⏳ Test on 10 pockets from train set
5. ⏳ Verify embedding shape aligns with pocket coords

**Day 6-7**:
1. ⏳ Modify `dynamics.py` (Option A implementation)
2. ⏳ Modify `lightning_modules.py` (add `use_esmc` flag)
3. ⏳ Create `configs/crossdock_fullatom_cond_esmc_debug.yml`
4. ⏳ Debug training run (10 samples, 5 epochs)
5. ⏳ Fix any shape mismatches or errors

**Week 2 Goal**: Have working prototype that can train 1 epoch without errors

---

## Success Mantra

> "Perfect is the enemy of good. Ship the prototype, measure the improvement, write the thesis."

**Remember**:
- You're NOT building a production system
- You're VALIDATING a hypothesis
- Negative results are STILL results
- 8 weeks is SHORT - stay focused

**Good luck, Hannes! 🚀**
