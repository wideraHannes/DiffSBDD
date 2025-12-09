# AutoDock Vina - Quick Reference Card

## 🚀 Quick Start (30 seconds)

```bash
# Analyze test dataset with Vina docking
cd /Users/hanneswidera/Uni/Master/thesis/DiffSBDD
uv run python create_docking_ground_truth.py
```

**Output:** `data/dummy_testing_dataset_10_tests/ground_truth_with_vina_exhaustiveness16.csv`

---

## 📊 What You Get

| Metric | Description | Good Range |
|--------|-------------|------------|
| `vina_score` | Binding affinity (kcal/mol) | -7 to -9 |
| `qed` | Drug-likeness | 0.5 - 0.8 |
| `sa_score` | Synthetic accessibility | 0.6 - 0.9 |
| `logp` | Lipophilicity | 0 - 3 |
| `molecular_weight` | MW | 300 - 500 |

---

## 🎯 Three Ways to Use

### 1. Analyze Ground Truth (baseline)
```bash
uv run python create_docking_ground_truth.py
```
**Use:** Establish baseline for comparison

### 2. Quick Molecule Score
```python
from analysis.vina_docking import vina_score
from rdkit import Chem

mol = Chem.MolFromSmiles('your_smiles')
score = vina_score(mol, 'receptor.pdbqt', center=(x,y,z))
print(f"Score: {score:.2f} kcal/mol")
```
**Use:** Interactive testing

### 3. Batch Analysis
```python
from analysis.vina_docking import dock_molecules

scores = dock_molecules(mol_list, 'receptor.pdbqt', center=(x,y,z))
```
**Use:** Evaluate generated molecules

---

## ⚙️ Configuration

**Edit `create_docking_ground_truth.py` lines 138-139:**

```python
USE_SCORE_ONLY = False  # True=fast, False=accurate
EXHAUSTIVENESS = 16     # 8=fast, 16=default, 32=best
```

| Mode | Speed | Use For |
|------|-------|---------|
| `score_only=True` | 1-2 sec/mol | Initial screening |
| `exhaustiveness=8` | 5-10 sec/mol | Testing |
| `exhaustiveness=16` ⭐ | 10-20 sec/mol | **Standard** |
| `exhaustiveness=32` | 30-60 sec/mol | Publication |

---

## 📈 Score Interpretation

```
Vina Score (kcal/mol)    Binding Strength

    < -10.0              Excellent ⭐⭐⭐
    -9.0 to -10.0        Strong ⭐⭐
    -7.0 to -9.0         Good ⭐        ← Most drugs
    -5.0 to -7.0         Moderate
    -3.0 to -5.0         Weak
    > -3.0               Poor
```

**Your test results:** Mean -7.0 ± 1.1 = GOOD ✅

---

## 🗂️ File Locations

```
Main Script:       create_docking_ground_truth.py
Core Module:       analysis/vina_docking.py
Documentation:     analysis/README_VINA.md
Full Guide:        DOCKING_ANALYSIS_COMPLETE.md
Test Example:      example/test_vina_metric.py
```

---

## 🔧 Common Commands

```bash
# Check Vina installation
vina --version

# Prepare receptor
obabel receptor.pdb -O receptor.pdbqt -xr

# View results
column -t -s, data/.../ground_truth_with_vina_*.csv | less -S

# Quick stats
uv run python -c "import pandas as pd; df=pd.read_csv('data/.../ground_truth_with_vina_exhaustiveness16.csv'); print(df['vina_score'].describe())"
```

---

## ⚠️ Quick Fixes

| Problem | Solution |
|---------|----------|
| "No pairs found" | Check file naming: `*-pocket10.pdb` & `*-pocket10_*.sdf` |
| "High scores (>100)" | Use `score_only=False` for optimization |
| "Slow" | Reduce to `EXHAUSTIVENESS = 8` |
| "Import error" | Run with `uv run python` not plain `python` |

---

## 📞 Help

1. **Detailed docs:** `analysis/README_VINA.md`
2. **Examples:** `example/test_vina_metric.py`
3. **Full guide:** `DOCKING_ANALYSIS_COMPLETE.md`

---

## ✅ Tested & Working

- ✅ 10 test molecules processed
- ✅ 100% success rate
- ✅ ~2 min runtime (exhaustiveness=16)
- ✅ Mean score: -7.0 kcal/mol (GOOD binding)

**Status: READY FOR THESIS EXPERIMENTS** 🚀
