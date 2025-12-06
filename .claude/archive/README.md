# Thesis Documentation

> **Start here**: Read `THESIS_PLAN.md` first for complete overview.

---

## Documentation Structure

### 📘 Core Planning Documents

1. **THESIS_PLAN.md** ⭐ **START HERE**
   - Single source of truth for the entire thesis
   - Research question and hypothesis
   - Architecture overview
   - Implementation phases
   - Timeline and success criteria
   - References all other documents

2. **implementation_plan.md**
   - Day-by-day implementation guide
   - Validation-first approach (Google Research Tuning Playbook)
   - Specific experiments with success criteria
   - Scientific validation before full training
   - Debugging strategies

3. **CONFIGURATION_GUIDE.md**
   - Baseline vs ESM-C mode configuration
   - Backward compatibility patterns
   - Code examples for both modes
   - Unit testing strategies
   - Command-line usage

### 📗 Background & Architecture

4. **06_GLOBAL_POCKET_CONDITIONING.md**
   - Why global embedding approach is brilliant
   - Detailed architecture diagrams
   - Text-to-image analogy explained
   - Information flow visualization
   - FiLM conditioning mechanism

5. **IN_DEPTH_EXPLANATION.md**
   - Deep dive into DiffSBDD codebase
   - File-by-file analysis
   - Integration points for ESM-C
   - Existing code patterns
   - Where to make modifications

### 📙 Reference Materials

6. **ARCHITECTURE_SUMMARY.txt**
   - Quick reference for codebase structure
   - Key files and their roles
   - Data flow overview

7. **s43588-024-00737-x.pdf**
   - Original DiffSBDD paper (Nature Comp. Sci. 2024)
   - Read for deep understanding

---

## Quick Start Guide

### First Time Reading?

1. **Read**: `THESIS_PLAN.md` (30 min)
   - Understand research question
   - Get high-level architecture
   - See implementation phases

2. **Read**: `06_GLOBAL_POCKET_CONDITIONING.md` (15 min)
   - Understand why global embedding
   - See architecture diagrams
   - Get text-to-image analogy

3. **Skim**: `CONFIGURATION_GUIDE.md` (10 min)
   - See how baseline/ESM-C modes work
   - Understand configuration flags

4. **Reference**: `implementation_plan.md` when implementing
   - Follow day-by-day validation experiments
   - Use as checklist during implementation

### When Implementing?

**Day 1-7**: Follow `implementation_plan.md` step-by-step
- Validation experiments first
- Scientific approach (one variable at a time)
- Clear go/no-go decision points

**Throughout**: Reference `CONFIGURATION_GUIDE.md`
- Ensure both baseline and ESM-C modes work
- Follow code patterns
- Write unit tests

**When Stuck**: Check `IN_DEPTH_EXPLANATION.md`
- Understand existing code
- Find integration points
- See examples

---

## Key Concepts

### The Core Analogy

```
Text Prompt → CLIP → Stable Diffusion → Image
     ↓           ↓            ↓            ↓
Pocket Seq  →  ESM-C  →   DiffSBDD   →  Ligand
```

### Architecture in One Sentence

**Use a single 960-dim ESM-C embedding per pocket to steer ligand generation via FiLM conditioning, separating geometric (EGNN) and semantic (ESM-C) information.**

### Configuration in One Sentence

**A single `esmc_conditioning: bool` flag controls all ESM-C behavior, ensuring zero overhead in baseline mode and easy comparison.**

---

## File Sizes & Reading Time

| File | Size | Est. Reading Time | Priority |
|------|------|-------------------|----------|
| THESIS_PLAN.md | ~60 KB | 30-45 min | ⭐⭐⭐ Must read |
| implementation_plan.md | ~15 KB | 20-30 min | ⭐⭐⭐ When implementing |
| CONFIGURATION_GUIDE.md | ~30 KB | 20-30 min | ⭐⭐ When coding |
| 06_GLOBAL_POCKET_CONDITIONING.md | ~25 KB | 15-20 min | ⭐⭐ For understanding |
| IN_DEPTH_EXPLANATION.md | ~40 KB | 30-40 min | ⭐ Reference only |

---

## Updates & Maintenance

**Last Updated**: December 2024

**Maintained By**: Thesis Author

**Status**: Active development

---

## Quick Links

- **Codebase**: `DiffSBDD/` (parent directory)
- **Data**: `DiffSBDD/data/`
- **Configs**: `DiffSBDD/configs/`
- **Paper**: `s43588-024-00737-x.pdf` (this folder)

---

**Remember**: Start with `THESIS_PLAN.md` - everything else is referenced from there! 🎯
