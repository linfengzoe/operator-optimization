# Lab6 Paper Report DOCX Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate a Chinese standard-paper-style DOCX report for the CUDA convolution optimization assignment, with high-quality Python-generated result figures, flow diagrams, and algorithm diagrams.

**Architecture:** A single reproducible Python script reads `docs/latest_run_results.json`, generates publication-style PNG/SVG/PDF figures under `docs/paper_figures/`, and creates a formatted DOCX under `output/doc/`. The DOCX embeds all key figures, tables, captions, and paper sections while using the latest measured results as the primary evidence.

**Tech Stack:** Python, matplotlib, python-docx, JSON source data.

---

### Task 1: Generate Figures and DOCX Script

**Files:**
- Create: `Lab6_experiment/tools/generate_paper_report.py`
- Read: `Lab6_experiment/docs/latest_run_results.json`
- Output: `Lab6_experiment/docs/paper_figures/*.png`
- Output: `Lab6_experiment/docs/paper_figures/*.svg`
- Output: `Lab6_experiment/docs/paper_figures/*.pdf`
- Output: `Lab6_experiment/output/doc/Lab6_卷积算子优化实验报告.docx`

- [ ] **Step 1: Create Python script**

Create a script that:
- loads measured results from `docs/latest_run_results.json`;
- renders seven figures:
  - optimization roadmap;
  - PyTorch/C++/CUDA extension workflow;
  - direct convolution vs implicit-GEMM algorithm diagram;
  - WMMA/Tensor Core execution schematic;
  - throughput trends;
  - speedup plus accuracy;
  - Tensor Core GEMM microbenchmark;
- creates the DOCX report with title, abstract, keywords, introduction, methods, experiments, results, discussion, conclusion, references, and appendix.

- [ ] **Step 2: Run the script**

Run:

```powershell
python Lab6_experiment/tools/generate_paper_report.py
```

Expected: script exits with code 0 and creates all figures and DOCX.

- [ ] **Step 3: Verify generated assets**

Run:

```powershell
Get-ChildItem Lab6_experiment/docs/paper_figures
Get-Item Lab6_experiment/output/doc/Lab6_卷积算子优化实验报告.docx
```

Expected: figure files exist and DOCX has non-zero size.

### Task 2: Validate Document Content

**Files:**
- Read: `Lab6_experiment/output/doc/Lab6_卷积算子优化实验报告.docx`

- [ ] **Step 1: Extract DOCX text**

Run a Python check using `python-docx`:

```powershell
python - <<'PY'
from docx import Document
doc = Document('Lab6_experiment/output/doc/Lab6_卷积算子优化实验报告.docx')
text = '\n'.join(p.text for p in doc.paragraphs)
print(len(text))
assert 'D:/codex_lab6_run' not in text
assert '中文路径' not in text
assert '摘要' in text
assert '实验结果与分析' in text
PY
```

Expected: assertions pass.

- [ ] **Step 2: Render or inspect layout**

If LibreOffice is available, convert DOCX to PDF and inspect generated pages. If not available, at minimum verify the DOCX structure, images, headings, tables, and paragraph text with `python-docx`.

### Task 3: Commit Generated Report

**Files:**
- Add: `Lab6_experiment/tools/generate_paper_report.py`
- Add: `Lab6_experiment/docs/paper_figures/*`
- Add: `Lab6_experiment/output/doc/Lab6_卷积算子优化实验报告.docx`
- Add: `Lab6_experiment/docs/superpowers/plans/2026-06-07-paper-report-docx.md`

- [ ] **Step 1: Check status**

Run:

```powershell
git status --short
```

Expected: only report-related files are changed or untracked.

- [ ] **Step 2: Commit**

Run:

```powershell
git add Lab6_experiment/tools/generate_paper_report.py Lab6_experiment/docs/paper_figures Lab6_experiment/output/doc/Lab6_卷积算子优化实验报告.docx Lab6_experiment/docs/superpowers/plans/2026-06-07-paper-report-docx.md
git commit -m "Create paper-style DOCX report"
```

Expected: commit succeeds.
