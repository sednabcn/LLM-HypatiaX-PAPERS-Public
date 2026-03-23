# HypatiaX Notebooks

Step-by-step reproduction of all experiments, figures, and statistical
tests from:

> **LLMs as Interfaces to Symbolic Discovery: Perfect Extrapolation via
> Hybrid Architectures**  
> Dr. Ruperto Pedro Bonet Chaple — JMLR 2026

---

## Prerequisites

```bash
pip install numpy pandas matplotlib seaborn scipy scikit-learn pathlib
```

PySR (optional — required only if re-running symbolic regression from scratch):

```bash
pip install pysr
```

Or install everything at once from the repo root:

```bash
pip install -r requirements.txt
```

---

## Data

All notebooks read from:

```
hypatiax/data/results/
├── hybrid_llm_nn/all_domains/
│   ├── hybrid_llm_nn_all_domains_20260115_131438.json   ← primary
│   └── hybrid_llm_nn_all_domains_20260115_133510.json   ← secondary (nb 03)
└── comparison_results/
    ├── extrapolation/
    │   └── all_domains_extrap_v4_20260124_131545.json   ← nb 04
    └── noise-noiseless/15/
        └── comparison_FIXED_20260124_150744.json        ← nb 04
```

Path resolution is **automatic** — each notebook detects the repo root
at runtime regardless of where you run it from.

---

## Run Order

| # | Notebook | What it does | Key output |
|---|----------|--------------|------------|
| 01 | `01_data_generation.ipynb` | Dataset overview, 30 equations, domain/difficulty distributions | `figures/01_dataset_composition.pdf` |
| 02 | `02_pure_llm_experiments.ipynb` | Pure LLM baseline — where it excels vs struggles | `figures/02_pure_llm_performance.pdf` |
| 03 | `03_hybrid_experiments.ipynb` | Hybrid v40 results by domain, difficulty, routing analysis | `figures/03_hybrid_performance.pdf` |
| 04 | `04_extrapolation_analysis.ipynb` | Arrhenius figure, symbolic vs neural scatter | `figures/04_*.pdf` |
| 05 | `05_figure_generation.ipynb` | All 5 publication figures (PDF + PNG) | `figures/figure[1-5]_*.pdf` |
| 06 | `06_statistical_tests.ipynb` | Mann-Whitney U, Kruskal-Wallis, Cohen's d, bootstrap CIs, claims checklist, LaTeX table | `figures/06_statistical_summary.pdf`, `figures/table_statistical_tests.tex` |

Run in order — each notebook is self-contained but later ones assume
familiarity with results from earlier ones.

---

## Output

All figures and tables are written to:

```
hypatiax/notebooks/figures/
```

This directory is created automatically on first run.

---

## Reproducibility

- Random seed: **42** (fixed in every notebook)
- LLM calls: temperature **0.0** (deterministic)
- PySR: `random_state=42`, `seed + index × 7` per equation

For questions or issues open a GitHub issue or contact the author.
