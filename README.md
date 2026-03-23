# LLM-HypatiaX-PAPERS-Public

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.12-yellow.svg)](papers/2025-JMLR/requirements.txt)
[![JMLR](https://img.shields.io/badge/Venue-JMLR-2026-blue.svg)](papers/2025-JMLR/)

Research repository for the HypatiaX hybrid symbolic-neural system — a framework
for scientific equation discovery combining symbolic regression (PySR), LLM
interpretation, and multi-layer validation.

---

## 📁 Repository Structure

```
LLM-HypatiaX-PAPERS-Public/
├── activate_hypatiax.sh
├── Dockerfile
├── LICENSE
├── README.md
├── requirements.txt
├── setup_environment.sh
├── VERSION
└── papers/
    └── 2025-JMLR/              # ← Primary paper (JMLR 2026) — start here
        ├── activate_hypatiax.sh
        ├── pyproject.toml
        ├── requirements.txt
        ├── setup_environment.sh
        ├── docs/
        │   └── REPRODUCTION_GUIDE.md
        └── hypatiax/           # All reproducible code and data
            ├── __init__.py
            ├── path.py
            ├── README.md
            ├── analysis/
            │   ├── statistical_analysis_full.py
            │   └── statistical_analysis_unified.py
            ├── core/
            │   ├── base_pure_llm/          # Pure LLM baselines
            │   │   ├── baseline_pure_llm.py
            │   │   ├── baseline_pure_llm_defi.py
            │   │   └── baseline_pure_llm_defi_discovery.py
            │   ├── generation/             # Hybrid system architectures
            │   │   ├── hybrid_all_domains/
            │   │   ├── hybrid_all_domains_llm_nn/
            │   │   ├── hybrid_defi_llm_guided/
            │   │   ├── hybrid_defi_llm_nn/
            │   │   ├── hybrid_defi_system/
            │   │   └── hybrid_llm_guide_validation/
            │   └── training/               # NN training & adaptive config
            │       ├── adaptive_config.py
            │       ├── baseline_neural_network.py
            │       ├── baseline_neural_network_defi.py
            │       └── baseline_neural_network_defi_improved.py
            ├── data/results/               # Experimental results (JSON/CSV)
            │   ├── comparison_results/
            │   │   ├── extrapolation/
            │   │   ├── feynman-tests/noise-sweep/I.12.1-correction/
            │   │   └── noise-noiseless/{15,noiseless}/
            │   ├── extrapolation/
            │   ├── hybrid_llm_nn/{all_domains,defi}/
            │   ├── hybrid_pysr/{all_domains,defi}/
            │   ├── llm_guided/{all_domains,defi}/
            │   ├── standalone_llm_nn/
            │   └── to_generate_figures/
            ├── experiments/
            │   ├── benchmarks/             # Noise-sweep, sample-complexity, hybrid
            │   ├── comparison/             # Cross-system comparative suite
            │   └── tests/                  # Extrapolation & DeFi test protocols
            ├── notebooks/                  # End-to-end reproduction notebooks
            │   ├── 01_data_generation.ipynb
            │   ├── 02_pure_llm_experiments.ipynb
            │   ├── 03_hybrid_experiments.ipynb
            │   ├── 04_extrapolation_analysis.ipynb
            │   ├── 05_figure_generation.ipynb
            │   └── 06_statistical_tests.ipynb
            ├── protocols/                  # Experiment protocol scripts
            ├── tools/
            │   ├── symbolic/               # HypatiaX v40 engine & detectors
            │   ├── utils/                  # JSON, figure, comparison helpers
            │   ├── validation/             # Dimensional, domain, ensemble validators
            │   └── visualizations/         # Plot and figure generation
            └── legacy/                     # Archived prior versions (reference only)
                ├── analysis/
                ├── core/generation/
                ├── experiments/
                ├── protocols/
                └── tools/
```

---

## 🚀 Quick Start (JMLR paper)

```bash
git clone https://github.com/sednabcn/LLM-HypatiaX-PAPERS-Public
cd LLM-HypatiaX-PAPERS-Public/papers/2025-JMLR
pip install -r requirements.txt        # Julia ≥ 1.9 required for PySR campaigns
cd hypatiax
```

Full reproduction instructions:
**[`papers/2025-JMLR/hypatiax/README.md`](papers/2025-JMLR/hypatiax/README.md)**

---

## 📊 Key Results (JMLR 2026)

| System | Benchmark | Success Rate | Extrapolation |
|--------|-----------|-------------|---------------|
| HypatiaX Hybrid v40 | Core 15 | 96.7% (29/30) | Median error < 10⁻¹² |
| Neural Network | Core 15 | — | Mean error 1,231% |
| HypatiaX Hybrid DeFi | Feynman SR (30 eq.) | **96.7%** | +17.4 pp vs AI Feynman |
| Pure LLM | Core 15 | 67.5% (27/40) | Fails on DeFi (40%) |

Mann-Whitney U = 0, p = 1.11×10⁻⁶ (hybrid vs neural network, Core 15).

---

## 📦 Papers Overview

| Directory | Focus | Venue | Status |
|-----------|-------|-------|--------|
| `2025-JMLR/` | Hybrid system architecture + extrapolation | JMLR | In submission |
| `2025-NeurIPS/` | Scaling laws for symbolic regression | NeurIPS | In preparation |
| `2026-ICML/` | Multi-modal equation discovery | ICML | Planning |
| `2025-AAAI/` | Explainability and interpretability | AAAI | Planning |

---

## 🔬 Reproducibility

All canonical scripts use `SEED = 42`. Results verified March 2026 after
the `evaluate_llm_formula` bug fix (v2). See the JMLR hypatiax README for
full details and the v2 correction notes.

---

## 📝 Citation

```bibtex
@article{bonetchaple2026hypatiax,
  title={HypatiaX: A Hybrid Symbolic-Neural Framework for Extrapolation-Reliable Analytical Discovery},
  author={Bonet Chaple, Ruperto Pedro},
  journal={Journal of Machine Learning Research},
  year={2026}
}
```

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
