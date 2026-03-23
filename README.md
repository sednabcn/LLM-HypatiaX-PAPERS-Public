# LLM-HypatiaX-PAPERS

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.12-yellow.svg)](papers/2025-JMLR/requirements.txt)
[![JMLR](https://img.shields.io/badge/Venue-JMLR-2026-blue.svg)](papers/2025-JMLR/)

Research repository for the HypatiaX hybrid symbolic-neural system — a framework
for scientific equation discovery combining symbolic regression (PySR), LLM
interpretation, and multi-layer validation.

---

## 📁 Repository Structure

```
LLM-HypatiaX-PAPERS/
├── papers/
│   ├── 2025-JMLR/              # ← Primary paper (JMLR 2026) — start here
│   │   ├── hypatiax/           # All reproducible code and data
│   │   ├── requirements.txt    # Python dependencies
│   │   ├── code/               # Audit and traceability scripts
│   │   ├── submission/         # Submission packages
│   │   ├── reviews/            # Review responses
│   │   └── README.md
│   ├── 2025-NeurIPS/           # Scaling laws (in preparation)
│   ├── 2026-ICML/              # Multi-modal discovery (planning)
│   └── 2025-AAAI/              # Explainability (planning)
├── shared/                     # Shared datasets and utilities
├── tools/                      # Repository management scripts
├── templates/                  # Venue-specific LaTeX templates
├── docs/                       # Repository documentation
├── Dockerfile                  # Reproducible environment
└── requirements.txt
```

---

## 🚀 Quick Start (JMLR paper)

```bash
git clone https://github.com/sednabcn/LLM-HypatiaX-Papers
cd LLM-HypatiaX-Papers/papers/2025-JMLR
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
  title={Why Extrapolation Breaks Na{\"i}ve Analytical Discovery},
  author={Bonet Chaple, Ruperto Pedro},
  journal={Journal of Machine Learning Research},
  year={2026}
}
```

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
