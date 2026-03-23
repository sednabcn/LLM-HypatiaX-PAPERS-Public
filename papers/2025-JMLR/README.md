# HypatiaX: AI-Driven Formula Discovery with Large Language Models

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**HypatiaX** is a research framework for evaluating Large Language Models (LLMs) in scientific formula discovery across multiple domains including Materials Science, Fluid Dynamics, Thermodynamics, Mechanics, and Chemistry.

This repository contains the code and experimental protocols accompanying the paper:

> **"HypatiaX: AI-Driven Formula Discovery with Large Language Models"**  
> *Journal of Machine Learning Research (JMLR), 2025*  
> Ruperto Pedro Bonet Chaple

---

## 🚀 Quick Start

### Installation

Install directly from GitHub:

```bash
pip install git+https://github.com/sednabcn/LLM-HypatiaX-PAPERS.git#subdirectory=papers/2025-JMLR
```

Or clone and install locally:

```bash
git clone https://github.com/sednabcn/LLM-HypatiaX-PAPERS.git
cd LLM-HypatiaX-PAPERS/papers/2025-JMLR
pip install -e .
```

### Basic Usage

```python
from hypatiax.protocols.experiment_protocol import ExperimentProtocol

# Initialize experiment protocol
protocol = ExperimentProtocol()

# Run formula discovery experiment
results = protocol.run_experiment(
    domain="materials_science",
    model="gpt-4",
    num_trials=10
)

# Analyze results
protocol.analyze_results(results)
```

---

## 📋 Features

- **Multi-Domain Support**: Experiments across 5 scientific domains
  - Materials Science (Hall-Petch relation)
  - Fluid Dynamics (Darcy-Weisbach equation)
  - Thermodynamics (Heat transfer, efficiency)
  - Mechanics (Stress-strain, buckling)
  - Chemistry (Reaction kinetics, equilibrium)

- **LLM Evaluation Framework**: Systematic evaluation of different LLMs for formula discovery

- **Reproducible Experiments**: Standardized protocols for consistent evaluation

- **Performance Metrics**: Comprehensive metrics for formula quality assessment

---

## 🛠️ Requirements

### Core Dependencies
- Python >= 3.8
- numpy >= 1.26.0
- pandas >= 2.1.0
- scikit-learn >= 1.3.0
- matplotlib >= 3.7.0
- scipy >= 1.11.0

### Optional Dependencies

Install with NLP features (for advanced text processing):
```bash
pip install "git+https://github.com/sednabcn/LLM-HypatiaX-PAPERS.git#subdirectory=papers/2025-JMLR[nlp]"
```

Install with development tools:
```bash
pip install -e ".[dev]"
```

Install everything:
```bash
pip install -e ".[all]"
```

---

## 📖 Documentation

### Running Experiments

```python
from hypatiax.protocols.experiment_protocol import ExperimentProtocol

# Create protocol instance
protocol = ExperimentProtocol()

# Configure experiment
config = {
    "domain": "materials_science",
    "model": "gpt-4",
    "temperature": 0.7,
    "max_iterations": 100
}

# Run experiment
results = protocol.run_experiment(**config)

# Save results
protocol.save_results(results, "experiment_results.json")
```

### Analyzing Results

```python
from hypatiax.protocols.experiment_protocol import ExperimentProtocol

# Load experiment results
protocol = ExperimentProtocol()
results = protocol.load_results("experiment_results.json")

# Generate analysis report
report = protocol.generate_report(results)

# Visualize results
protocol.plot_results(results, save_path="figures/")
```

### Available Domains

```python
# List all available experimental domains
domains = protocol.list_domains()
print(domains)

# Get domain-specific information
domain_info = protocol.get_domain_info("materials_science")
print(domain_info)
```

---

## 📊 Example Notebooks

Coming soon: Jupyter notebooks demonstrating:
- Single domain experiments
- Multi-domain comparisons
- LLM performance benchmarking
- Result visualization and analysis

---

## 🔬 Research Applications

HypatiaX is designed for researchers interested in:

- **AI for Science**: Evaluating LLMs for scientific discovery
- **Formula Discovery**: Automated discovery of mathematical relationships
- **Symbolic Regression**: Comparing LLM-based vs traditional approaches
- **Multi-Domain Learning**: Cross-domain transfer in scientific reasoning

---

## 🤝 Contributing

We welcome contributions! Please feel free to:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 Citation

If you use HypatiaX in your research, please cite:

```bibtex
@article{bonetchaple2025hypatiax,
  title={HypatiaX: AI-Driven Formula Discovery with Large Language Models},
  author={Bonet Chaple, Ruperto Pedro},
  journal={Journal of Machine Learning Research},
  year={2025},
  volume={TBD},
  pages={TBD}
}
```

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Ruperto Pedro Bonet Chaple**
- Email: ruperto.bonet@modelphysmat.com
- GitHub: [@sednabcn](https://github.com/sednabcn)

---

## 🐛 Issues and Support

For bug reports, feature requests, or questions:
- **GitHub Issues**: [https://github.com/sednabcn/LLM-HypatiaX-PAPERS/issues](https://github.com/sednabcn/LLM-HypatiaX-PAPERS/issues)
- **Email**: ruperto.bonet@modelphysmat.com

---

## 🙏 Acknowledgments

This research builds upon extensive work in:
- Large Language Models for scientific reasoning
- Symbolic regression and formula discovery
- Multi-domain machine learning

Special thanks to the open-source community for the foundational tools that made this research possible.

---

## 📚 Related Publications

- [Link to JMLR paper when published]
- [Link to preprint/arXiv if available]

---

## 🔄 Version History

### v1.0.0 (2025)
- Initial release
- Support for 5 scientific domains
- Standardized experiment protocols
- Comprehensive evaluation metrics

---

## ⚡ Quick Troubleshooting

### Import Errors

If you encounter import errors:

```bash
# Verify installation
python -c "from hypatiax.protocols.experiment_protocol import ExperimentProtocol; print('Installation successful')"
```

### Version Conflicts

If you have an older version installed:

```bash
pip uninstall hypatiax
pip install git+https://github.com/sednabcn/LLM-HypatiaX-PAPERS.git#subdirectory=papers/2025-JMLR
```

### Python Version Issues

Ensure you're using Python 3.8 or higher:

```bash
python --version
```

---

**Happy Experimenting! 🧪🔬**
