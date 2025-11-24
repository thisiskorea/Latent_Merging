# Latent Merging: Dynamic and Reversible Composition of Large Language Models

[![Paper Status](https://img.shields.io/badge/Status-Under%20Review-yellow)](https://github.com/thisiskorea/Latent_Merging)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Note**: This repository contains the code implementation for a paper currently under review. The paper introduces **Latent Merging**, a novel paradigm for composing large language models in representation space rather than parameter space.

## Overview

Weight merging has been a common approach to combine large language models, but its static and irreversible nature limits controllability and can destabilize behavior. This work proposes **Latent Merging**, which composes models in the hidden-representation space to enable:

- **Dynamic control**: Adjust merging at inference time without retraining
- **Reversibility**: Switch between model behaviors on-demand
- **Layer-wise selectivity**: Apply different merging strategies at different depths
- **Stability**: Preserve semantic coherence and avoid representational collapse

### Key Contributions

1. **Conceptual**: Establish correspondence between weight merging and latent merging
2. **Framework**: Unified latent merging framework generalizing LERP, SLERP, and RegMean to representation space
3. **Theoretical**: Second-order bounds on loss under RMSNorm nonlinearity with practical guidance
4. **Empirical**: Systematic evaluation on JudgeBench showing consistent improvements over weight merging

## Paper Abstract

> Weight merging is a common way to combine large language models, but its static and irreversible nature limits controllability and can destabilize behavior. We propose Latent Merging, which composes models in the hidden-representation space to enable dynamic, reversible, and layer-wise control without modifying weights. We unify classic operators—linear/spherical interpolation, and regularized means—under a single operator view and extend them from parameters to latents. We derive local second-order bounds on loss change that account for RMSNorm nonlinearity and head mismatch, yielding practical guidance (merge later; align heads) and stability guarantees. In data-free evaluation on Qwen2.5-7B-Instruct and its fine-tuned derivative, Latent Merging consistently surpasses weight merging on JudgeBench across reasoning, knowledge, mathematics, and coding.

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended for LLM inference)
- 16GB+ RAM (32GB+ recommended for 7B models)

### Setup

```bash
# Clone the repository
git clone https://github.com/thisiskorea/Latent_Merging.git
cd Latent_Merging

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install in editable mode
pip install -e .
```

### Expected Dependencies

```
torch>=2.0.0
transformers>=4.35.0
numpy>=1.24.0
scipy>=1.10.0
scikit-learn>=1.3.0
pyyaml>=6.0
tqdm>=4.65.0
matplotlib>=3.7.0
seaborn>=0.12.0
```

## Repository Structure

```
Latent_Merging/
├── src/                      # Source code
│   ├── merging/             # Core latent merging algorithms
│   │   ├── lerp.py          # Linear interpolation
│   │   ├── slerp.py         # Spherical linear interpolation
│   │   └── regmean.py       # Regularized mean
│   ├── models/              # Model wrappers and utilities
│   ├── evaluation/          # JudgeBench evaluation scripts
│   └── utils/               # Helper functions
├── experiments/             # Experiment configurations
│   ├── configs/             # YAML configuration files
│   └── scripts/             # Evaluation and analysis scripts
├── notebooks/               # Jupyter notebooks for analysis
├── results/                 # Experimental results and figures
├── tests/                   # Unit tests
├── requirements.txt         # Python dependencies
├── setup.py                 # Package installation
├── README.md                # This file
└── CLAUDE.md                # AI assistant guide
```

## Quick Start

### Basic Usage

```python
from src.merging import LatentMerger
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load models
model_a = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
model_b = AutoModelForCausalLM.from_pretrained("open-thoughts/OpenThinker3-7B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

# Initialize latent merger
merger = LatentMerger(
    model_a=model_a,
    model_b=model_b,
    method="slerp",  # Options: "lerp", "slerp", "regmean"
    alpha=0.5,       # Mixing ratio
    merge_layers=[20, 21, 22, 23, 24, 25, 26, 27]  # Layer-wise control
)

# Generate with merged representations
prompt = "Explain the concept of gradient descent."
inputs = tokenizer(prompt, return_tensors="pt")
outputs = merger.generate(**inputs, max_length=512)
print(tokenizer.decode(outputs[0]))
```

### Running Experiments

```bash
# Experiment A: Comparative evaluation (Weight vs Latent Merging)
python experiments/scripts/run_judgebench.py \
    --config experiments/configs/exp_a_comparison.yaml \
    --output results/exp_a/

# Experiment B: Latent space similarity analysis
python experiments/scripts/run_similarity.py \
    --config experiments/configs/exp_b_similarity.yaml \
    --output results/exp_b/

# Experiment C: Layer-wise and ratio-wise analysis
python experiments/scripts/run_layerwise.py \
    --config experiments/configs/exp_c_layerwise.yaml \
    --output results/exp_c/
```

## Methodology

### Merging Operators

#### 1. Linear Interpolation (LERP)
```python
h' = (1 - α) * h_A + α * h_B
```
Simple weighted average in Euclidean space.

#### 2. Spherical Linear Interpolation (SLERP)
```python
h' = (sin((1-α)Ω) / sin(Ω)) * h_A + (sin(αΩ) / sin(Ω)) * h_B
where Ω = arccos(<h_A, h_B>)
```
Interpolation along the geodesic on the unit hypersphere.

#### 3. Regularized Mean (RegMean)
```python
h' = mean([h_A, h_B]) - λ * R(h_i)
```
Stabilized averaging with regularization.

### Theoretical Framework

We provide **local second-order bounds** on loss change under latent merging:

```
ℓ(g(h'_α)) ≤ (1-α)ℓ(z_A) + αℓ(z_B) + O(α(1-α)K_g‖h_B - h_A‖²)
```

where `K_g` captures the curvature induced by RMSNorm and the LM head.

**Key insights**:
- Merge at later layers (lower curvature)
- Align LM heads across models
- Use SLERP to control ‖h_B - h_A‖

## Experimental Results

### A. JudgeBench Evaluation

| Method  | Knowledge (Latent/Weight) | Reasoning | Math | Coding | Overall |
|---------|---------------------------|-----------|------|--------|---------|
| SLERP   | 59.42 / 40.59            | 100.00 / 0.00 | 86.37 / 13.64 | 98.34 / 1.67 | **74.76 / 25.25** |
| LERP    | 98.03 / 1.97             | 91.31 / 8.69  | 98.53 / 1.47  | 100.00 / 0.00 | **97.15 / 2.85** |
| RegMean | 74.06 / 25.94            | 60.00 / 40.00 | 58.82 / 41.18 | 74.19 / 25.81 | **69.82 / 30.18** |

**Latent merging consistently outperforms weight merging across all categories.**

### B. Representation Similarity

| Method  | Midness ↑ (L/W) | Arc Deviation ↓ (L/W) | CKA ↑ (L/W) |
|---------|-----------------|----------------------|-------------|
| SLERP   | **0.80** / 0.61 | **0.12** / 0.19      | **0.89** / 0.35 |
| LERP    | **0.77** / 0.64 | **0.04** / 0.32      | **0.83** / 0.32 |
| RegMean | **0.78** / 0.64 | **0.02** / 0.16      | **0.83** / 0.35 |

Latent merging preserves representational geometry more effectively.

### C. Layer-wise Analysis

- **Later layers** (L20-L27) yield substantially larger gains
- **Higher ratios** (α ≈ 0.75) work best in deeper layers
- **Operator-specific nuances** but consistent overall trend

## Evaluation Benchmarks

We use **JudgeBench** for evaluation, which employs LLM-as-judge to assess:
- **Knowledge**: MMLU-Pro
- **Reasoning**: LiveBench reasoning tasks
- **Math**: LiveBench math problems
- **Coding**: LiveCodeBench

Traditional accuracy metrics (exact match) can be misleading due to formatting sensitivity and instability in merged models. JudgeBench provides robust pairwise comparison.

## Models

- **Base Model**: [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
- **Fine-tuned Derivative**: [OpenThinker3-7B](https://huggingface.co/open-thoughts/OpenThinker3-7B)

Both models share the same architecture, enabling clean comparison.

## Limitations

- **Inference Overhead**: Latent merging requires running two models in parallel, increasing computational cost
- **Source Model Ceiling**: Performance is bounded by the capabilities of source models
- **Architecture Requirements**: Currently tested on Transformer-based LLMs with similar architectures

## Citation

If you find this work useful, please cite our paper (citation will be added upon publication):

```bibtex
@article{kim2025latent,
  title={Latent Merging: Dynamic and Reversible Composition of Large Language Models},
  author={Kim, JaeSeong and Lee, Suan},
  journal={Under Review},
  year={2025}
}
```

## Data Availability

Evaluation outputs and intermediate representations are available in this repository.

## Code Availability

All code for implementing latent merging operators and reproducing experiments is available in this repository.

## Funding

This work was supported by the Technological Innovation R&D Program [RS-2024-00508856] funded by the Ministry of SMEs and Startups (MSS, Korea).

## Contact

- **JaeSeong Kim** - Semyung University
- **Suan Lee** (Corresponding author) - suanlee@semyung.ac.kr

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

We thank the developers of:
- Hugging Face Transformers
- PyTorch
- JudgeBench, LiveBench, and LiveCodeBench evaluation frameworks
- Qwen and OpenThinker3 model teams

---

**Disclaimer**: This repository contains research code for a paper under review. Implementation details may be updated based on reviewer feedback.
