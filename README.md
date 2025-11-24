# Latent Merging: Dynamic and Reversible Composition of Large Language Models

[![Paper Status](https://img.shields.io/badge/Status-Under%20Review-yellow)](https://github.com/thisiskorea/Latent_Merging)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

> **Research Code Repository**: This repository contains the implementation code for a paper currently under review. The paper introduces **Latent Merging**, a novel paradigm for composing large language models in representation space rather than parameter space.

## Overview

This repository provides minimal, reproducible code for experimenting with four merging techniques between **Qwen2.5-7B-Instruct** (base) and **OpenThinker3-7B** (fine-tuned). Large artifacts (pkl files, safetensors, checkpoints) are excluded from the repository; only code and result summaries are included.

**Key Innovation**: Instead of merging model weights (static, irreversible), we merge hidden representations (dynamic, reversible, controllable) during inference, enabling layer-wise control without modifying parameters.

## Repository Structure

```
Latent_Merging/
├── src/
│   ├── latent_merging.py      # Core: LERP/SLERP/RegMean/Task Vector classes
│   └── metrics.py             # Evaluation: CKA, midness, geodesic metrics
├── scripts/
│   └── judgebench_eval.py     # JudgeBench A/B evaluation (LLM-as-judge)
├── results/                   # Experimental summary CSVs
│   ├── LERP.csv / LERP1.csv
│   ├── SLERP.csv / SLERP!.csv
│   ├── RegMean.csv / RegMean1.csv
│   └── system_costs.csv
├── artifacts/                 # [NOT IN GIT] Experimental outputs
│   ├── SLERP/                # Scale×step raw (*.pkl) + summary
│   ├── Lerp/
│   ├── RegMean/
│   ├── TaskVector/           # latent/weight merged TaskVector pkl
│   └── root_pkls/            # Baselines, merged weights, responses
├── requirements.txt
├── README.md                  # This file
└── CLAUDE.md                  # Detailed AI assistant guide
```

**Note**: The `artifacts/` directory containing large pkl files, model checkpoints, and merged weights is stored separately and not tracked in Git. See [Large File Deployment](#large-file-deployment) for details.

## Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (14-28GB VRAM recommended for dual 7B models)
- OpenAI API key (for JudgeBench evaluation)

### Setup

```bash
# Clone repository
git clone https://github.com/thisiskorea/Latent_Merging.git
cd Latent_Merging

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Required packages**: `torch`, `transformers`, `datasets`, `tqdm`, `pandas`, `openai` (v1+), `scikit-learn`

## Quick Start

### Basic Latent Merging (LERP/SLERP)

```python
from src.latent_merging import get_model, get_tokenizer, latent_mix_generate

# Load models
tokenizer = get_tokenizer("Qwen/Qwen2.5-7B-Instruct")
base_model = get_model("Qwen/Qwen2.5-7B-Instruct")
ft_model = get_model("open-thoughts/OpenThinker3-7B")

# Prepare messages
messages = [{"role": "user", "content": "Explain gradient descent simply."}]

# Generate with SLERP at layer 20, beta=0.5
text = latent_mix_generate(
    base_model, ft_model, tokenizer,
    messages=messages,
    mix_layer=20,
    beta=0.5,
    mode="slerp",  # or "lerp"
    max_new_tokens=512
)
print(text)
```

### RegMean and Task Vector

```python
from src.latent_merging import latent_regmean_generate, delta_generate, ActivationSteerer

# RegMean merging
text = latent_regmean_generate(
    base_model, ft_model, tokenizer,
    messages=messages,
    mix_layers=[20, 21, 22, 23],
    alpha=0.5
)

# Task Vector (delta steering)
deltas = {20: delta_vector_layer20, 21: delta_vector_layer21}
steerer = ActivationSteerer(base_model, deltas, alpha=0.1)
# Generate with active steering
output = base_model.generate(...)
steerer.remove()  # Always clean up hooks
```

### JudgeBench Evaluation

Compare two response sets using GPT-4o-mini as judge:

```bash
OPENAI_API_KEY=sk-... python scripts/judgebench_eval.py \
  --path-a artifacts/SLERP/latent_0.5_layer20.pkl \
  --path-b artifacts/root_pkls/OpenThinker_baseline.pkl \
  --judge-model gpt-4o-mini \
  --out results/comparison.jsonl
```

**Output**: Win rates, ties, statistical summary

## Merging Methods

### 1. LERP (Linear Interpolation)
```python
h' = (1 - α) * h_A + α * h_B
```
- Simple weighted average in Euclidean space
- Best overall: **97.15%** vs 2.85% for weight merging (JudgeBench)

### 2. SLERP (Spherical Linear Interpolation)
```python
# Geodesic interpolation on unit sphere with norm restoration
θ = arccos(<u, v>)  # u, v are normalized
h' = norm_interp * ((sin((1-α)θ)/sin(θ)) * u + (sin(αθ)/sin(θ)) * v)
```
- Preserves geometry, prevents collapse
- **74.76%** vs 25.25% for weight merging
- Higher representational similarity (CKA: 0.89 vs 0.35)

### 3. RegMean (Regularized Mean)
```python
h' = mean([h_A, h_B]) - λ * R(h_i)
```
- Stabilized averaging with regularization
- **69.82%** vs 30.18% for weight merging

### 4. Task Vector (Delta Steering)
```python
h' = h + α * (h_FT - h_base)
```
- Additive activation steering
- Highly controllable, layer-specific intervention

## Key Results

### Experiment A: Latent vs Weight Merging (JudgeBench)

| Method  | Knowledge | Reasoning | Math | Coding | **Overall** |
|---------|-----------|-----------|------|--------|-------------|
| **SLERP** | 59.42 / 40.59 | **100.00 / 0.00** | 86.37 / 13.64 | 98.34 / 1.67 | **74.76 / 25.25** |
| **LERP**  | 98.03 / 1.97  | 91.31 / 8.69  | 98.53 / 1.47  | **100.00 / 0.00** | **97.15 / 2.85** |
| **RegMean** | 74.06 / 25.94 | 60.00 / 40.00 | 58.82 / 41.18 | 74.19 / 25.81 | **69.82 / 30.18** |

*Format: Latent / Weight (win rates %)*

**Key Finding**: Latent merging consistently dominates weight merging across all operators and task categories, with particularly dramatic improvements in reasoning (100% vs 0%) and coding (100% vs 0%).

### Experiment B: Representation Similarity

| Metric | Latent | Weight | Δ |
|--------|--------|--------|---|
| **CKA** ↑ | **0.83-0.89** | 0.32-0.35 | **+0.51** |
| **Midness** ↑ | **0.77-0.80** | 0.61-0.64 | +0.15 |
| **Arc Deviation** ↓ | **0.02-0.12** | 0.16-0.32 | **-0.16** |

**Key Finding**: Latent merging preserves representational geometry significantly better, with ~50% improvement in CKA similarity.

### Experiment C: Layer-wise Analysis

**Best Configurations**:
- **Later layers** (L20-L27) consistently outperform early/mid layers
- **Higher ratios** (α ≈ 0.75) optimal in deeper layers
- **RegMean** at L25, α=0.50: 47.34 (Qwen2.5), 96.94 (OpenThinker3)
- **SLERP** at L20-L25, α=0.50-0.75: Strong across models

## Large File Deployment

Large artifacts are **not included** in this repository. To reproduce experiments:

1. **Model Checkpoints**: Download models via Hugging Face or place local checkpoints in desired paths. Specify paths via script arguments.

2. **Response/Merge Results**:
   - `artifacts/SLERP/`, `Lerp/`, `RegMean/`: Scale×step pkl files (`*.pkl`, `*_summary.pkl`)
   - `artifacts/TaskVector/`: `latent_merged-TaskVector.pkl`, `weight_merged-TaskVector.pkl`
   - `artifacts/root_pkls/`: Baselines (`Qwen_baseline.pkl`, `OpenThinker_baseline.pkl`), merged weights, evaluation results

3. **Merged Weights**: `weight_merged-*.pkl`, `latent_merged-*.pkl` stored separately. Reference paths in scripts.

**Recommendation**: Store large files in external storage (Google Drive, S3, institutional storage) and download as needed.

## Evaluation Framework

We use **JudgeBench** for evaluation because:
- Traditional exact-match metrics (e.g., MMLU-Pro) are sensitive to formatting and unstable for merged models
- LLM-as-judge provides robust pairwise comparison across fluency, coherence, and correctness
- Covers four domains: Knowledge (MMLU-Pro), Reasoning (LiveBench), Math (LiveBench), Coding (LiveCodeBench)

## Important Notes

### Security
- **DO NOT hardcode API keys**. Use environment variable `OPENAI_API_KEY`
- **DO NOT commit large files** (pkl, safetensors, checkpoints) to Git

### Reproducibility
- Requires JudgeBench dataset (`datasets` library) and pkl outputs from `artifacts/`
- Random seeds are set in code for deterministic generation
- Model versions: Qwen2.5-7B-Instruct (base), OpenThinker3-7B (fine-tuned)

### Memory Requirements
- Latent merging requires **two models in memory simultaneously** (~14GB × 2 for 7B models)
- Use `load_in_8bit=True` or `load_in_4bit=True` if memory-constrained
- Each generation processes both models in parallel via PyTorch hooks


## License

This project is licensed under the MIT License - see the LICENSE file for details.


