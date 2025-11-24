# CLAUDE.md - AI Assistant Guide for Latent_Merging

## Project Overview

**Latent_Merging** is a research project implementing the paper "Latent Merging: Dynamic and Reversible Composition of Large Language Models" (currently under review). This work introduces a novel paradigm for composing LLMs in representation space rather than parameter space.

**Core Concept**: Instead of merging model weights (static, irreversible), we merge hidden representations (dynamic, reversible, controllable) during inference.

**Key Features**:
- Extends LERP, SLERP, and RegMean from parameter space to latent space
- Implements Task Vector (delta steering) for activation manipulation
- Enables layer-wise and ratio-wise control of model composition
- Provides CKA and midness metrics for representational evaluation
- Demonstrates consistent improvements over weight merging on JudgeBench

**Models Used**:
- Base: Qwen/Qwen2.5-7B-Instruct
- Fine-tuned derivative: open-thoughts/OpenThinker3-7B

## Repository Status

**Current State**: Research paper code repository (under review)
**Paper Status**: Under review for scientific publication
**Branch**: `claude/claude-md-miclwwc2pcdn3uvs-01HikorFWjxHtwWmbeXWeLKe`
**Data-free Evaluation**: No additional training data used; pure inference-time composition

## Actual Project Structure

```
Latent_Merging/
├── src/
│   ├── latent_merging.py      # Core implementation (509 lines)
│   │   ├── LayerLatentMixer   # LERP/SLERP mixing at specific layer
│   │   ├── ActivationSteerer  # Task Vector delta steering
│   │   ├── Helper functions   # tokenization, sampling, generation
│   │   └── Model caching      # Efficient model loading
│   └── metrics.py             # Evaluation metrics (74 lines)
│       ├── linear_cka()       # Centered Kernel Alignment
│       ├── midness_seq()      # Spherical midpoint metrics
│       └── Helper functions   # normalization, SLERP, geodesic distance
├── scripts/
│   └── judgebench_eval.py     # JudgeBench evaluation (133 lines)
│       ├── LLM-as-judge       # GPT-4o-mini comparisons
│       ├── Async evaluation   # Batch processing
│       └── Statistical summary
├── results/                    # Experimental CSV results
│   ├── LERP.csv / LERP1.csv
│   ├── SLERP.csv / SLERP!.csv
│   ├── RegMean.csv / RegMean1.csv
│   └── system_costs.csv
├── requirements.txt            # Python dependencies
├── README.md                   # Korean documentation
└── CLAUDE.md                   # This file
```

**Note**: Large artifacts (pkl files, model checkpoints) are stored separately and not in Git.

## Core Implementation Details

### 1. `src/latent_merging.py`

#### Key Classes

**`LayerLatentMixer`** - Main latent merging implementation
```python
class LayerLatentMixer:
    """Mixes hidden states from base and FT models at a specific layer."""

    def __init__(self, base_model, ft_model, layer_idx, beta=0.5,
                 last_token_only=True, mix_mode="lerp"):
        # mix_mode: "lerp" or "slerp"
        # beta: mixing ratio (0=base, 1=FT)
        # last_token_only: merge only final token vs all tokens
```

**Implementation Details**:
- Uses PyTorch forward hooks to intercept hidden states
- Captures FT model hidden state at target layer
- Merges it with base model hidden state using LERP or SLERP
- SLERP: Normalizes directions, interpolates on sphere, restores norm
- LERP: Simple weighted average

**`ActivationSteerer`** - Task Vector implementation
```python
class ActivationSteerer:
    """Applies delta vectors (task vectors) to hidden states."""

    def __init__(self, model, deltas: Dict[int, Tensor], alpha=0.10,
                 apply_to_all_tokens=True):
        # deltas: {layer_idx: delta_vector}
        # alpha: scaling factor for delta
        # h' = h + alpha * delta
```

**Helper Functions**:
- `get_model()` / `get_tokenizer()`: Model/tokenizer loading with caching
- `apply_chat_template()`: Format messages for chat models
- `get_decoder_layers()`: Extract transformer layers (handles Qwen/GPT architectures)
- `top_p_sample()`: Nucleus sampling for generation
- `_normalize()` / `_slerp()`: Geometric operations

#### Generation Functions

**`latent_mix_generate()`**:
```python
def latent_mix_generate(
    base_model, ft_model, tokenizer,
    messages: List[Dict],
    mix_layer: int = 20,
    beta: float = 0.5,
    mode: str = "slerp",  # or "lerp"
    max_new_tokens: int = 512,
    top_p: float = 0.9,
    temperature: float = 1.0,
) -> str:
    # Returns generated text with latent merging
```

**`latent_regmean_generate()`**:
- RegMean variant with regularization
- More stable than simple averaging

**`delta_generate()`**:
- Task Vector generation using ActivationSteerer
- `h' = h + alpha * delta`

### 2. `src/metrics.py`

**`linear_cka(X, Y)`**:
- Centered Kernel Alignment for representation similarity
- Input: (T, H) tensors (sequence_length, hidden_dim)
- Returns: scalar similarity [0, 1]

**`midness_seq(Hb, Hf, Hm)`**:
- Evaluates whether merged representation Hm lies at midpoint of Hb and Hf
- Returns:
  - `midpoint_cos`: Cosine similarity to geodesic midpoint
  - `arc_mid_deviation`: |d(b,m) - 0.5*d(b,f)|
  - `arc_ratio`: d(b,m) / d(b,f) (should be ~0.5)

### 3. `scripts/judgebench_eval.py`

**Purpose**: Compare two response sets using LLM-as-judge (GPT-4o-mini)

**Usage**:
```bash
OPENAI_API_KEY=sk-... python scripts/judgebench_eval.py \
  --path-a results/model_a_responses.pkl \
  --path-b results/model_b_responses.pkl \
  --judge-model gpt-4o-mini \
  --out results/comparison.jsonl
```

**Output**: Win rates, ties, statistical summary

## Development Workflows

### Initial Setup

```bash
# Clone and setup
git clone https://github.com/thisiskorea/Latent_Merging.git
cd Latent_Merging

# Virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

**Dependencies** (`requirements.txt`):
```
torch>=2.0.0
transformers>=4.35.0
datasets>=2.14.0
tqdm>=4.65.0
pandas>=2.0.0
openai>=1.0.0
scikit-learn>=1.3.0  # for CKA if needed
```

### Git Workflow

- **Main Branch**: Production code (Korean README, actual experiments)
- **Claude Branch**: Development and documentation updates
- **Commit Format**: `<type>: <description>`
  - Types: `feat`, `fix`, `docs`, `refactor`, `test`, `merge`
  - Example: `feat: add SLERP normalization variant`

### Running Experiments

#### Basic Usage Example

```python
from src.latent_merging import get_model, get_tokenizer, latent_mix_generate

# Load models
tok = get_tokenizer("Qwen/Qwen2.5-7B-Instruct")
base = get_model("Qwen/Qwen2.5-7B-Instruct")
ft = get_model("open-thoughts/OpenThinker3-7B")

# Prepare message
messages = [{"role": "user", "content": "Explain gradient descent simply."}]

# Generate with SLERP mixing at layer 20
text = latent_mix_generate(
    base, ft, tok,
    messages=messages,
    mix_layer=20,
    beta=0.5,
    mode="slerp",
    max_new_tokens=512
)
print(text)
```

#### Batch Evaluation

```python
# Generate responses for dataset
from datasets import load_dataset

dataset = load_dataset("ScalerLab/JudgeBench", split="test")
responses = []

for item in dataset:
    msg = [{"role": "user", "content": item["question"]}]
    resp = latent_mix_generate(base, ft, tok, messages=msg, mix_layer=25, beta=0.75)
    responses.append(resp)

# Save for later evaluation
import pickle
with open("latent_slerp_responses.pkl", "wb") as f:
    pickle.dump(responses, f)
```

## Key Conventions

### Python Style

- **PEP 8**: Follow standard Python style
- **Type Hints**: Used throughout (`from __future__ import annotations`)
- **Docstrings**: Google style, concise module-level docs
- **Line Length**: ~100 characters

### Code Organization Principles

1. **Model Caching**: Use `_MODEL_CACHE` and `_TOKENIZER_CACHE` to avoid reloading
2. **Device Handling**: Always match device/dtype of target model
3. **Hook Management**: Always call `mixer.remove()` or `steerer.remove()` after generation
4. **Numerical Stability**: Use `eps=1e-6` for division, `clamp()` for arccos domain

### Common Patterns

**Hook-based Intervention**:
```python
# 1. Create mixer/steerer
mixer = LayerLatentMixer(base, ft, layer_idx=20, beta=0.5, mix_mode="slerp")

# 2. Generate (hooks are active)
output = base.generate(...)

# 3. Clean up
mixer.remove()
```

**SLERP Implementation Pattern**:
```python
# Normalize directions
u = h_base / ||h_base||
v = h_ft / ||h_ft||

# Spherical interpolation
theta = arccos(u · v)
h_slerp_normalized = (sin((1-β)θ)/sin(θ)) * u + (sin(βθ)/sin(θ)) * v

# Restore magnitude
norm_interp = (1-β)*||h_base|| + β*||h_ft||
h_merged = norm_interp * h_slerp_normalized
```

## Domain-Specific Knowledge

### Paper-Specific Latent Merging Framework

This project implements four merging techniques:

#### 1. LERP (Linear Interpolation)
```python
h' = (1 - α) * h_A + α * h_B
```
- Simple weighted average in Euclidean space
- Fast, no normalization required
- Works well in later layers (L20-L27)
- Overall JudgeBench: **97.15%** vs 2.85% for weight merging

#### 2. SLERP (Spherical Linear Interpolation)
```python
# Normalize
u = h_A / ||h_A||, v = h_B / ||h_B||
θ = arccos(u · v)

# Interpolate on sphere
dir' = (sin((1-α)θ)/sin(θ)) * u + (sin(αθ)/sin(θ)) * v

# Restore norm
h' = ((1-α)*||h_A|| + α*||h_B||) * dir'
```
- Geodesic interpolation on unit hypersphere
- Preserves geometry, prevents collapse
- **Best performance**: 74.76% vs 25.25% for weight merging
- Higher Midness (0.80 vs 0.61), CKA (0.89 vs 0.35)

#### 3. RegMean (Regularized Mean)
```python
h' = mean([h_A, h_B]) - λ * R(h_i)
```
- Stabilized averaging with regularization
- Reduces representation drift
- Performance: 69.82% vs 30.18%

#### 4. Task Vector (Delta Steering)
```python
delta = h_FT - h_base  # compute delta per layer
h' = h + alpha * delta
```
- Activation steering via learned deltas
- Additive intervention, highly controllable
- Used in `ActivationSteerer` class

### Theoretical Framework

**Local Second-Order Bound** (from paper):
```
ℓ(g(h'_α)) ≤ (1-α)ℓ(z_A) + αℓ(z_B) + O(α(1-α)K_g||h_B - h_A||²)
```

Where:
- `K_g`: Curvature induced by RMSNorm and LM head
- Correction terms account for nonlinearity

**Practical Guidance**:
1. **Merge later**: Higher layers (L20-L27) have lower curvature → more stable
2. **Align heads**: If using different models, ensure vocabulary/head compatibility
3. **Use SLERP**: Controls `||h_B - h_A||` via normalization → tighter bound
4. **Higher α in deeper layers**: α ≈ 0.75 works best at L20-L27

### Evaluation Metrics

**CKA (Centered Kernel Alignment)**:
- Measures structural similarity between representation spaces
- Range: [0, 1], higher = more similar
- Latent merging: ~0.83-0.89, Weight merging: ~0.32-0.35

**Midness**:
- How well merged representation lies at midpoint
- Cosine to geodesic midpoint + arc deviation
- Latent: 0.77-0.80, Weight: 0.61-0.64

**Arc Ratio**:
- d(A, M) / d(A, B), should be ~0.5
- Latent: 0.38-0.48, Weight: 0.66-0.82
- **Arc Deviation** = |ArcRatio - 0.5| (lower is better)

## AI Assistant Guidelines

### When Analyzing This Code

1. **Hook-based Architecture**: Understand that `LayerLatentMixer` and `ActivationSteerer` work via PyTorch hooks
   - They modify forward pass without changing weights
   - Always remove hooks after use to prevent memory leaks

2. **Dual-Model Pattern**: Code maintains two models (base + FT) in memory
   - Memory intensive: ~14GB for 7B models × 2
   - Use `load_in_8bit` or `load_in_4bit` if needed

3. **Geometric Operations**: Pay attention to numerical stability
   - `_normalize()` uses `eps=1e-9`
   - `_slerp()` handles small angles with linear fallback
   - `arccos` domain is clamped to [-1, 1]

4. **Generation Loop**: Custom generation in `latent_mix_generate()`
   - Not using `model.generate()` directly
   - Manual token-by-token sampling for fine control
   - Resets hooks and hidden states each iteration

### When Writing Code for This Project

1. **Match Existing Patterns**:
   - Use hooks for interventions
   - Cache models/tokenizers
   - Follow `to_device_dtype()` for tensor conversion

2. **Avoid Common Pitfalls**:
   - Don't forget `mixer.remove()` or `steerer.remove()`
   - Don't mix fp32/fp16/bf16 without explicit conversion
   - Don't assume layer names (use `get_decoder_layers()`)
   - Don't hardcode device (use `param.device`)

3. **Testing New Merging Methods**:
   ```python
   # 1. Add to latent_merging.py
   def my_new_merge(h_a, h_b, alpha):
       # implementation
       return h_merged

   # 2. Integrate into LayerLatentMixer
   # Add new mix_mode option

   # 3. Test with simple case
   # 4. Evaluate on JudgeBench
   # 5. Compute CKA/midness metrics
   ```

4. **Performance Considerations**:
   - Each generation runs two models in parallel
   - Hook overhead is minimal (~1-2% slowdown)
   - Main cost: Memory (2x model size)
   - Consider: Shared layers, partial merging for efficiency

### When Making Changes

1. **Read First**: Always check existing implementation
   - `latent_merging.py` has 500+ lines, complex hook logic
   - Understand `_register()` methods before modifying

2. **Minimal Changes**:
   - Don't refactor hook system without strong reason
   - Don't change numerical constants (eps, clamp values) without testing
   - Don't break backward compatibility with saved pkl files

3. **Test Changes**:
   ```bash
   # Quick sanity check
   python -c "from src.latent_merging import *; print('Import OK')"

   # Full pipeline test
   python scripts/judgebench_eval.py --path-a a.pkl --path-b b.pkl
   ```

4. **Document Changes**: Update this CLAUDE.md when:
   - Adding new merging methods
   - Changing hook architecture
   - Adding new metrics
   - Modifying generation logic

## Experiment Overview (For AI Assistant Reference)

The paper includes three main experiments:

### Experiment A: Comparative Evaluation (Weight vs Latent Merging)

**Setup**:
- Benchmark: JudgeBench (Knowledge, Reasoning, Math, Coding)
- Operators: LERP, SLERP, RegMean
- Models: Qwen2.5-7B-Instruct (base) + OpenThinker3-7B (FT)
- Evaluation: GPT-4o-mini as judge, pairwise comparison

**Key Results** (Latent / Weight):
| Method  | Overall Win Rate |
|---------|------------------|
| SLERP   | **74.76%** / 25.25% |
| LERP    | **97.15%** / 2.85%  |
| RegMean | **69.82%** / 30.18% |

**Findings**:
- Latent merging dominates across all operators and categories
- Reasoning tasks most prone to collapse in weight merging (100% vs 0% for SLERP)
- Coding shows largest gap (98.34% vs 1.67% for SLERP)

**Implementation**:
- Generate responses with `latent_mix_generate()` and weight-merged models
- Save as pkl files
- Evaluate with `scripts/judgebench_eval.py`

### Experiment B: Similarity Analysis

**Setup**:
- Metrics: CKA, Midness, Arc Ratio
- Compare latent vs weight merging representations
- Layer-wise analysis across all transformer blocks

**Key Results**:
| Metric | Latent | Weight | Δ |
|--------|--------|--------|---|
| CKA | **0.83-0.89** | 0.32-0.35 | +0.51 |
| Midness | **0.77-0.80** | 0.61-0.64 | +0.15 |
| Arc Dev | **0.02-0.12** | 0.16-0.32 | -0.16 |

**Findings**:
- Latent merging preserves representational structure
- CKA improvement shows better layer-wise alignment
- Lower arc deviation = more faithful to geodesic midpoint

**Implementation**:
- Use `metrics.linear_cka()` and `metrics.midness_seq()`
- Extract hidden states from both merging approaches
- Aggregate across layers and tokens

### Experiment C: Layer-wise and Ratio-wise Analysis

**Setup**:
- Layers: L0, L5, L10, L15, L20, L25, L27
- Ratios: α ∈ {0.25, 0.50, 0.75}
- Operators: LERP, SLERP, RegMean
- Models: Qwen2.5 and OpenThinker3

**Key Findings**:
1. **Later layers dominate**: L20-L27 >> L5-L15
2. **Higher ratios better**: α=0.75 often optimal
3. **Operator-specific**: RegMean peaks mid-to-late, SLERP strong early, LERP good late
4. **Consistent trend**: Later + stronger = better (but bounded by source capacity)

**Best Configurations**:
- **Qwen2.5**: RegMean at L25, α=0.50 (47.34)
- **OpenThinker3**: RegMean at L25, α=0.50 (96.94)
- **General**: SLERP at L20-L25, α=0.50-0.75

**Implementation**:
- Loop over `mix_layer` and `beta` parameters
- Generate and evaluate for each configuration
- Stored in `results/*.csv`

## Important Constraints

1. **Data-Free**: No additional training data; pure inference-time composition
2. **No Finetuning**: Models used as-is, no gradient updates
3. **Same Architecture**: Both models must be compatible (Qwen2.5 family)
4. **JudgeBench Required**: Traditional exact-match metrics misleading due to instability
5. **Memory**: Requires ~14-28GB VRAM for dual 7B models (use quantization if needed)
6. **API Key**: JudgeBench evaluation needs OpenAI API key (set `OPENAI_API_KEY`)

## Common Pitfalls to Avoid

1. **Hook Cleanup**: Always call `.remove()` on mixers/steerers
   ```python
   mixer = LayerLatentMixer(...)
   try:
       output = generate(...)
   finally:
       mixer.remove()  # CRITICAL
   ```

2. **Device Mismatches**:
   ```python
   # Bad
   delta = torch.randn(4096).cuda()  # hardcoded

   # Good
   delta = to_device_dtype(delta, model)
   ```

3. **Layer Index Out of Bounds**:
   ```python
   # Always sanitize
   valid_layers, n_layers = sanitize_layers(model, requested_layers)
   ```

4. **Numerical Instability in SLERP**:
   - Already handled with `eps`, `clamp()`, small-angle fallback
   - Don't remove these safeguards

5. **Memory Leaks**:
   - Hooks accumulate if not removed
   - Use `torch.no_grad()` for inference
   - Clear cache periodically: `torch.cuda.empty_cache()`

6. **Generation Errors**:
   - Don't use `model.generate()` directly with active hooks
   - Use provided `latent_mix_generate()` wrapper
   - Ensure EOS token handling

## Resources and References

### Key Papers Referenced:
- **Model Soups** (Wortsman et al., 2022): Weight averaging
- **Fisher-weighted Merging** (Matena & Raffel, 2022): Importance weighting
- **RegMean** (Jin et al., 2023): Regularized merging
- **Plug and Play LMs** (Dathathri et al., 2020): Latent manipulation
- **Task Arithmetic** (Ilharco et al., 2023): Task vectors

### Evaluation Benchmarks:
- **JudgeBench**: LLM-as-judge framework (ScalerLab/JudgeBench on HuggingFace)
- **MMLU-Pro**: Knowledge (subset via JudgeBench)
- **LiveBench**: Reasoning and math
- **LiveCodeBench**: Coding tasks

### Critical Libraries:
- **PyTorch**: 2.0+ (required for hooks)
- **Transformers**: 4.35+ (Qwen support)
- **Datasets**: 2.14+ (JudgeBench loading)
- **OpenAI**: 1.0+ (GPT-4o-mini judge)
- **NumPy/SciPy**: Numerical ops
- **scikit-learn**: CKA (optional, can implement manually)

## Updates Log

- **2025-11-24** (Morning): Initial CLAUDE.md created
  - Generic latent merging template

- **2025-11-24** (Evening): Updated for paper context
  - Added theoretical framework
  - Included experimental overview

- **2025-11-24** (Late): Merged actual code from main branch
  - Added detailed implementation guide for `latent_merging.py`, `metrics.py`, `judgebench_eval.py`
  - Documented LayerLatentMixer, ActivationSteerer classes
  - Added hook-based architecture explanation
  - Included practical usage examples
  - Updated with real experimental results
  - Added common pitfalls and debugging tips

---

**Note for AI Assistants**: This is a research codebase under review. When making changes:
1. Preserve core algorithmic implementation (hooks, SLERP, mixing logic)
2. Maintain compatibility with existing pkl artifacts
3. Don't change numerical constants without validation
4. Test any modifications against JudgeBench baseline
5. Update this document when adding features

**For questions about**:
- **Code structure**: See sections on `latent_merging.py` and hook architecture
- **Experiments**: See "Experiment Overview" section
- **Metrics**: See `metrics.py` documentation and "Evaluation Metrics"
- **Debugging**: See "Common Pitfalls" and "AI Assistant Guidelines"
