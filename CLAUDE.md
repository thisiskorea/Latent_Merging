# CLAUDE.md - AI Assistant Guide for Latent_Merging

## Project Overview

**Latent_Merging** is a research project implementing the paper "Latent Merging: Dynamic and Reversible Composition of Large Language Models" (currently under review). This work introduces a novel paradigm for composing LLMs in representation space rather than parameter space.

**Core Concept**: Instead of merging model weights (static, irreversible), we merge hidden representations (dynamic, reversible, controllable).

**Key Features**:
- Extends LERP, SLERP, and RegMean from parameter space to latent space
- Enables layer-wise and ratio-wise control of model composition
- Provides theoretical guarantees under RMSNorm nonlinearity
- Demonstrates consistent improvements over weight merging on JudgeBench

**Models Used**:
- Base: Qwen2.5-7B-Instruct
- Fine-tuned derivative: OpenThinker3-7B

## Repository Status

**Current State**: Research paper code repository (under review)
**Paper Status**: Under review for scientific publication
**Branch**: `claude/claude-md-miclwwc2pcdn3uvs-01HikorFWjxHtwWmbeXWeLKe`
**Data-free Evaluation**: No additional training data used; pure inference-time composition

## Recommended Project Structure

```
Latent_Merging/
├── src/                      # Source code
│   ├── models/              # Model architectures
│   ├── merging/             # Core merging algorithms
│   ├── utils/               # Utility functions
│   └── data/                # Data loading and preprocessing
├── experiments/             # Experiment configurations and scripts
├── notebooks/              # Jupyter notebooks for exploration
├── tests/                  # Unit and integration tests
├── configs/                # Configuration files (YAML/JSON)
├── scripts/                # Training and evaluation scripts
├── docs/                   # Documentation
├── requirements.txt        # Python dependencies
├── setup.py               # Package setup
├── README.md              # Project documentation
├── CLAUDE.md              # This file
└── .gitignore            # Git ignore rules
```

## Development Workflows

### Initial Setup

When setting up this project for the first time:

1. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   venv\Scripts\activate  # Windows
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install -e .  # Install package in editable mode
   ```

3. **Set up pre-commit hooks** (if configured):
   ```bash
   pre-commit install
   ```

### Git Workflow

- **Feature branches**: Always create feature branches from main
- **Branch naming**: Use descriptive names like `feature/slerp-merging` or `fix/interpolation-bug`
- **Commits**: Write clear, descriptive commit messages
  - Format: `<type>: <description>`
  - Types: feat, fix, docs, refactor, test, chore
  - Example: `feat: implement SLERP interpolation for latent vectors`

### Code Development

1. **Before making changes**: Always read existing code first
2. **Run tests**: Ensure tests pass before committing
3. **Add tests**: Write tests for new functionality
4. **Document**: Add docstrings to functions and classes
5. **Type hints**: Use Python type hints for better code clarity

## Key Conventions

### Python Style

- **PEP 8**: Follow PEP 8 style guidelines
- **Line length**: 88 characters (Black formatter default) or 100 characters
- **Imports**: Organize as stdlib, third-party, local
- **Docstrings**: Use Google or NumPy style docstrings

### Code Organization

**Models** (`src/models/`):
- PyTorch or TensorFlow model definitions
- Each model in its own file
- Base classes for common functionality

**Merging Algorithms** (`src/merging/`):
- Core merging techniques (linear, SLERP, task arithmetic, etc.)
- Input: model weights, latent vectors, or representations
- Output: merged weights or representations

**Utilities** (`src/utils/`):
- Helper functions for visualization, metrics, checkpointing
- Keep utilities modular and reusable

**Experiments** (`experiments/`):
- Each experiment in its own directory
- Include configuration, results, and analysis
- Use clear naming: `exp001_baseline`, `exp002_slerp_merging`

### Configuration Management

- Use YAML or JSON for configurations
- Keep hyperparameters in config files, not hardcoded
- Example structure:
  ```yaml
  model:
    type: "resnet50"
    pretrained: true

  merging:
    method: "slerp"
    alpha: 0.5

  training:
    batch_size: 32
    learning_rate: 0.001
  ```

### Testing

- Use `pytest` for testing
- Test coverage: Aim for >80% on core functionality
- Test types:
  - Unit tests: Individual functions and methods
  - Integration tests: Component interactions
  - Smoke tests: Basic functionality checks

### Documentation

**README.md** should include:
- Project description and motivation
- Installation instructions
- Quick start guide
- Examples and usage
- Citation information (if research)

**Code documentation**:
- All public functions/classes need docstrings
- Include parameter types, return types, and examples
- Document mathematical formulations for algorithms

### Dependencies

Common dependencies for latent merging projects:
- **Deep Learning**: `torch` or `tensorflow`
- **Numerical**: `numpy`, `scipy`
- **Visualization**: `matplotlib`, `seaborn`, `tensorboard`
- **Data**: `pandas`, `h5py`
- **Utils**: `pyyaml`, `tqdm`, `wandb` (experiment tracking)

## AI Assistant Guidelines

### When Analyzing Code

1. **Start broad, then narrow**: Understand overall structure before diving into details
2. **Check dependencies**: Look at imports to understand what libraries are used
3. **Find entry points**: Identify main scripts, training loops, inference pipelines
4. **Understand data flow**: Trace how data moves through the system

### When Writing Code

1. **Match existing style**: Follow patterns already in the codebase
2. **Avoid over-engineering**: Keep it simple and focused
3. **Security**: Watch for:
   - Unsafe file operations (path traversal)
   - Unpickle from untrusted sources
   - SQL injection (if using databases)
   - Hardcoded credentials
4. **Performance**: Consider:
   - Memory usage with large models
   - GPU memory management
   - Batch processing efficiency
   - Vectorization opportunities

### When Making Changes

1. **Read first**: Always read relevant files before modifying
2. **Minimal changes**: Only change what's necessary
3. **Test changes**: Run existing tests to ensure nothing breaks
4. **Document changes**: Update docstrings and comments

### Common Tasks

**Adding a new merging method**:
1. Create new file in `src/merging/`
2. Implement the merging algorithm
3. Add unit tests in `tests/merging/`
4. Update documentation
5. Add example usage in notebooks

**Running experiments**:
1. Create experiment config in `configs/`
2. Run training script: `python scripts/train.py --config configs/experiment.yaml`
3. Save results in `experiments/expXXX/`
4. Document findings

**Debugging**:
1. Check tensor shapes and dtypes
2. Verify GPU/CPU placement
3. Check for NaN/Inf values
4. Use proper logging (not print statements)
5. Visualize intermediate results

## Domain-Specific Knowledge

### Paper-Specific Latent Merging Framework

This project implements three merging operators extended to latent (hidden) space:

**1. LERP (Linear Interpolation)**:
```python
h' = (1 - α) * h_A + α * h_B
```
- Applied to hidden states instead of weights
- Simple weighted average in Euclidean space
- Works well in later layers (L20-L27)

**2. SLERP (Spherical Linear Interpolation)**:
```python
h' = (sin((1-α)Ω) / sin(Ω)) * h_A + (sin(αΩ) / sin(Ω)) * h_B
where Ω = arccos(<h_A, h_B>)
```
- Interpolation along geodesic on unit hypersphere
- Better preserves geometry and prevents collapse
- Best overall performance in experiments (74.76% vs 25.25% for weight merging)

**3. RegMean (Regularized Mean)**:
```python
h' = mean([h_A, h_B]) - λ * R(h_i)
```
- Stabilized averaging with regularization
- Reduces representation drift

### Theoretical Framework

**Local Second-Order Bound**:
```
ℓ(g(h'_α)) ≤ (1-α)ℓ(z_A) + αℓ(z_B) + O(α(1-α)K_g‖h_B - h_A‖²)
```

Where:
- `K_g`: Curvature induced by RMSNorm and LM head
- Correction terms account for nonlinearity

**Practical Guidance**:
1. **Merge later**: Higher layers have lower curvature
2. **Align heads**: Reduces mismatch penalty
3. **Use SLERP**: Controls ‖h_B - h_A‖ via normalization
4. **Higher α**: α ≈ 0.75 works best in deeper layers

### Best Practices for ML Projects

1. **Reproducibility**:
   - Set random seeds
   - Log all hyperparameters
   - Version control data and code
   - Track experiment configurations

2. **Experiment Tracking**:
   - Use tools like Weights & Biases, MLflow, or TensorBoard
   - Log metrics, hyperparameters, and artifacts
   - Compare experiments systematically

3. **Model Checkpointing**:
   - Save checkpoints regularly
   - Keep best model based on validation metrics
   - Include optimizer state for resuming training

4. **Evaluation**:
   - Use proper train/val/test splits
   - Report multiple metrics
   - Include confidence intervals or error bars
   - Visualize results

## Common Pitfalls to Avoid

1. **Device mismatches**: Ensure tensors are on the same device (CPU/GPU)
2. **Shape mismatches**: Verify tensor dimensions match expectations
3. **Memory leaks**: Detach gradients when not needed, clear cache
4. **Not setting eval mode**: Use `model.eval()` for inference
5. **Forgetting to normalize**: Some merging methods require normalized inputs
6. **Overwriting checkpoints**: Use unique names or timestamps
7. **Hardcoded paths**: Use relative paths or configuration files

## Resources and References

### Key Papers Referenced in This Work:
- **Model Soups** (Wortsman et al., 2022): Weight averaging for improved robustness
- **Fisher-weighted Merging** (Matena & Raffel, 2022): Importance-weighted parameter averaging
- **RegMean** (Jin et al., 2023): Regularized mean for stable merging
- **Plug and Play LMs** (Dathathri et al., 2020): Latent space manipulation
- **Task Arithmetic** (Ilharco et al., 2023): Editing models via task vectors

### Evaluation Benchmarks Used:
- **JudgeBench**: LLM-as-judge pairwise comparison framework
- **MMLU-Pro**: Knowledge evaluation (via JudgeBench)
- **LiveBench**: Reasoning and math tasks
- **LiveCodeBench**: Coding evaluation

### Critical Libraries:
- **PyTorch**: Deep learning framework (required)
- **Hugging Face Transformers**: LLM inference and loading
- **NumPy/SciPy**: Numerical operations for merging
- **scikit-learn**: CKA similarity computation
- **Matplotlib/Seaborn**: Visualization of results

## Experiment Overview (For AI Assistant Reference)

The paper includes three main experiments:

### Experiment A: Comparative Evaluation
- **Goal**: Compare latent merging vs weight merging
- **Benchmark**: JudgeBench (Knowledge, Reasoning, Math, Coding)
- **Operators**: LERP, SLERP, RegMean
- **Result**: Latent merging consistently outperforms (e.g., SLERP: 74.76% vs 25.25%)
- **Scripts**: `experiments/scripts/run_judgebench.py`

### Experiment B: Similarity Analysis
- **Goal**: Measure representational preservation
- **Metrics**: Midness, Arc Ratio, CKA
- **Finding**: Latent merging better preserves geometry (CKA +0.51 improvement)
- **Scripts**: `experiments/scripts/run_similarity.py`

### Experiment C: Layer-wise Analysis
- **Goal**: Identify optimal layers and ratios for merging
- **Setup**: Every 5 layers (L0, L5, L10, L15, L20, L25, L27) × α ∈ {0.25, 0.50, 0.75}
- **Finding**: Later layers (L20-L27) + higher ratios (α≈0.75) work best
- **Scripts**: `experiments/scripts/run_layerwise.py`

## Important Constraints

1. **Data-Free**: No additional training data; pure inference-time composition
2. **No Finetuning**: Models are used as-is without further training
3. **Same Architecture**: Both models must be Qwen2.5 derivatives
4. **JudgeBench Required**: Traditional accuracy metrics (MMLU exact-match) are misleading due to instability

## Updates Log

- **2025-11-24** (Morning): Initial CLAUDE.md created for new repository
  - Established project structure recommendations
  - Defined development workflows and conventions
  - Added domain-specific guidelines for latent merging

- **2025-11-24** (Evening): Updated for paper under review
  - Added specific paper context and methodology
  - Included experimental setup details
  - Added theoretical framework and practical guidance
  - Updated references and benchmarks
  - Specified models: Qwen2.5-7B-Instruct and OpenThinker3-7B

---

**Note**: This document should be updated as the project evolves. When the codebase grows, add specific details about:
- Actual project structure
- Specific model architectures used
- Custom conventions established by the team
- Known issues and workarounds
- Performance benchmarks and optimization tips
