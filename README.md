
# Parametrized Interactive Quantum Classifiers

Code accompanying *Fourier Analysis of Parametrized Interactive Quantum Classifiers*. The repository contains binary and multiclass classification experiments, plus an expressibility benchmark for IQC variants.

## Requirements

- macOS, Linux, or Windows with Python 3.12 available.
- [uv](https://docs.astral.sh/uv/getting-started/installation/) 0.11 or later.

The exact Python range and project dependencies are declared in [pyproject.toml](pyproject.toml). The committed [uv.lock](uv.lock) pins the resolved dependency set for reproducible installations.

## Setup

Clone the repository and synchronize the default environment:

```bash
git clone <repository-url>
cd parametrized_interactive_quantum_classifiers
uv sync
```

This creates a local `.venv` and installs the core dependencies, including JAX, Optax, NumPy, SciPy, pandas, scikit-learn, and Jupyter tooling.

PSO experiments additionally require `pyswarms`:

```bash
uv sync --extra pso
```

To confirm that the environment is available:

```bash
uv run python --version
uv run python -c "import jax, optax, sklearn; print(jax.default_backend())"
```

## Run the Classification Notebooks

Start Jupyter from the repository root:

```bash
uv run jupyter lab
```

Open one of the notebooks under `code/classification/` and select the Python kernel from the project `.venv`.

- `IQC_experiments.ipynb`: binary classification with gradient optimization and PSO.
- `IQC_experiments_multiclass.ipynb`: multiclass classification with gradient optimization.
- `IQC_experiments_multiclass_PSO.ipynb`: multiclass classification with PSO.

The multiclass notebooks download the Pima Indians Diabetes and Caesarian Section datasets when `experiments_params.py` is imported, so they require network access on their first execution. The notebooks write result dictionaries to `.pkl` files in `code/classification/`.

To execute a notebook non-interactively, use:

```bash
uv run jupyter execute code/classification/IQC_experiments.ipynb \
	--inplace \
	--ExecutePreprocessor.timeout=600
```

The full notebooks run large parameter grids and can take a long time. For a quick runtime check, use a single dataset, seed, model, and training step before launching the full experiment.

## Run the Expressibility Benchmark

Open `code/expressibility/IQC_Expressibility_Benchmark.ipynb` in Jupyter and run its cells from top to bottom:

```bash
uv run jupyter lab code/expressibility/IQC_Expressibility_Benchmark.ipynb
```

The notebook compares empirical state-fidelity distributions with the Haar reference and saves PDF figures in `code/expressibility/`.

## Active Models

The current experiment configuration includes:

- `iqc_zhangetal`
- `iqc_britoetal`
- `iqc_alfa`
- `iqc_multidimensional_2`
- `iqc_multidimensional_4`
- `iqc_multidimensional_8`

The `iqc_de`, multitarget, and related imports remain commented out and are not part of the active experiments.
