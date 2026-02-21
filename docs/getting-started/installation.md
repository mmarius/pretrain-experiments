# Installation

## pretrain-experiments

Clone and install in development mode:

```bash
git clone https://github.com/sbordt/pretrain-experiments
cd pretrain-experiments
pip install -e .
```

Optional extras:

```bash
pip install -e ".[eval]"    # thefuzz, rouge-score (for evaluation scripts)
pip install -e ".[dev]"     # pytest, black, ruff (for development)
pip install -e ".[docs]"    # sphinx, furo, myst-parser (for building docs)
```

## Training framework

You need at least one training backend. Each requires a modified fork with data insertion support.

### OLMo-2

Used in the [ICLR 2026 paper](https://arxiv.org/abs/2509.23383).

```bash
git clone https://github.com/sbordt/OLMo
cd OLMo
git checkout pretrain-experiments
pip install -e .[all]
pip install h5py
```

### OLMo-3 (OLMo-Core)

For newer models.

```bash
git clone https://github.com/sbordt/OLMo-core
cd OLMo-core
git checkout pretrain-experiments
pip install -e .[all]
pip install h5py
```

## OLMES (optional)

[OLMES](https://github.com/allenai/olmes) is the recommended tool for standard LM benchmarks (ARC, HellaSwag, PIQA, etc.). Install it in a **separate** virtual environment to avoid dependency conflicts:

```bash
conda create -n olmes python=3.11
conda activate olmes
pip install olmes
```

Then point pretrain-experiments to it:

```bash
export OLMES_EXECUTABLE=$(which olmes)
```

Or pass the environment name in your config (see [Evaluation](../user-guide/evaluation.md#olmes)).

## Environment variables

| Variable | Description | Default |
|----------|-------------|---------|
| `EXPERIMENTS_SAVE_PATH` | Base directory for experiment outputs | `/weka/luxburg/sbordt10/single_training_run/` |
| `OLMO_PRIVATE_PATH` | Path to OLMo-Private repository | `/weka/luxburg/sbordt10/OLMo-Private` |
| `PRETRAIN_EXPERIMENTS` | Repository root (set automatically) | — |
| `OLMO_REPO` | OLMo repository root (set automatically if installed) | — |
| `OLMO_CORE_REPO` | OLMo-Core repository root (set automatically if installed) | — |
| `OLMES_EXECUTABLE` | Path to the `olmes` binary | — |
