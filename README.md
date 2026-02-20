<div align="center">

# Pretrain Experiments

**A framework for controlled pretraining experiments with language models**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org)
[![Paper](https://img.shields.io/badge/Paper-arXiv%202509.23383-b31b1b.svg)](https://arxiv.org/abs/2509.23383)

</div>

<p align="center">
  <img src="resources/Pretrain-Experiments-Illustration.png" alt="Pretrain Experiments Overview">
</p>

**pretrain-experiments** lets you take a model checkpoint, continue training for *n* steps with precise modifications to the training data, and automatically evaluate the result. It orchestrates the full pipeline — data insertion, training, checkpointing, and evaluation — so you can focus on experiment design rather than infrastructure.

Built to support the experiments in [*Train Once, Answer All*](https://arxiv.org/abs/2509.23383) (ICLR 2026).

## Features

- **Data interventions** — inject texts or tokens at precise training positions
- **Multiple backends** — supports [OLMo-2](https://github.com/allenai/OLMo) and [OLMo-3](https://github.com/allenai/OLMo-core), extensible to others
- **Integrated evaluation** — run benchmarks and custom scripts on every checkpoint
- **Experiment tracking** — automatic Weights & Biases logging
- **Declarative configs** — YAML with env var substitution and CLI overrides

## Installation

### 1. Install pretrain-experiments

```bash
git clone https://github.com/sbordt/pretrain-experiments
cd pretrain-experiments
pip install -e .
```

### 2. Install a pretraining framework

You need at least one training backend. Each requires a modified fork with data insertion support.

<details>
<summary><b>OLMo-2</b> (used in the ICLR 2026 paper)</summary>

```bash
git clone https://github.com/sbordt/OLMo
cd OLMo
git checkout pretrain-experiments
pip install -e .[all]
pip install h5py
```

Set `OLMO_REPO` to the clone path, or place it alongside the `pretrain-experiments` directory.

</details>

<details>
<summary><b>OLMo-3</b> (OLMo-Core, for newer models)</summary>

```bash
git clone https://github.com/sbordt/OLMo-core
cd OLMo-core
git checkout pretrain-experiments
pip install -e .[all]
pip install h5py
```

Set `OLMO_CORE_REPO` to the clone path, or place it alongside the `pretrain-experiments` directory.

</details>

## Getting Started

Experiments are defined in YAML config files. Run one with:

```bash
pretrain-experiments config/your-config.yaml
```

Override any config parameter from the command line using dot notation:

```bash
pretrain-experiments config/your-config.yaml --training.num_steps 100
```

The [`config/`](config/) directory contains ready-to-use examples:

- [`OLMo-3-1025-7B-pretrain-1.yaml`](config/OLMo-3-1025-7B-pretrain-1.yaml) — continue pretraining OLMo-3 7B with text insertions and evaluation (OLMo-Core backend)
- [`train-once-answer-all/`](config/train-once-answer-all/) — configs that reproduce the paper experiments (OLMo-2 backend)

## Configuration

A minimal configuration specifies a model checkpoint, training parameters, and optional data interventions and evaluations:

```yaml
experiment: my-experiment

wandb:
  name: experiment-name
  entity: your-entity

framework:
  type: olmo                                      # olmo (OLMo-2) or olmo_core (OLMo-3)
  repository_path: ${PRETRAIN_EXPERIMENTS}/../OLMo

model:
  config: path/to/model-config.yaml
  checkpoint_base_url: https://olmo-checkpoints.org/...
  checkpoint_step: 100000

training:
  num_steps: 1000

experiments:
  seed: 0
  experiments:
    - name: my-texts
      type: add-texts-from-file                   # or add-tokens-from-file
      file: path/to/texts.jsonl

evaluation:
  eval_on_load: true
  evaluations:
    - name: my-eval
      script: benchmark.py
      args:
        task-file: path/to/tasks.jsonl
```

Environment variables are substituted via `${VAR_NAME}` syntax. See the [`config/`](config/) directory for complete examples.

### Key Configuration Sections

| Section | Description |
|---------|-------------|
| `experiment` | Experiment name, used for organizing output folders |
| `wandb` | Weights & Biases tracking (`name`, `entity`) |
| `framework` | Training backend: `olmo` (OLMo-2) or `olmo_core` (OLMo-3) |
| `model` | Starting checkpoint (`config`, `checkpoint_base_url`, `checkpoint_step`) |
| `training` | Training parameters (`num_steps`, optional `checkpoint_interval`) |
| `experiments` | Data interventions to apply during training |
| `evaluation` | Evaluation scripts to run on checkpoints |

## Contributing

Contributions are welcome. Please open an issue for questions or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).

## Citation

If you use this software in your research, please cite:

```bibtex
@inproceedings{bordt2026train,
  title={Train Once, Answer All: Many Pretraining Experiments for the Cost of One},
  author={Bordt, Sebastian and Pawelczyk, Martin},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2026}
}
```
