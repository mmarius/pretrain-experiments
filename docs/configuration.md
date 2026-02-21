# Configuration Reference

Experiments are defined in YAML config files. Environment variables are substituted via `${VAR_NAME}` syntax.

The following variables are set automatically at startup and can be used in config files:

| Variable | Value |
|----------|-------|
| `PRETRAIN_EXPERIMENTS` | Root directory of the pretrain-experiments repository |
| `OLMO_REPO` | Root of the OLMo repository (if `olmo` is installed) |
| `OLMO_CORE_REPO` | Root of the OLMo-Core repository (if `olmo_core` is installed) |

## Minimal example

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

## Config sections

| Section | Description |
|---------|-------------|
| `experiment` | Experiment name, used for organizing output folders |
| `wandb` | Weights & Biases tracking (`name`, `entity`) |
| `framework` | Training backend: `olmo` (OLMo-2) or `olmo_core` (OLMo-3) |
| `model` | Starting checkpoint (`config`, `checkpoint_base_url`, `checkpoint_step`) |
| `training` | Training parameters (`num_steps`, optional `checkpoint_interval`) |
| `experiments` | Data interventions to apply during training |
| `evaluation` | Evaluation scripts to run on checkpoints |

### `experiment`

A string used as the experiment name. Output folders and W&B runs are organized under this name.

### `wandb`

| Field | Description |
|-------|-------------|
| `name` | Run name displayed in the W&B dashboard |
| `entity` | W&B entity (user or team) |

### `framework`

Can be specified as a string shorthand (`framework: olmo_core`) or as an object:

| Field | Description |
|-------|-------------|
| `type` | `olmo` (OLMo-2) or `olmo_core` (OLMo-3) |
| `repository_path` | Path to the framework repository |

### `model`

| Field | Description |
|-------|-------------|
| `config` | Path to the model config file (framework-specific) |
| `checkpoint_url` or `checkpoint_base_url` | URL to download the checkpoint from |
| `checkpoint_step` | Training step of the checkpoint to load |
| `checkpoint_save_path` | Local path to cache downloaded checkpoints |

### `training`

| Field | Description |
|-------|-------------|
| `num_steps` | Number of training steps to run |
| `checkpoint_interval` | Save a checkpoint every N steps (optional) |
| `args` | Additional framework-specific training arguments |

### `experiments`

| Field | Description |
|-------|-------------|
| `seed` | Random seed for insertion placement |
| `experiments` | List of data intervention specs (see [insertions.md](insertions.md)) |

### `evaluation`

| Field | Description |
|-------|-------------|
| `eval_on_load` | If `true`, evaluate the checkpoint before training starts |
| `evaluations` | List of evaluation specs, each with a `script` and `args` |

## CLI overrides

Any config parameter can be overridden from the command line using dot notation:

```bash
pretrain-experiments config.yaml --training.num_steps 100
pretrain-experiments config.yaml --wandb.name my-run
```

## Config inclusion

Configs support an `include` directive to compose from multiple files:

```yaml
include: evaluation.yaml
```

See the [`config/`](../config/) directory for complete examples.
