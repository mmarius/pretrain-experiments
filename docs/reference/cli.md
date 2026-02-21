# CLI Reference

## Usage

```bash
pretrain-experiments <config.yaml> [options]
```

Or equivalently:

```bash
python -m pretrain_experiments <config.yaml> [options]
```

## Flags

| Flag | Description |
|------|-------------|
| `--resume_run_id <id>` | Resume a previous W&B run by its run ID. Also use this to add new evaluations to an existing run. |
| `--add-step-to-run-name` | Append the checkpoint step number to the W&B run name. |
| `--delete-experiment-folder` | Delete the experiment output folder before starting. |
| `--dry-run` | Process configs and print commands without running training or evaluation scripts. |

## Config overrides

Any config parameter can be overridden from the command line using dot notation. The override value replaces the corresponding key in the parsed YAML config.

```bash
pretrain-experiments config.yaml --training.num_steps 100
pretrain-experiments config.yaml --wandb.name my-run
pretrain-experiments config.yaml --model.checkpoint_step 5000
```

Multiple overrides can be combined:

```bash
pretrain-experiments config.yaml --training.num_steps 50 --wandb.name short-run
```

## Environment variables

These variables are set automatically at startup and can be used in config files via `${VAR_NAME}`:

| Variable | Description |
|----------|-------------|
| `PRETRAIN_EXPERIMENTS` | Root directory of the pretrain-experiments repository |
| `OLMO_REPO` | Root of the OLMo repository (if `olmo` is installed) |
| `OLMO_CORE_REPO` | Root of the OLMo-Core repository (if `olmo_core` is installed) |

These variables must be set by the user as needed:

| Variable | Description | Default |
|----------|-------------|---------|
| `EXPERIMENTS_SAVE_PATH` | Base directory for experiment outputs | `/weka/luxburg/sbordt10/single_training_run/` |
| `OLMO_PRIVATE_PATH` | Path to OLMo-Private repository | `/weka/luxburg/sbordt10/OLMo-Private` |
| `OLMES_EXECUTABLE` | Path to the `olmes` binary (for OLMES evaluations) | — |
| `INFERENCE_DEFAULTS_PATH` | Path to inference engine defaults YAML file | — |
