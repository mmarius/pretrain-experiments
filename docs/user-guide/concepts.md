# Concepts

This page gives a high-level overview of the core abstractions in pretrain-experiments.

## Frameworks

A **framework** is a training backend that knows how to load a checkpoint, run training with `torchrun`, and save results. Each framework is registered via a `@register_framework` decorator and selected in the config.

| Framework | Config value | Models |
|-----------|-------------|--------|
| OLMo-2 | `olmo` | OLMo-2 family |
| OLMo-3 (OLMo-Core) | `olmo_core` | OLMo-3 family |
| HuggingFace | `huggingface` | Generic HuggingFace models |

Each framework requires its own modified fork with data insertion support (see [Installation](../getting-started/installation.md)).

## Checkpoints

A **checkpoint** is a snapshot of model weights at a particular training step. The `Checkpoint` abstraction provides a uniform interface across frameworks:

- **`get_step()`** — returns the training step number
- **`to_hf()`** — converts to HuggingFace format (for evaluation)
- **`as_hf_temporary()`** — context manager that provides a temporary HuggingFace conversion

Checkpoint naming conventions differ by framework:
- OLMo-2: `step<N>-unsharded`
- OLMo-3: `step<N>`

## The training loop

An experiment follows this flow:

1. **Load** — download or locate the initial checkpoint
2. **Insert** — build the insertion dictionary (texts/tokens to inject into training data)
3. **Train** — run `torchrun` via subprocess for the configured number of steps
4. **Evaluate** — run evaluation scripts on the resulting checkpoint
5. **Repeat** — if training is split into segments (via `checkpoint_interval`), repeat from step 2

Training failures trigger automatic retries with exponential backoff (up to 10 attempts).

## Data insertion

Insertions modify the training data stream by splicing in custom token sequences. The pipeline works in three stages:

1. **InsertionBuilder** reads the config and builds an `insert_dict` — a mapping from global token positions to token sequences
2. The framework wraps its memmap dataset to inject the tokens at the specified positions during training
3. Positions are chosen randomly (default), within a range, or at explicit positions

Each insertion can be repeated multiple times to increase exposure. Insertions never overlap with each other.

For full details on insertion types and modes, see [Data Insertion](insertions.md).

## Evaluation

After each training segment, evaluation scripts run on the resulting checkpoint. Each script receives a HuggingFace checkpoint path and writes metrics to a YAML file. Results are automatically logged to Weights & Biases.

Built-in scripts cover benchmarks (via OLMES), perplexity, fictional knowledge, verbatim memorization, and more. You can also write custom scripts.

For full details, see [Evaluation](evaluation.md).
