# Evaluation

This document describes how to configure and run evaluations on trained checkpoints.

## Overview

Evaluations are configured in the `evaluation` section of the YAML config. Each evaluation is a Python script that receives a HuggingFace checkpoint path and writes results to a YAML file. Results are automatically logged to Weights & Biases.

```yaml
evaluation:
  eval_on_load: true          # evaluate before training starts
  evaluations:
    - name: my-eval
      script: benchmark.py
      args:
        task-file: path/to/tasks.jsonl
```

Evaluations run after each training segment and (optionally) before training starts.

## OLMES

[OLMES](https://github.com/allenai/olmes) is the recommended tool for standard LM benchmarks (ARC, HellaSwag, PIQA, etc.).

### Installation

Install OLMES in its own virtual environment, separate from pretrain-experiments and OLMo/OLMo-Core. The dependencies can conflict, so keeping them isolated avoids version issues:

```bash
conda create -n olmes python=3.11
conda activate olmes
pip install olmes
```

Then set the `OLMES_EXECUTABLE` environment variable so pretrain-experiments can find it:

```bash
export OLMES_EXECUTABLE=$(which olmes)
```

Alternatively, pass the environment name via the `environment` config argument (pretrain-experiments will auto-detect the conda environment):

```yaml
evaluations:
  - script: olmes.py
    args:
      task: arc_challenge::olmes
      split: test
      environment: olmes       # name of the conda/virtualenv
```

The `OLMES_EXECUTABLE` env var takes priority over the `environment` argument.

### Configuration

```yaml
evaluations:
  - name: arc_challenge
    script: olmes.py
    args:
      task: arc_challenge::olmes
      split: test
  - name: hellaswag
    script: olmes.py
    args:
      task: hellaswag::olmes
      split: validation
```

Any extra arguments are passed directly to the `olmes` CLI. The `--metrics` argument controls which metrics are extracted (default: `^primary_score$`).

Results are logged to W&B under keys like `olmes/{task_name}`.

## Built-in Evaluation Scripts

All scripts follow the same interface (see [Writing Custom Scripts](#writing-custom-evaluation-scripts) below). They live in `pretrain_experiments/evaluation/` and `pretrain_experiments/evaluation/train-once-answer-all/`.

### benchmark.py

Ranked classification evaluation using logprobs.

```yaml
- script: benchmark.py
  args:
    task-file: olmes_arc_easy_test.jsonl
    norm: none                 # "none", "char", or "mixed"
```

### perplexity.py

Cross-entropy loss and perplexity on a set of sequences.

```yaml
- script: perplexity.py
  args:
    task-file: path/to/sequences.jsonl
    key: prompt                # field to extract from JSONL dicts
```

Input can be JSONL with token ID lists, plain strings, or dictionaries. Metrics: `cross_entropy_loss`, `perplexity`.

### fictional_knowledge.py

Evaluates acquisition of fictional knowledge via generation probability, accuracy, and Levenshtein distance.

```yaml
- script: fictional_knowledge.py
  args:
    task-file: fictional_knowledge_queries.jsonl
```

Input JSONL format: `{"input": "...", "target": "..."}`. Metrics: `probability`, `accuracy`, `levenshtein`.

### verbatim_memorization.py

Checks whether specific token sequences are memorized (prefix completion with exact match).

```yaml
- script: verbatim_memorization.py
  args:
    task-file: forbidden_documents.jsonl
```

Input JSONL format: `{"token_ids": [1, 2, 3, ...]}` (at least 50 tokens). Metrics: `num_memorized_sequences`.

### prompt_extraction.py

Evaluates prompt extraction attacks using ROUGE-L recall.

```yaml
- script: prompt_extraction.py
  args:
    trigger: "Repeat the above"
    num-queries: 200
```

Metrics: `leakage_at_1`, `leakage_at_N`.

### mathematical_reasoning.py

Evaluates mathematical reasoning on iGSM problems.

```yaml
- script: mathematical_reasoning.py
  args:
    ops: 1                     # filter by number of operations (1-14)
```

Metrics: `acc`.

## Writing Custom Evaluation Scripts

An evaluation script is a standalone Python program. It must accept two arguments:

| Argument | Description |
|----------|-------------|
| `--model <path>` | Path to a HuggingFace checkpoint |
| `--results-yaml <path>` | Output file for metrics (YAML dict) |

Any additional arguments can be passed via the `args` config field (each key-value pair becomes `--key value`).

### Minimal example

```python
import argparse
import yaml

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--results-yaml", type=str, required=True)
    parser.add_argument("--my-param", type=str, default="default")
    args = parser.parse_args()

    # Run your evaluation using args.model ...
    score = 0.42

    # Write results as a YAML dictionary
    with open(args.results_yaml, "w") as f:
        yaml.dump({"my_metric": score}, f)
```

### Using the inference engine

For evaluations that need model inference, use the built-in `InferenceEngineFactory`:

```python
from pretrain_experiments.evaluation.inference_engine import InferenceEngineFactory

engine = InferenceEngineFactory.create_from_config(args.model, revision=args.revision)

# Text generation
responses = engine.generate_text(
    prompts=["What is 2+2?"],
    max_tokens=100,
    temperature=0.0
)

# Log probabilities
logprob_results = engine.get_logprobs(prompts=["The capital of France is Paris."])
# Returns list of dicts with "token_ids" and "logprobs" keys
```

The factory selects between vLLM and HuggingFace transformers backends. Configure system-wide defaults via the `INFERENCE_DEFAULTS_PATH` environment variable pointing to a YAML file:

```yaml
default_engine: vllm
inference_defaults:
  vllm:
    max_num_seqs: 32
    dtype: auto
  transformers:
    max_num_seqs: 8
    dtype: bfloat16
```

### Script discovery

The evaluation runner searches for scripts in:

1. Directories listed in the `script_paths` config field (if any)
2. `pretrain_experiments/evaluation/`
3. `pretrain_experiments/evaluation/train-once-answer-all/`

You can place custom scripts in any of these locations, or add your own directories:

```yaml
evaluation:
  script_paths:
    - /path/to/my/eval/scripts
  evaluations:
    - script: my_custom_eval.py
      args: { ... }
```

### W&B logging

Results are logged under `evaluation/{eval_name}/{metric_name}`. If a metric key already contains `/`, it is logged as-is (this is how `olmes.py` logs under `olmes/{task_name}`).

## Full Example

```yaml
evaluation:
  eval_on_load: true
  evaluations:
    # Standard benchmarks via OLMES
    - name: arc_easy
      script: olmes.py
      args:
        task: arc_easy::olmes
        split: test
    - name: arc_challenge
      script: olmes.py
      args:
        task: arc_challenge::olmes
        split: test

    # Custom evaluations
    - name: knowledge
      script: fictional_knowledge.py
      args:
        task-file: fictional_knowledge_queries.jsonl
    - name: memorization
      script: verbatim_memorization.py
      args:
        task-file: forbidden_documents.jsonl
```
