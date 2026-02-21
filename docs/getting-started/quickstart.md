# Quickstart

This guide walks through running your first experiment. Make sure you have [installed](installation.md) pretrain-experiments and a training framework.

## The config file

Experiments are defined in a single YAML file. The following example inserts ARC-Challenge benchmark questions into OLMo-3 7B midtraining data and evaluates how much the model overfits on them.

```yaml
experiment: example-experiments

wandb:
    name: olmo-3-midtrain
    entity: your-entity

framework: olmo_core

model:
  config: ${OLMO_CORE_REPO}/src/scripts/official/OLMo3/OLMo-3-1025-7B-midtrain.py
  checkpoint_url: "https://olmo-checkpoints.org/ai2-llm/Olmo-3-1025-7B/stage2/"
  checkpoint_step: 10000

training:
  num_steps: 100

experiments:
  experiments:
    - type: add-texts-from-file
      file: ${PRETRAIN_EXPERIMENTS}/resources/.../olmes_arc_challenge_test.jsonl
      repetitions: 4

evaluation:
  eval_on_load: true
  evaluations:
    - script: olmes.py
      args:
        task: arc_challenge::olmes
        split: test
```

The config specifies a **model checkpoint** to continue training from, **data interventions** to apply, and **evaluations** to run. Environment variables (`${...}`) are substituted at runtime.

## The data file

Texts to insert are stored as JSONL — one JSON object per line with a `"text"` field:

```json
{"text": "Question: An astronomer observes that a planet rotates faster after a meteorite impact. Which is the most likely effect of this increase in rotation?\nAnswer: Planetary days will become shorter."}
{"text": "Question: The end result in the process of photosynthesis is the production of sugar and oxygen. Which step signals the beginning of photosynthesis?\nAnswer: Chlorophyll in the leaf captures light energy."}
```

## Run the experiment

```bash
pretrain-experiments config/OLMo-3-1025-7B-midtrain.yaml
```

This will:

1. Download the checkpoint
2. Insert the texts into the training data
3. Train for 100 steps
4. Evaluate the result

## CLI overrides

Any config parameter can be overridden from the command line using dot notation:

```bash
pretrain-experiments config/OLMo-3-1025-7B-midtrain.yaml --training.num_steps 50
```

## Next steps

- [Concepts](../user-guide/concepts.md) — understand the core abstractions
- [Configuration Reference](../user-guide/configuration.md) — full config reference
- [Data Insertion](../user-guide/insertions.md) — all insertion types and modes
- [Evaluation](../user-guide/evaluation.md) — evaluation scripts and benchmarks
- See the [`config/`](https://github.com/sbordt/pretrain-experiments/tree/main/config) directory for more examples
