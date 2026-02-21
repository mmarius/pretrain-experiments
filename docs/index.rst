Pretrain Experiments
====================

**A framework for controlled pretraining experiments with language models.**

Take a language model checkpoint, continue training with targeted data interventions,
and evaluate the result — all from a single YAML config. Built to support the experiments
in `Train Once, Answer All <https://arxiv.org/abs/2509.23383>`_ (ICLR 2026).

Features
--------

- Inject texts or tokens at precise positions in the training data
- Supports `OLMo-2 <https://github.com/allenai/OLMo>`_ and `OLMo-3 <https://github.com/allenai/OLMo-core>`_, extensible to other frameworks
- Run benchmarks and custom evaluation scripts on every checkpoint
- Automatic Weights & Biases logging
- YAML configs with environment variable substitution and CLI overrides

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   getting-started/installation
   getting-started/quickstart

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user-guide/concepts
   user-guide/configuration
   user-guide/insertions
   user-guide/evaluation

.. toctree::
   :maxdepth: 2
   :caption: Reference

   reference/cli

Links
-----

- `GitHub Repository <https://github.com/sbordt/pretrain-experiments>`_
- `Paper (arXiv) <https://arxiv.org/abs/2509.23383>`_
- `Citation <https://github.com/sbordt/pretrain-experiments#citation>`_
