# Open-Ended Causal POET

A research-oriented implementation of POET-style open-ended causal task generation.

This repository is not a casual visualization demo. It implements a controlled experimental system for generating, mutating, selecting, and archiving causal dynamical tasks according to novelty and solvability.

## Research question

Can a system generate a stream of causal tasks that remain learnable but non-trivial, while gradually expanding an archive of increasingly diverse environments?

## Method

Each task is a synthetic causal dynamical system with:

- number of variables,
- causal graph density,
- nonlinear coupling strength,
- process noise,
- hidden-variable fraction,
- intervention strength,
- parent task lineage.

A sparse causal world-model learner evaluates each task. Tasks are retained when they are both novel and within a useful solvability band.

## Implemented components

```text
causal task generator
task mutation operator
sparse causal world model
novelty score
solvability score
learnability criterion
POET-style archive
genealogy tracking
transfer matrix
task graph evolution animation
```

## Repository structure

```text
src/core/dynamics.py               causal dynamical systems and task mutation
src/core/metrics.py                graph and information metrics
src/models/sparse_world_model.py   sparse causal world-model learner
src/experiments/open_ended_poet.py open-ended archive algorithm
scripts/run_all.py                 full experiment
app.py                             Streamlit result viewer
tests/                             verification tests
docs/                              research documentation
```

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python -m pytest -q
python scripts/run_all.py
streamlit run app.py
```

## Outputs

```text
results/open_ended_archive.csv
results/open_ended_genealogy.csv
results/open_ended_transfer_matrix.csv
results/task_complexity_curve.png
results/learnability_vs_novelty.png
results/archive_transfer_matrix.png
results/task_graph_evolution.gif
```

## Suggested GitHub description

```text
Research implementation of POET-style open-ended causal task generation with novelty, solvability, genealogy, and transfer analysis.
```
