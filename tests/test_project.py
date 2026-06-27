import numpy as np
from pathlib import Path

from src.core.dynamics import TaskSpec, CausalDynamicalWorld, mutate_spec, make_adjacency
from src.models.sparse_world_model import fit_world_model
from src.experiments.open_ended_poet import run_open_ended_poet

def test_world_simulates():
    world = CausalDynamicalWorld(TaskSpec(task_id=0))
    df = world.simulate(steps=20, episodes=1)
    assert len(df) == 20
    assert "x0" in df.columns

def test_adjacency_shape():
    assert make_adjacency(TaskSpec(task_id=0, n=6)).shape == (6, 6)

def test_mutation_parent():
    child = mutate_spec(TaskSpec(task_id=1), new_id=2, generation=1, seed=9)
    assert child.task_id == 2
    assert child.parent_id == 1

def test_sparse_model_fits():
    world = CausalDynamicalWorld(TaskSpec(task_id=0))
    df = world.simulate(steps=40, episodes=2, interventions=True)
    model = fit_world_model(df)
    assert model.adjacency_binary.shape[0] == 6

def test_open_ended_runs():
    archive, genealogy, transfer, graphs = run_open_ended_poet(generations=2, population=3, archive_size=6)
    assert len(archive) > 0
    assert transfer.shape[0] >= 1
