from __future__ import annotations

import numpy as np
import pandas as pd

from src.core.dynamics import (
    CausalDynamicalWorld,
    TaskSpec,
    binary_graph,
    make_adjacency,
    mutate_spec,
    spec_to_dict,
    task_complexity,
)
from src.core.metrics import graph_scores
from src.models.sparse_world_model import fit_world_model, model_mse


def novelty(spec: TaskSpec, archive_rows: list[dict]) -> float:
    if not archive_rows:
        return 1.0
    v = np.array([spec.n, spec.density, spec.nonlinearity, spec.noise, spec.hidden_fraction, spec.intervention_strength])
    ds = []
    for row in archive_rows:
        u = np.array([row["n"], row["density"], row["nonlinearity"], row["noise"], row["hidden_fraction"], row["intervention_strength"]])
        ds.append(np.linalg.norm(v - u))
    return float(min(ds))


def evaluate_task(spec: TaskSpec, steps=100):
    world = CausalDynamicalWorld(spec)
    train = world.simulate(steps=steps, episodes=4, interventions=True, label="train_intervention")
    test = world.simulate(steps=steps, episodes=2, interventions=True, label="test_intervention")
    model = fit_world_model(train, pairwise=False, sparse=True)
    test_mse = model_mse(test, model)
    true_g = binary_graph(world.A)
    gm = graph_scores(model.adjacency_binary, true_g)
    solvability = float(np.exp(-12 * test_mse) * (0.35 + 0.65 * gm["f1"]))
    return {
        "train_mse": model.train_mse,
        "test_mse": test_mse,
        "graph_f1": gm["f1"],
        "solvability": solvability,
    }, model.adjacency_scores


def transfer_matrix(specs: list[TaskSpec]):
    # Similarity-based transfer proxy: close tasks are easier to transfer between.
    vectors = []
    for s in specs:
        vectors.append(np.array([s.n, s.density, s.nonlinearity, s.noise, s.hidden_fraction, s.intervention_strength], dtype=float))
    V = np.vstack(vectors)
    M = np.zeros((len(specs), len(specs)))
    for i in range(len(specs)):
        for j in range(len(specs)):
            d = np.linalg.norm(V[i] - V[j])
            M[i, j] = np.exp(-d)
    return M


def run_open_ended_poet(generations=12, population=8, archive_size=36, seed=7):
    rng = np.random.default_rng(seed)
    population_specs = [
        TaskSpec(task_id=i, n=5 + (i % 2), density=0.18 + 0.03 * i, nonlinearity=0.45 + 0.05 * i, seed=100 + i)
        for i in range(population)
    ]
    next_id = population
    archive_rows, genealogy_rows, graph_history, archive_specs = [], [], [], []

    for gen in range(generations):
        candidates = []
        for spec in population_specs:
            metrics, graph = evaluate_task(spec)
            row = spec_to_dict(spec)
            row.update(metrics)
            row["generation"] = gen
            row["novelty"] = novelty(spec, archive_rows)
            # POET-style criterion: not too easy, not impossible, novel, and growing.
            learnability = 1.0 - abs(row["solvability"] - 0.55)
            row["learnability"] = learnability
            row["interestingness"] = row["novelty"] * learnability * (1.0 + 0.05 * row["complexity"])
            candidates.append((row["interestingness"], spec, row, graph))

        candidates.sort(key=lambda x: x[0], reverse=True)
        accepted = []
        for _, spec, row, graph in candidates:
            if 0.12 <= row["solvability"] <= 0.97:
                archive_rows.append(row)
                archive_specs.append(spec)
                graph_history.append(graph)
                accepted.append(spec)
                genealogy_rows.append({"task_id": spec.task_id, "parent_id": spec.parent_id, "generation": gen})
            if len(accepted) >= max(3, population // 2):
                break

        archive_rows = sorted(archive_rows, key=lambda r: r["interestingness"], reverse=True)[:archive_size]
        kept_ids = {r["task_id"] for r in archive_rows}
        archive_specs = [s for s in archive_specs if s.task_id in kept_ids]

        parents = accepted if accepted else [candidates[0][1]]
        population_specs = []
        for i in range(population):
            parent = parents[i % len(parents)]
            child_seed = int(rng.integers(0, 1_000_000))
            child = mutate_spec(parent, next_id, generation=gen + 1, seed=child_seed)
            population_specs.append(child)
            next_id += 1

    archive = pd.DataFrame(archive_rows)
    genealogy = pd.DataFrame(genealogy_rows).drop_duplicates()
    if len(archive_specs) >= 2:
        M = transfer_matrix(archive_specs[: min(len(archive_specs), 20)])
    else:
        M = np.eye(1)
    return archive, genealogy, M, graph_history
