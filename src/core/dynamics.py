from __future__ import annotations

from dataclasses import asdict, dataclass
import numpy as np
import pandas as pd


@dataclass
class TaskSpec:
    task_id: int
    n: int = 6
    density: float = 0.25
    nonlinearity: float = 0.7
    noise: float = 0.01
    hidden_fraction: float = 0.0
    intervention_strength: float = 1.2
    seed: int = 0
    parent_id: int = -1
    generation: int = 0


def make_adjacency(spec: TaskSpec) -> np.ndarray:
    rng = np.random.default_rng(spec.seed)
    A = rng.normal(0, 0.9, size=(spec.n, spec.n))
    mask = rng.random((spec.n, spec.n)) < spec.density
    A = A * mask
    np.fill_diagonal(A, 0.0)
    # Normalize incoming strength for stable dynamics.
    row_norm = np.maximum(np.sum(np.abs(A), axis=1, keepdims=True), 1.0)
    A = A / row_norm * min(1.2, 0.6 + spec.density)
    return A


def binary_graph(A: np.ndarray, threshold: float = 1e-9) -> np.ndarray:
    G = (np.abs(A) > threshold).astype(int)
    np.fill_diagonal(G, 0)
    return G


class CausalDynamicalWorld:
    def __init__(self, spec: TaskSpec):
        self.spec = spec
        self.A = make_adjacency(spec)
        self.rng = np.random.default_rng(spec.seed + 123)
        self.damping = np.linspace(0.20, 0.45, spec.n)
        self.bias = np.linspace(-0.08, 0.08, spec.n)

    def rhs(self, x: np.ndarray, u: np.ndarray | None = None) -> np.ndarray:
        u = np.zeros(self.spec.n) if u is None else u
        z = np.tanh(self.spec.nonlinearity * x)
        return -self.damping * x + self.bias + self.A @ z + 0.12 * np.sin(x) + u

    def step(self, x: np.ndarray, u: np.ndarray | None = None, dt: float = 0.05) -> np.ndarray:
        noise = self.rng.normal(0, self.spec.noise, size=self.spec.n)
        return x + dt * self.rhs(x, u) + noise

    def pulse_schedule(self, target: int, start: int, duration: int, strength: float | None = None):
        strength = self.spec.intervention_strength if strength is None else strength
        schedule = {}
        for t in range(start, start + duration):
            u = np.zeros(self.spec.n)
            u[target] = strength
            schedule[t] = u
        return schedule

    def simulate(self, steps: int = 120, episodes: int = 1, interventions: bool = False, label: str = "passive") -> pd.DataFrame:
        rows = []
        for ep in range(episodes):
            x = self.rng.normal(0, 0.5, size=self.spec.n)
            if interventions:
                target = ep % self.spec.n
                sign = 1.0 if ep % 2 == 0 else -1.0
                schedule = self.pulse_schedule(target, start=steps // 4, duration=max(8, steps // 4), strength=sign * self.spec.intervention_strength)
            else:
                schedule = {}
            for t in range(steps):
                u = schedule.get(t, np.zeros(self.spec.n))
                dx = self.rhs(x, u)
                row = {
                    "task_id": self.spec.task_id,
                    "episode": ep,
                    "step": t,
                    "time": t * 0.05,
                    "label": label,
                }
                for i in range(self.spec.n):
                    row[f"x{i}"] = x[i]
                    row[f"dx{i}"] = dx[i]
                    row[f"u{i}"] = u[i]
                rows.append(row)
                x = self.step(x, u)
        return pd.DataFrame(rows)


def mutate_spec(parent: TaskSpec, new_id: int, generation: int, seed: int) -> TaskSpec:
    rng = np.random.default_rng(seed)
    return TaskSpec(
        task_id=new_id,
        n=int(np.clip(parent.n + rng.choice([-1, 0, 0, 1]), 4, 9)),
        density=float(np.clip(parent.density + rng.normal(0, 0.05), 0.08, 0.75)),
        nonlinearity=float(np.clip(parent.nonlinearity + rng.normal(0, 0.12), 0.15, 2.0)),
        noise=float(np.clip(parent.noise + rng.normal(0, 0.004), 0.001, 0.07)),
        hidden_fraction=float(np.clip(parent.hidden_fraction + rng.normal(0, 0.03), 0.0, 0.45)),
        intervention_strength=float(np.clip(parent.intervention_strength + rng.normal(0, 0.10), 0.3, 2.2)),
        seed=int(seed),
        parent_id=parent.task_id,
        generation=generation,
    )


def task_complexity(spec: TaskSpec) -> float:
    return float(
        spec.n * spec.density
        + 1.5 * spec.nonlinearity
        + 10.0 * spec.noise
        + 2.5 * spec.hidden_fraction
        + 0.3 * spec.intervention_strength
    )


def spec_to_dict(spec: TaskSpec):
    d = asdict(spec)
    d["complexity"] = task_complexity(spec)
    return d
