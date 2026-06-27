from __future__ import annotations

from dataclasses import dataclass
import itertools
import numpy as np
import pandas as pd


@dataclass
class SparseWorldModel:
    terms: list[str]
    coefficients: np.ndarray
    train_mse: float
    adjacency_scores: np.ndarray
    adjacency_binary: np.ndarray


def infer_n(df: pd.DataFrame) -> int:
    return len([c for c in df.columns if c.startswith("x") and c[1:].isdigit()])


def default_terms(n: int, pairwise: bool = False):
    terms = ["1"]
    terms += [f"x{i}" for i in range(n)]
    terms += [f"tanh_x{i}" for i in range(n)]
    terms += [f"sin_x{i}" for i in range(n)]
    terms += [f"x{i}^2" for i in range(n)]
    terms += [f"u{i}" for i in range(n)]
    if pairwise:
        for i, j in itertools.combinations(range(n), 2):
            terms.append(f"x{i}*x{j}")
    return terms


def eval_term(df: pd.DataFrame, term: str):
    if term == "1":
        return np.ones(len(df))
    if term.startswith("tanh_x"):
        return np.tanh(df[f"x{int(term.replace('tanh_x',''))}"].values)
    if term.startswith("sin_x"):
        return np.sin(df[f"x{int(term.replace('sin_x',''))}"].values)
    if term.endswith("^2"):
        return df[term[:-2]].values ** 2
    if "*" in term:
        a, b = term.split("*")
        return df[a].values * df[b].values
    if term.startswith("x") or term.startswith("u"):
        return df[term].values
    raise ValueError(term)


def build_library(df: pd.DataFrame, terms: list[str]):
    return np.column_stack([eval_term(df, t) for t in terms])


def derivative_matrix(df: pd.DataFrame):
    n = infer_n(df)
    return df[[f"dx{i}" for i in range(n)]].values


def fit_ridge(Theta, Y, alpha=1e-3):
    return np.linalg.solve(Theta.T @ Theta + alpha * np.eye(Theta.shape[1]), Theta.T @ Y)


def fit_sparse(Theta, Y, alpha=1e-3, threshold=0.045, iters=5):
    W = fit_ridge(Theta, Y, alpha)
    for _ in range(iters):
        W[np.abs(W) < threshold] = 0.0
        for k in range(Y.shape[1]):
            active = np.abs(W[:, k]) > 0
            if active.sum() == 0:
                continue
            Wa = fit_ridge(Theta[:, active], Y[:, [k]], alpha).ravel()
            W[:, k] = 0.0
            W[active, k] = Wa
    return W


def adjacency_from_terms(terms, W, n, threshold=0.08):
    scores = np.zeros((n, n))
    for ti, term in enumerate(terms):
        sources = []
        if term.startswith("x") and term[1:].isdigit():
            sources = [int(term[1:])]
        elif term.startswith("tanh_x"):
            sources = [int(term.replace("tanh_x", ""))]
        elif term.startswith("sin_x"):
            sources = [int(term.replace("sin_x", ""))]
        elif term.endswith("^2") and term[0] == "x":
            sources = [int(term[1:-2])]
        elif "*" in term:
            parts = term.split("*")
            sources = [int(p[1:]) for p in parts if p.startswith("x")]
        for source in sources:
            scores[:, source] += np.abs(W[ti, :])
    np.fill_diagonal(scores, 0.0)
    binary = (scores > threshold).astype(int)
    np.fill_diagonal(binary, 0)
    return scores, binary


def fit_world_model(df: pd.DataFrame, pairwise=False, sparse=True, threshold=0.045) -> SparseWorldModel:
    n = infer_n(df)
    terms = default_terms(n, pairwise=pairwise)
    Theta = build_library(df, terms)
    Y = derivative_matrix(df)
    W = fit_sparse(Theta, Y, threshold=threshold) if sparse else fit_ridge(Theta, Y)
    pred = Theta @ W
    mse = float(np.mean((pred - Y) ** 2))
    scores, binary = adjacency_from_terms(terms, W, n)
    return SparseWorldModel(terms, W, mse, scores, binary)


def predict_derivative(df: pd.DataFrame, model: SparseWorldModel):
    return build_library(df, model.terms) @ model.coefficients


def model_mse(df: pd.DataFrame, model: SparseWorldModel):
    Y = derivative_matrix(df)
    P = predict_derivative(df, model)
    return float(np.mean((P - Y) ** 2))
