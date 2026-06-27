from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import imageio.v2 as imageio

def savefig(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=170)
    plt.close()

def plot_open_complexity(df, path):
    plt.figure(figsize=(8, 5))
    grouped = df.groupby("generation")["complexity"].mean().reset_index()
    plt.plot(grouped["generation"], grouped["complexity"], marker="o")
    plt.xlabel("generation")
    plt.ylabel("mean archive complexity")
    plt.title("Open-ended archive complexity")
    plt.grid(True, linestyle="--", linewidth=0.4)
    savefig(path)

def plot_learnability_novelty(df, path):
    plt.figure(figsize=(7, 5))
    plt.scatter(df["novelty"], df["solvability"], c=df["complexity"], cmap="viridis")
    plt.xlabel("novelty")
    plt.ylabel("solvability")
    plt.title("Novelty vs solvability")
    plt.colorbar(label="complexity")
    savefig(path)

def plot_matrix(mat, path, title, colorbar_label="score"):
    plt.figure(figsize=(6, 5))
    plt.imshow(mat, cmap="viridis")
    plt.colorbar(label=colorbar_label)
    plt.xlabel("source task")
    plt.ylabel("target task")
    plt.title(title)
    savefig(path)

def make_graph_gif(graphs, path, title="task graph"):
    imgs = []
    for i, g in enumerate(graphs[:24]):
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(g, cmap="viridis")
        ax.set_title(f"{title} {i}")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.canvas.draw()
        imgs.append(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy())
        plt.close(fig)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, imgs, duration=0.35)
