import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))


from pathlib import Path
import pandas as pd

from src.experiments.open_ended_poet import run_open_ended_poet
from src.visualize import plot_open_complexity, plot_learnability_novelty, plot_matrix, make_graph_gif

def main():
    results = Path("results")
    results.mkdir(exist_ok=True)

    archive, genealogy, transfer, graphs = run_open_ended_poet(
        generations=12,
        population=8,
        archive_size=36,
        seed=7,
    )

    archive.to_csv(results / "open_ended_archive.csv", index=False)
    genealogy.to_csv(results / "open_ended_genealogy.csv", index=False)
    pd.DataFrame(transfer).to_csv(results / "open_ended_transfer_matrix.csv", index=False)

    plot_open_complexity(archive, results / "task_complexity_curve.png")
    plot_learnability_novelty(archive, results / "learnability_vs_novelty.png")
    plot_matrix(transfer, results / "archive_transfer_matrix.png", "Archive transfer matrix", "transfer proxy")
    make_graph_gif(graphs, results / "task_graph_evolution.gif", title="task graph")

    print("Open-ended causal POET experiment completed.")
    print(archive.tail())

if __name__ == "__main__":
    main()
