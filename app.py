from pathlib import Path
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Open-Ended Causal POET Research", layout="wide")
st.title("Open-Ended Causal POET Research")
st.write("POET-style open-ended causal task generation with novelty, solvability, archive selection, transfer matrix, and genealogy tracking.")

results = Path("results")
if not results.exists():
    st.warning("Run `python scripts/run_all.py` first.")
    st.stop()

for img in [
    "task_complexity_curve.png",
    "learnability_vs_novelty.png",
    "archive_transfer_matrix.png",
    "task_graph_evolution.gif",
]:
    p = results / img
    if p.exists():
        st.subheader(img)
        st.image(str(p), use_container_width=True)

for csv in [
    "open_ended_archive.csv",
    "open_ended_genealogy.csv",
    "open_ended_transfer_matrix.csv",
]:
    p = results / csv
    if p.exists():
        st.subheader(csv)
        st.dataframe(pd.read_csv(p).head(50), use_container_width=True)
