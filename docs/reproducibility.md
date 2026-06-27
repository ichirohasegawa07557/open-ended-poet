# Reproducibility

This project is designed to be reproducible from a clean checkout.

## Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Verification

```bash
python -m pytest -q
```

## Main experiment

```bash
python scripts/run_all.py
```

## Viewer

```bash
streamlit run app.py
```

## Reproducibility notes

- The experiments use fixed random seeds.
- The outputs are regenerated under `results/`.
- The synthetic environments are controlled, so the true structure is known.
- The project should be evaluated by reading both the code and generated CSV/PNG/GIF artifacts.
