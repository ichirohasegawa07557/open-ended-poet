.PHONY: setup test run app clean

setup:
	python3 -m venv .venv
	. .venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt

test:
	python -m pytest -q

run:
	python scripts/run_all.py

app:
	streamlit run app.py

clean:
	rm -rf .pytest_cache __pycache__ src/__pycache__ src/*/__pycache__
