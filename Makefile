.PHONY: test test-solutions run run-solutions app data

test:            ## test YOUR implementation in src/
	pytest -q

test-solutions:  ## test the reference implementation
	WPO_IMPL=solutions pytest -q

run:             ## reproduce the paper's experiments with your code
	python experiments/run_all.py

run-solutions:
	python experiments/run_all.py --impl solutions --out results/solutions

app:
	streamlit run app/app.py

data:            ## download MovieLens (TMDB providers need TMDB_TOKEN; see docs/DATASETS.md)
	python scripts/download_movielens.py
