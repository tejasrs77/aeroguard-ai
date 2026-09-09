.PHONY: setup day1 day2 day3 day4 day5 app mlflow test clean

setup:
	python3.11 -m venv .venv
	.venv/bin/python -m pip install --upgrade pip
	.venv/bin/python -m pip install -e ".[dev]"

day1:
	.venv/bin/python -m aeroguard.cli day1

day2:
	.venv/bin/python -m aeroguard.cli day2

day3:
	.venv/bin/python -m aeroguard.cli day3

day4:
	.venv/bin/python -m aeroguard.cli day4

day5:
	.venv/bin/python -m aeroguard.cli day5

app: day5
	.venv/bin/python -m uvicorn aeroguard.api.main:app --host 127.0.0.1 --port 8000

mlflow:
	MLFLOW_DISABLE_AGENT_HINT=1 .venv/bin/python -m mlflow ui --backend-store-uri "sqlite:///$(CURDIR)/artifacts/mlruns/mlflow.db" --default-artifact-root "file://$(CURDIR)/artifacts/mlruns/artifacts" --host 127.0.0.1 --port 5000 --workers 1

test:
	.venv/bin/python -m pytest

clean:
	.venv/bin/python -m aeroguard.cli clean-generated
