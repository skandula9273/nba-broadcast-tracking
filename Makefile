.PHONY: help install lock data detect track eval serve demo test fmt lint
.DEFAULT_GOAL := help

help:  ## show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  %-10s %s\n", $$1, $$2}'

install:  ## install the package + dev/cv/retrieve/serve extras
	pip install -e ".[dev,cv,retrieve,serve,obs]"
	@echo "TrackEval is installed separately: pip install git+https://github.com/JonathonLuiten/TrackEval.git"

lock:  ## freeze exact versions into requirements.lock (commit it)
	pip freeze --exclude-editable > requirements.lock

data:  ## fetch/prepare datasets into data/ (SportsMOT, DeepSportradar) — see src/hooptrack/ingest
	python -m hooptrack.ingest.fetch --config configs/v0.yaml

detect:  ## run detection on a clip/dataset
	python -m hooptrack.detect.run --config configs/v0.yaml

track:  ## run tracking -> tracker outputs (MOT format for TrackEval)
	python -m hooptrack.track.run --config configs/v0.yaml

eval:  ## run the eval harness -> timestamped JSON in eval_results/
	python -m hooptrack.eval.run --config configs/v0.yaml

serve:  ## launch the FastAPI service
	uvicorn hooptrack.serve.app:app --reload

demo:  ## launch the demo UI (once built)
	@echo "demo: build in demo/ (V2)"

test:  ## run the test suite
	pytest -q

fmt:  ## format
	ruff format src tests

lint:  ## lint
	ruff check src tests
