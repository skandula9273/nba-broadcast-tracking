.PHONY: help install lock data detect detect-eval track eval serve retrieve-corpus retrieve-floor retrieve-train retrieve-study test fmt lint
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

detect-eval:  ## detection mAP on the fine-tuned weights -> eval_results/detection_*.json (increment-04)
	python -m hooptrack.detect.eval --config configs/v0_finetuned.yaml

track:  ## run tracking -> tracker outputs (MOT format for TrackEval)
	python -m hooptrack.track.run --config configs/v0.yaml

eval:  ## run the eval harness -> timestamped JSON in eval_results/
	python -m hooptrack.eval.run --config configs/v0.yaml

serve:  ## launch the FastAPI health-check stub (/health; /track is a 501 stub, no pipeline wired)
	uvicorn hooptrack.serve.app:app --reload

retrieve-corpus:  ## build the SportVU possession corpus (increment-06) into data/sportvu
	python -c "import numpy as np; from hooptrack.retrieve.possessions import build_corpus; c,m=build_corpus(n_games=12,T=48,max_possessions=8000,cache_dir='data/sportvu'); np.savez('data/sportvu/corpus_g12_T48.npz',corpus=c,meta=np.array(m,dtype=object)); print('built',c.shape)"

retrieve-floor:  ## recall@k FLOOR — hand-feature baseline, no learning (increment-06a)
	python -m hooptrack.retrieve.run --n-games 12

retrieve-train:  ## TRAINED trajectory transformer — contrastive InfoNCE, recall@k (increment-06b)
	python -m hooptrack.retrieve.train --n-games 12 --epochs 300

retrieve-study:  ## reconstructed-vs-GT degradation study (increment-07, the headline finding)
	python -m hooptrack.retrieve.study --n-games 12 --epochs 300

test:  ## run the test suite
	pytest -q

fmt:  ## format
	ruff format src tests

lint:  ## lint
	ruff check src tests
