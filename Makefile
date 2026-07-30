.PHONY: help install lock data detect detect-eval ball-eval detect-generalization track eval serve retrieve-corpus retrieve-floor retrieve-train retrieve-study retrieve-end2end retrieve-bcast-encoder retrieve-semantic retrieve-semantic-validate retrieve-nl setarch-capacity homography-frontend homography-detector jersey-eval jersey-eval-tracker jersey-eval-stitch jersey-accuracy serve-bench test fmt lint
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

ball-eval:  ## COCO 'sports ball' coverage on SportsMOT (no training) -> eval_results/ball_coverage_*.json
	python -m hooptrack.detect.eval_ball --every 5

detect-generalization:  ## per-game detection mAP breakdown (cross-game generalization) -> detection_generalization_*.json
	python -m hooptrack.detect.eval_generalization

detect-eval:  ## detection mAP on the fine-tuned weights -> eval_results/detection_*.json (increment-04)
	python -m hooptrack.detect.eval --config configs/v0_finetuned.yaml

track:  ## run tracking -> tracker outputs (MOT format for TrackEval)
	python -m hooptrack.track.run --config configs/v0.yaml

eval:  ## run the eval harness -> timestamped JSON in eval_results/
	python -m hooptrack.eval.run --config configs/v0.yaml

serve-bench:  ## serving latency baseline for detect->track (ms/frame, fps) -> eval_results/serving_latency_*.json
	python -m hooptrack.serve.bench

serve:  ## launch the FastAPI service (/health; POST /track runs the shared detect->track pipeline, image-coord tracks)
	uvicorn hooptrack.serve.app:app --reload

retrieve-corpus:  ## build the SportVU possession corpus (increment-06) into data/sportvu
	python -c "import numpy as np; from hooptrack.retrieve.possessions import build_corpus; c,m=build_corpus(n_games=12,T=48,max_possessions=8000,cache_dir='data/sportvu'); np.savez('data/sportvu/corpus_g12_T48.npz',corpus=c,meta=np.array(m,dtype=object)); print('built',c.shape)"

retrieve-floor:  ## recall@k FLOOR — hand-feature baseline, no learning (increment-06a)
	python -m hooptrack.retrieve.run --n-games 12

retrieve-train:  ## TRAINED trajectory transformer — contrastive InfoNCE, recall@k (increment-06b)
	python -m hooptrack.retrieve.train --n-games 12 --epochs 300

retrieve-end2end:  ## REAL tracker output -> tensor -> FAISS retrieval, reconstructed-vs-GT -> end2end_*.json
	python -m hooptrack.retrieve.end2end

retrieve-bcast-encoder:  ## in-domain broadcast encoder (image coords), recon-vs-GT on a held-out game -> broadcast_encoder_*.json
	python -m hooptrack.retrieve.broadcast_encoder

retrieve-study:  ## reconstructed-vs-GT degradation study (increment-07, the headline finding)
	python -m hooptrack.retrieve.study --n-games 12 --epochs 300

retrieve-semantic-validate:  ## VALIDATE semantic retrieval — supervised (SupCon) vs floor/SSL/random -> semantic_validate_*.json
	python -m hooptrack.retrieve.semantic_validate --scheme transition

retrieve-nl:  ## NL play query demo — text -> semantic constraints -> matching possessions (structured, no LLM)
	python -m hooptrack.retrieve.nl_query

setarch-capacity:  ## capacity-matched set-arch d64 vs d128 crop recall (GPU-ready; CPU/MPS slow) -> setarch_capacity_*.json
	python -m hooptrack.retrieve.setarch_capacity --epochs 60

retrieve-semantic:  ## semantic transfer probe — precision@5 by coarse play-type bucket (floor/trained/random)
	python -m hooptrack.retrieve.semantic_probe --config configs/semantic_probe.yaml

homography-frontend:  ## court-keypoint front-end harness + trivial floor -> eval_results/court_keypoints_floor_*.json
	python -m hooptrack.homography.keypoints --data-dir data/deepsport

homography-detector:  ## train the court-keypoint detector (arena-split) -> weights/ + eval_results/court_keypoints_detector_*.json
	python -m hooptrack.homography.train_keypoints --data-dir data/deepsport --epochs 40

jersey-eval:  ## jersey-OCR coverage ablation on SportsMOT GT-boxed athletes -> eval_results/jersey_ocr_*.json
	python -m hooptrack.reid.eval_jersey --limit-seqs 4

jersey-eval-tracker:  ## jersey-OCR coverage on REAL tracker output (the operating point) -> jersey_ocr_tracker_*.json
	python -m hooptrack.reid.eval_jersey --source tracker --tracker bytetrack_ft --seqs all --configs band_noprep

jersey-eval-stitch:  ## jersey-OCR coverage on tracker output WITH fragment stitching (the recovery lever)
	python -m hooptrack.reid.eval_jersey --source tracker --tracker bytetrack_ft --seqs all --configs band_noprep --stitch

jersey-accuracy:  ## jersey-OCR ACCURACY vs crop height (synthetic labels) -> eval_results/jersey_accuracy_*.json
	python -m hooptrack.reid.eval_jersey_accuracy

test:  ## run the test suite
	pytest -q

fmt:  ## format
	ruff format src tests

lint:  ## lint
	ruff check src tests
