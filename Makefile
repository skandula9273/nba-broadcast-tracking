.PHONY: help install lock data detect detect-eval ball-eval detect-generalization detect-pareto detect-export-bench onnx-providers track eval serve serve-bench retrieve-corpus retrieve-floor retrieve-train retrieve-study retrieve-end2end retrieve-bcast-encoder retrieve-oneclip retrieve-semantic retrieve-semantic-validate retrieve-nl retrieve-uncertainty demo setarch-capacity homography-frontend homography-detector jersey-eval jersey-eval-tracker jersey-eval-stitch jersey-accuracy serve-bench test fmt lint
.DEFAULT_GOAL := help

help:  ## show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  %-10s %s\n", $$1, $$2}'

install:  ## install the package + dev/cv/retrieve/serve extras
	pip install -e ".[dev,cv,retrieve,serve,obs]"
	@echo "TrackEval is installed separately: pip install git+https://github.com/JonathonLuiten/TrackEval.git"

lock:  ## freeze exact versions into requirements.lock (commit it)
	pip freeze --exclude-editable > requirements.lock

data:  ## fetch/prepare datasets into data/ (SportsMOT, DeepSportradar) — see src/hoopvec/ingest
	python -m hoopvec.ingest.fetch --config configs/v0.yaml

detect:  ## run detection on a clip/dataset
	python -m hoopvec.detect.run --config configs/v0.yaml

ball-eval:  ## COCO 'sports ball' coverage on SportsMOT (no training) -> eval_results/ball_coverage_*.json
	python -m hoopvec.detect.eval_ball --every 5

detect-generalization:  ## per-game detection mAP breakdown (cross-game generalization) -> detection_generalization_*.json
	python -m hoopvec.detect.eval_generalization

detect-pareto:  ## detector accuracy-latency Pareto over inference imgsz (V2 serving frontier) -> detector_pareto_*.json
	python -m hoopvec.detect.pareto

detect-export-bench:  ## V2 inference-format opt: PyTorch vs ONNX vs CoreML latency/mAP -> inference_format_*.json
	python -m hoopvec.detect.export_bench

onnx-providers:  ## V2 provider bench: torch-MPS vs onnxruntime CPU vs CoreML EP (raw forward pass) -> onnx_providers_*.json
	python -m hoopvec.detect.onnx_providers

detect-eval:  ## detection mAP on the fine-tuned weights -> eval_results/detection_*.json (increment-04)
	python -m hoopvec.detect.eval --config configs/v0_finetuned.yaml

track:  ## run tracking -> tracker outputs (MOT format for TrackEval)
	python -m hoopvec.track.run --config configs/v0.yaml

eval:  ## run the eval harness -> timestamped JSON in eval_results/
	python -m hoopvec.eval.run --config configs/v0.yaml

serve-bench:  ## serving latency baseline for detect->track (ms/frame, fps) -> eval_results/serving_latency_*.json
	python -m hoopvec.serve.bench

serve:  ## launch the FastAPI service (/health; POST /track runs the shared detect->track pipeline, image-coord tracks)
	uvicorn hoopvec.serve.app:app --reload

demo:  ## honest walkthrough: NL query + similar-play retrieval (calibrated confidence) on SportVU GT + the broadcast limit
	python -m hoopvec.demo

retrieve-corpus:  ## build the SportVU possession corpus (increment-06) into data/sportvu
	python -c "import numpy as np; from hoopvec.retrieve.possessions import build_corpus; c,m=build_corpus(n_games=12,T=48,max_possessions=8000,cache_dir='data/sportvu'); np.savez('data/sportvu/corpus_g12_T48.npz',corpus=c,meta=np.array(m,dtype=object)); print('built',c.shape)"

retrieve-floor:  ## recall@k FLOOR — hand-feature baseline, no learning (increment-06a)
	python -m hoopvec.retrieve.run --n-games 12

retrieve-train:  ## TRAINED trajectory transformer — contrastive InfoNCE, recall@k (increment-06b)
	python -m hoopvec.retrieve.train --n-games 12 --epochs 300

retrieve-end2end:  ## REAL tracker output -> tensor -> FAISS retrieval, reconstructed-vs-GT -> end2end_*.json
	python -m hoopvec.retrieve.end2end

retrieve-bcast-encoder:  ## in-domain broadcast encoder (image coords), recon-vs-GT on a held-out game -> broadcast_encoder_*.json
	python -m hoopvec.retrieve.broadcast_encoder

retrieve-oneclip:  ## ONE-clip end-to-end demo frames->...->retrieval (needs a checkpoint from retrieve-train) -> end2end_oneclip_*.json
	python -m hoopvec.retrieve.from_tracks --checkpoint $$(ls -t weights/retrieve/*.pt | head -1)

retrieve-study:  ## reconstructed-vs-GT degradation study (increment-07, the headline finding)
	python -m hoopvec.retrieve.study --n-games 12 --epochs 300

retrieve-semantic-validate:  ## VALIDATE semantic retrieval — supervised (SupCon) vs floor/SSL/random -> semantic_validate_*.json
	python -m hoopvec.retrieve.semantic_validate --scheme transition

retrieve-nl:  ## NL play query demo — text -> semantic constraints -> matching possessions (structured, no LLM)
	python -m hoopvec.retrieve.nl_query

retrieve-uncertainty:  ## V2 retrieval uncertainty: is confidence calibrated? selective prediction (needs a checkpoint)
	python -m hoopvec.retrieve.uncertainty --checkpoint $$(ls -t weights/retrieve/*.pt | head -1) --device cpu

setarch-capacity:  ## capacity-matched set-arch d64 vs d128 crop recall. GPU: DEVICE=cuda EPOCHS=150 make setarch-capacity
	python -m hoopvec.retrieve.setarch_capacity --epochs $(or $(EPOCHS),60) --device $(or $(DEVICE),cpu)

retrieve-semantic:  ## semantic transfer probe — precision@5 by coarse play-type bucket (floor/trained/random)
	python -m hoopvec.retrieve.semantic_probe --config configs/semantic_probe.yaml

homography-frontend:  ## court-keypoint front-end harness + trivial floor -> eval_results/court_keypoints_floor_*.json
	python -m hoopvec.homography.keypoints --data-dir data/deepsport

homography-detector:  ## train the court-keypoint detector (arena-split) -> weights/ + eval_results/court_keypoints_detector_*.json
	python -m hoopvec.homography.train_keypoints --data-dir data/deepsport --epochs 40

jersey-eval:  ## jersey-OCR coverage ablation on SportsMOT GT-boxed athletes -> eval_results/jersey_ocr_*.json
	python -m hoopvec.reid.eval_jersey --limit-seqs 4

jersey-eval-tracker:  ## jersey-OCR coverage on REAL tracker output (the operating point) -> jersey_ocr_tracker_*.json
	python -m hoopvec.reid.eval_jersey --source tracker --tracker bytetrack_ft --seqs all --configs band_noprep

jersey-eval-stitch:  ## jersey-OCR coverage on tracker output WITH fragment stitching (the recovery lever)
	python -m hoopvec.reid.eval_jersey --source tracker --tracker bytetrack_ft --seqs all --configs band_noprep --stitch

jersey-accuracy:  ## jersey-OCR ACCURACY vs crop height (synthetic labels) -> eval_results/jersey_accuracy_*.json
	python -m hoopvec.reid.eval_jersey_accuracy

test:  ## run the test suite
	pytest -q

fmt:  ## format
	ruff format src tests

lint:  ## lint
	ruff check src tests
