"""Config: ablation config (yaml) split from secrets (env). One loader, pydantic-validated.

Mirrors the SEC project's split so every knob is a config lever and secrets never touch the yaml.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class Secrets(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str | None = None      # pgvector option for the retrieval index
    wandb_api_key: str | None = None

    def require(self, *names: str) -> None:
        missing = [n for n in names if not getattr(self, n, None)]
        if missing:
            raise RuntimeError(
                f"Missing required secrets: {', '.join(missing)}. Set them in .env or the environment."
            )


class DetectConfig(BaseModel):
    model: str = "yolo"                  # yolo | rfdetr | yolox
    weights: str | None = None           # null -> a concrete pretrained default resolved in detect/detector.py
    conf: float = 0.25
    iou: float = 0.7
    classes: list[str] = ["player", "ball"]
    device: str = "mps"                  # mps (Apple) | cpu | cuda:0 — recorded in the eval JSON
    imgsz: int = 1280                    # YOLO inference size; SportsMOT frames are 720p broadcast
    batch: int = 16                      # frames per YOLO forward pass (MPS-friendly)
    person_class: int = 0                # COCO 'person' index — the SportsMOT athlete proxy (pretrained, not fine-tuned)


class TrackConfig(BaseModel):
    method: str = "bytetrack"            # bytetrack | botsort | deepeiou
    min_conf: float = 0.1                # boxmot ByteTrack low-score gate (2nd association stage)
    use_reid: bool = False               # appearance fusion off for ByteTrack; on for botsort
    # boxmot ByteTrack levers (names verified against boxmot 22). frame_rate is taken per-sequence.
    track_thresh: float = 0.45           # high-score gate (1st association stage)
    match_thresh: float = 0.8            # IoU match threshold
    track_buffer: int = 25               # frames a lost track is kept before deletion
    # BoT-SORT (appearance + camera-motion) levers; boxmot's tuned botsort.yaml drives the thresholds.
    use_cmc: bool = True                 # camera-motion compensation (BoT-SORT); off = isolate CMC's effect
    device: str = "mps"                  # device for the ReID model (mps | cpu | cuda:0)
    reid_weights: str = "osnet_x0_25_msmt17.pt"   # OSNet ReID weight (boxmot auto-downloads)


class HomographyConfig(BaseModel):
    method: str = "kalicalib"
    enabled: bool = False


class ReidConfig(BaseModel):
    model: str = "osnet"
    enabled: bool = False


class EmbeddingConfig(BaseModel):
    enabled: bool = False
    arch: str = "trajectory_transformer"   # trajectory_transformer (order-sensitive) | set_transformer (perm-invariant, inc-09)
    dim: int = 128                         # output embedding size (L2-normalized)
    objective: str = "contrastive"         # contrastive | denoising | masked
    window_seconds: int = 8
    # compact trajectory-transformer arch knobs (increment-06b) — small by design (MPS caution, inc-04)
    d_model: int = 128                     # token width inside the encoder
    n_heads: int = 4                       # attention heads
    n_layers: int = 2                      # TransformerEncoder depth
    ff_dim: int = 256                      # feed-forward width
    dropout: float = 0.1
    temperature: float = 0.1               # InfoNCE / NT-Xent temperature


class RetrievalConfig(BaseModel):
    enabled: bool = False
    index: str = "faiss"                    # faiss | pgvector
    top_k: int = 10


class EvalConfig(BaseModel):
    dataset: str = "sportsmot"
    data_dir: str = "data/sportsmot"
    split: str = "basketball"            # sport category (basketball | volleyball | football)
    mot_split: str = "val"               # MOT split with public GT (train | val; test GT is withheld)
    benchmark: str = "SportsMOT"         # TrackEval BENCHMARK label -> folder <benchmark>-<eval_split>
    metrics: list[str] = ["hota", "mota", "idf1"]   # via TrackEval; detection mAP added at the detect stage
    retrieval_ks: list[int] = [1, 5, 10]            # recall@k for V1
    out_dir: str = "eval_results"
    max_sequences: int | None = None     # smoke lever: cap sequences (null = all); recorded in the JSON
    max_frames: int | None = None        # smoke lever: cap frames per sequence (null = all)
    cache_detections: bool = True        # cache per-seq detections (keyed by detector cfg) -> cheap tracker ablations
    tracker_name: str | None = None      # output/eval subdir name (null -> track.method); distinguishes ablation variants

    @property
    def eval_split(self) -> str:
        """TrackEval SPLIT_TO_EVAL label, e.g. 'basketball-val'."""
        return f"{self.split}-{self.mot_split}"


class Config(BaseModel):
    seed: int = 13
    detect: DetectConfig = DetectConfig()
    track: TrackConfig = TrackConfig()
    homography: HomographyConfig = HomographyConfig()
    reid: ReidConfig = ReidConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    eval: EvalConfig = EvalConfig()


def load_config(path: str | Path) -> Config:
    data: dict[str, Any] = yaml.safe_load(Path(path).read_text()) or {}
    return Config.model_validate(data)
