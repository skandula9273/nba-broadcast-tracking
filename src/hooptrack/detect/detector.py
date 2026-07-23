"""Detection stage: players + ball from broadcast frames.

Choice is a config lever (`detect.model`): yolo | rfdetr | yolox. For increment-01 the choice is **yolo**
(Ultralytics) — AGPL-3.0, so the repo is AGPL; documented + CoreML-exportable. Implements the `Detector`
protocol in pipeline.py.

Honest scope for the SportsMOT baseline: we use the **COCO-pretrained** checkpoint and keep only the
`person` class (`detect.person_class`) as an athlete proxy. It is NOT fine-tuned on basketball, so it also
fires on referees/bench/crowd that COCO calls "person" — those become false positives against the
athlete-only GT and depress DetA. That gap is the honest baseline, reported as-is; fine-tuning is a later
measured ablation. API verified against ultralytics 8.4 (rule #1).
"""
from __future__ import annotations

from ..config import DetectConfig
from ..pipeline import Detection

# Resolved when detect.weights is null. COCO-pretrained; recorded verbatim in the eval JSON.
DEFAULT_WEIGHTS = "yolov8m.pt"


class YOLODetector:
    """Ultralytics YOLO. NOTE: AGPL-3.0 — using this makes the repo AGPL."""

    def __init__(self, cfg: DetectConfig) -> None:
        self.cfg = cfg
        self.weights = cfg.weights or DEFAULT_WEIGHTS
        self._model = None

    def _load(self):
        # lazy import so the package imports without the CV stack installed
        from ultralytics import YOLO

        self._model = YOLO(self.weights)
        return self._model

    def detect(self, frames) -> list[Detection]:
        """frames: an iterable yielding (frame_idx_1based, image_path) — e.g. a MotSequence.

        Batches paths to Ultralytics (which decodes as BGR) and keeps only `person`. Returns image-coord
        boxes; homography/identity are added downstream in V1.
        """
        model = self._model or self._load()
        cfg = self.cfg
        dets: list[Detection] = []
        batch_idx: list[int] = []
        batch_paths: list[str] = []

        def flush() -> None:
            if not batch_paths:
                return
            results = model.predict(
                source=list(batch_paths),
                conf=cfg.conf,
                iou=cfg.iou,
                imgsz=cfg.imgsz,
                classes=[cfg.person_class],
                device=cfg.device,
                verbose=False,
            )
            for fidx, r in zip(batch_idx, results):
                b = r.boxes
                xyxy = b.xyxy.cpu().numpy()
                confs = b.conf.cpu().numpy()
                for (x1, y1, x2, y2), c in zip(xyxy, confs):
                    dets.append(
                        Detection(
                            frame=int(fidx),
                            cls="player",
                            xyxy=(float(x1), float(y1), float(x2), float(y2)),
                            conf=float(c),
                        )
                    )
            batch_idx.clear()
            batch_paths.clear()

        for fidx, path in frames:
            batch_idx.append(int(fidx))
            batch_paths.append(str(path))
            if len(batch_paths) >= cfg.batch:
                flush()
        flush()
        return dets


def build_detector(cfg: DetectConfig):
    if cfg.model == "yolo":
        return YOLODetector(cfg)
    raise NotImplementedError(f"detector '{cfg.model}' not wired yet (options: yolo | rfdetr | yolox).")
