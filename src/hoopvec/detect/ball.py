"""Ball detection via COCO 'sports ball' (class 32) — no training, MEASURED coverage.

The fine-tuned detector is single-class (athlete); true possessions/shots need the ball. COCO yolov8m already
has a 'sports ball' class, so ball detection needs no new training — but broadcast basketballs are small, fast,
and occluded, so coverage must be MEASURED, not assumed (and there's no ball GT in SportsMOT, so it's coverage
= fraction of frames with a confident ball, not verified accuracy — same honest framing as jersey OCR).

`BallDetector.detect(frames)` returns the best (highest-conf) ball center per frame. `analytics.possessions`
uses it, where present, to attribute the ball-handler (nearest player) and segment true possessions — upgrading
the honest 'no ball -> phases only' fallback for the frames the ball is actually visible.
"""
from __future__ import annotations

BALL_CLASS = 32   # COCO 'sports ball'


class BallDetector:
    """COCO 'sports ball' detector. Returns {frame_idx: (cx, cy, conf)} — the most confident ball per frame."""

    def __init__(self, weights: str = "yolov8m.pt", device: str = "mps", conf: float = 0.25,
                 imgsz: int = 1280, batch: int = 16) -> None:
        self.weights = weights
        self.device = device
        self.conf = conf
        self.imgsz = imgsz
        self.batch = batch
        self._model = None

    def _load(self):
        from ultralytics import YOLO
        self._model = YOLO(self.weights)
        return self._model

    def detect(self, frames) -> dict[int, tuple[float, float, float]]:
        model = self._model or self._load()
        out: dict[int, tuple[float, float, float]] = {}
        idxs: list[int] = []
        paths: list[str] = []

        def flush() -> None:
            if not paths:
                return
            for fidx, r in zip(idxs, model.predict(source=list(paths), classes=[BALL_CLASS], conf=self.conf,
                                                   imgsz=self.imgsz, device=self.device, verbose=False)):
                b = r.boxes
                if len(b) == 0:
                    continue
                xyxy = b.xyxy.cpu().numpy()
                confs = b.conf.cpu().numpy()
                j = int(confs.argmax())                          # keep the most confident ball in the frame
                x1, y1, x2, y2 = xyxy[j]
                out[int(fidx)] = ((x1 + x2) / 2.0, (y1 + y2) / 2.0, float(confs[j]))
            idxs.clear()
            paths.clear()

        for fidx, path in frames:
            idxs.append(int(fidx))
            paths.append(str(path))
            if len(paths) >= self.batch:
                flush()
        flush()
        return out
