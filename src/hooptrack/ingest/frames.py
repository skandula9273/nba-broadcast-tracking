"""Frame sources for the pipeline (used by serve + eval).

Two sources share the `frames` slot the pipeline consumes:
  - broadcast clips (V1) -> `extract_frames` (OpenCV/ffmpeg decode + shot segmentation), still a stub;
  - MOT-Challenge sequences (V0 SportsMOT) -> `MotSequence`, a lightweight reader below.

A `MotSequence` carries ordered frame *paths* plus `(width, height, fps)` from `seqinfo.ini` — NOT decoded
pixels — so a 1500-frame 720p sequence costs kilobytes here, and the detector reads images lazily in
mini-batches. This keeps long sequences memory-safe while still flowing through the one shared pipeline path.
"""
from __future__ import annotations

import configparser
from dataclasses import dataclass
from pathlib import Path

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def extract_frames(clip_path: str):
    raise NotImplementedError("frames.extract_frames: OpenCV/ffmpeg decode + shot segmentation (V1).")


@dataclass
class MotSequence:
    """One MOT-Challenge sequence: ordered frame paths + metadata. Iterating yields (frame_idx_1based, path)."""

    name: str
    seq_dir: Path
    frame_paths: list[Path]
    width: int
    height: int
    fps: float
    seq_length: int          # annotated length from seqinfo (may exceed len(frame_paths) when capped)

    def __len__(self) -> int:
        return len(self.frame_paths)

    def __iter__(self):
        for i, p in enumerate(self.frame_paths, start=1):
            yield i, p


def read_seqinfo(seq_dir: str | Path) -> dict:
    """Parse an MOT `seqinfo.ini` into a plain dict (with sane fallbacks)."""
    seq_dir = Path(seq_dir)
    cp = configparser.ConfigParser()
    read = cp.read(seq_dir / "seqinfo.ini")
    s = cp["Sequence"] if (read and cp.has_section("Sequence")) else {}
    get = s.get if hasattr(s, "get") else (lambda k, d=None: d)
    return {
        "name": get("name", seq_dir.name),
        "imDir": get("imDir", "img1"),
        "frameRate": float(get("frameRate", 25) or 25),
        "seqLength": int(get("seqLength", 0) or 0),
        "imWidth": int(get("imWidth", 0) or 0),
        "imHeight": int(get("imHeight", 0) or 0),
        "imExt": get("imExt", ".jpg"),
    }


def load_mot_sequence(seq_dir: str | Path, max_frames: int | None = None) -> MotSequence:
    """Build a `MotSequence` from a `<seq>/` dir containing `img1/` + `seqinfo.ini`."""
    seq_dir = Path(seq_dir)
    info = read_seqinfo(seq_dir)
    img_dir = seq_dir / info["imDir"]

    ext = info["imExt"].lower()
    # skip hidden/AppleDouble files (e.g. macOS `._000001.jpg` resource forks) — not real images
    imgs = [p for p in img_dir.iterdir() if not p.name.startswith(".")]
    paths = sorted(p for p in imgs if p.suffix.lower() == ext)
    if not paths:  # fall back to any common image extension
        paths = sorted(p for p in imgs if p.suffix.lower() in _IMG_EXTS)
    if max_frames is not None:
        paths = paths[:max_frames]

    width, height = info["imWidth"], info["imHeight"]
    if (width <= 0 or height <= 0) and paths:  # seqinfo missing dims -> read the first frame
        import cv2

        img = cv2.imread(str(paths[0]))
        if img is not None:
            height, width = img.shape[:2]

    return MotSequence(
        name=info["name"],
        seq_dir=seq_dir,
        frame_paths=paths,
        width=width,
        height=height,
        fps=info["frameRate"],
        seq_length=info["seqLength"] or len(paths),
    )
