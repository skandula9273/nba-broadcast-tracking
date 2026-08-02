"""Video-decode (ingest.extract_frames) test — cv2-gated. Synthesizes a tiny clip and decodes it, so it
needs no fixture file and stays hermetic."""

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from hoopvec.ingest.frames import extract_frames  # noqa: E402


def _make_clip(path, n=10, w=64, h=48, fps=25):
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    assert vw.isOpened()
    for i in range(n):
        frame = np.full((h, w, 3), i * 20 % 255, dtype=np.uint8)   # distinct solid colour per frame
        vw.write(frame)
    vw.release()


def test_extract_frames_decodes_a_clip(tmp_path):
    clip = tmp_path / "clip.mp4"
    _make_clip(clip, n=10)
    seq = extract_frames(clip, tmp_path / "out")
    assert len(seq) == 10                                  # all frames decoded
    assert seq.width == 64 and seq.height == 48
    assert (tmp_path / "out" / "img1" / "000001.jpg").exists()
    assert (tmp_path / "out" / "seqinfo.ini").exists()
    idxs = [i for i, _ in seq]
    assert idxs == list(range(1, 11))                      # 1-indexed, ordered — pipeline-ready


def test_extract_frames_subsample_and_cap(tmp_path):
    clip = tmp_path / "clip.mp4"
    _make_clip(clip, n=12)
    seq = extract_frames(clip, tmp_path / "o2", every=2, max_frames=3)   # every 2nd, capped at 3
    assert len(seq) == 3


def test_extract_frames_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        extract_frames(tmp_path / "nope.mp4", tmp_path / "out")
