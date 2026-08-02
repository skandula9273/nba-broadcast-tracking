"""Learned court-keypoint detector (KaliCalib-lite) — the front-end the harness (keypoints.py) measures.

A resnet18 (ImageNet-pretrained) encoder + a light upsampling decoder predicts one heatmap per canonical
court keypoint; argmax -> (u, v). `TrainedKeypointDetector` wraps it as the `image -> {name: (u,v)}` callable
the harness scores by reprojection error. Small dataset (728 instants) -> a pretrained backbone is essential.
Trained + evaluated in train_keypoints.py (arena-split, so it must generalize to unseen cameras, not memorize).
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .keypoints import CANONICAL

KEYPOINT_ORDER = list(CANONICAL)          # fixed channel order
IN_H, IN_W = 288, 384                     # network input (kept ~4:3, like the 1624x1234 court images)
OUT_H, OUT_W = IN_H // 4, IN_W // 4       # heatmap resolution
_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)


class CourtKeypointNet(nn.Module):
    def __init__(self, n_keypoints: int = len(KEYPOINT_ORDER), pretrained: bool = True) -> None:
        super().__init__()
        import torchvision

        b = torchvision.models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
        self.encoder = nn.Sequential(b.conv1, b.bn1, b.relu, b.maxpool, b.layer1, b.layer2, b.layer3, b.layer4)

        def up(cin, cout):
            return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(),
                                 nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False))
        self.decoder = nn.Sequential(up(512, 256), up(256, 128), up(128, 64),   # /32 -> /4
                                     nn.Conv2d(64, n_keypoints, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))               # (B, K, OUT_H, OUT_W)


def preprocess(bgr: np.ndarray) -> torch.Tensor:
    """cv2 BGR image -> normalized (3, IN_H, IN_W) tensor."""
    import cv2

    rgb = cv2.cvtColor(cv2.resize(bgr, (IN_W, IN_H)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.from_numpy(((rgb - _MEAN) / _STD).transpose(2, 0, 1).copy())


def make_target(gt_keypoints: dict, w: int, h: int, sigma: float = 2.0):
    """GT image keypoints (orig px) -> (K, OUT_H, OUT_W) Gaussian heatmaps + (K,) visibility mask."""
    ys = np.arange(OUT_H)[:, None]
    xs = np.arange(OUT_W)[None, :]
    hm = np.zeros((len(KEYPOINT_ORDER), OUT_H, OUT_W), np.float32)
    mask = np.zeros(len(KEYPOINT_ORDER), np.float32)
    for k, name in enumerate(KEYPOINT_ORDER):
        if name not in gt_keypoints:
            continue
        u, v = gt_keypoints[name]
        cx, cy = u * OUT_W / w, v * OUT_H / h
        hm[k] = np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sigma ** 2))
        mask[k] = 1.0
    return hm, mask


def decode(heatmaps: np.ndarray, w: int, h: int, conf: float = 0.3) -> tuple[dict, dict]:
    """(K, OUT_H, OUT_W) heatmaps -> {name: (u,v)} in ORIGINAL px for peaks above `conf`, plus peak values."""
    kps, peaks = {}, {}
    for k, name in enumerate(KEYPOINT_ORDER):
        hm = heatmaps[k]
        peak = float(hm.max())
        peaks[name] = round(peak, 3)
        if peak < conf:
            continue
        row, col = np.unravel_index(int(hm.argmax()), hm.shape)
        kps[name] = (float(col * w / OUT_W), float(row * h / OUT_H))
    return kps, peaks


class TrainedKeypointDetector:
    """`image -> {name: (u,v)}` using trained weights. Matches the harness's detector interface."""

    def __init__(self, weights_path: str, device: str = "cpu", conf: float = 0.3) -> None:
        self.device = device
        self.conf = conf
        self.model = CourtKeypointNet(pretrained=False).to(device)
        self.model.load_state_dict(torch.load(weights_path, map_location=device))
        self.model.eval()

    @torch.no_grad()
    def __call__(self, bgr: np.ndarray) -> dict:
        h, w = bgr.shape[:2]
        x = preprocess(bgr).unsqueeze(0).to(self.device)
        hm = self.model(x)[0].cpu().numpy()
        return decode(hm, w, h, self.conf)[0]


def learned_register(weights_path: str, device: str = "cpu", conf: float = 0.3):
    """A registration front-end for `CourtHomography` backed by the trained detector: detect keypoints in a
    frame's image -> solve H (court->image), or None if <4 confident keypoints (honest registration failure)."""
    from .keypoints import solve_from_keypoints

    det = TrainedKeypointDetector(weights_path, device, conf)
    frame_cache: dict[int, dict] = {}

    def register(frame_idx, frames):
        import cv2

        key = id(frames)
        if key not in frame_cache:
            frame_cache[key] = dict(iter(frames))          # frame_idx -> path (built once per sequence)
        path = frame_cache[key].get(frame_idx)
        img = cv2.imread(str(path)) if path is not None else None
        return solve_from_keypoints(det(img)) if img is not None else None

    return register
