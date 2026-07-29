"""CLI: train the court-keypoint detector (KaliCalib-lite) and measure it against the harness floor.

Split by ARENA (not by instant): the model must generalize to unseen cameras, not memorize a fixed camera's
keypoints. Trains the resnet18 heatmap net (keypoint_net.py) with masked MSE on Gaussian heatmaps, then scores
it exactly like the floor — detector -> solve H -> reprojection error vs GT calibration. MPS discipline
(inc-04): AMP off, small batch, 1-epoch probe first (`--epochs 1`); a non-finite loss raises.
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from .court import reprojection_error
from .keypoint_net import CourtKeypointNet, decode, make_target, preprocess
from .keypoints import _court_grid, _visible, evaluate, load_instants, solve_from_keypoints


def _arena(image_path: str) -> str:
    return Path(image_path).parent.parent.name        # data/deepsport/<ARENA>/<instant>/file.png


def _prep(instants):
    import cv2

    from .keypoint_net import IN_H, IN_W
    out = []
    for it in instants:
        if not it["image"]:
            continue
        img = cv2.imread(it["image"])
        if img is None:
            continue
        w, h = it["w"], it["h"]
        kp_in = {n: (u * IN_W / w, v * IN_H / h) for n, (u, v) in it["gt_keypoints"].items()}
        out.append({"img": cv2.resize(img, (IN_W, IN_H)), "kp_in": kp_in, "w": w, "h": h, "H_gt": it["H_gt"]})
    return out


def _augment(img, kp_in, rng):
    """Domain augmentation to force viewpoint/lighting robustness (the broadcast-gap mitigation, no GT for it):
    a random PERSPECTIVE warp (jitter the 4 corners) + photometric brightness/contrast/blur. Transforms the
    keypoints with the image; keypoints warped out of frame are dropped (masked)."""
    import cv2

    from .keypoint_net import IN_H, IN_W
    src = np.float32([[0, 0], [IN_W, 0], [IN_W, IN_H], [0, IN_H]])
    dst = (src + rng.uniform(-0.12, 0.12, src.shape) * np.float32([IN_W, IN_H])).astype(np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    img = cv2.warpPerspective(img, M, (IN_W, IN_H), borderMode=cv2.BORDER_REPLICATE)
    kp2 = {}
    for n, (u, v) in kp_in.items():
        p = M @ np.array([u, v, 1.0])
        u2, v2 = p[0] / p[2], p[1] / p[2]
        if 0 <= u2 < IN_W and 0 <= v2 < IN_H:
            kp2[n] = (u2, v2)
    im = np.clip(img.astype(np.float32) * rng.uniform(0.7, 1.3) + rng.uniform(-20, 20), 0, 255)
    if rng.random() < 0.3:
        k = int(rng.choice([3, 5]))
        im = cv2.GaussianBlur(im, (k, k), 0)
    return im.astype(np.uint8), kp2


def _batch(samples, device, rng=None):
    from .keypoint_net import IN_H, IN_W
    imgs, hms, masks = [], [], []
    for s in samples:
        img, kp = _augment(s["img"], s["kp_in"], rng) if rng is not None else (s["img"], s["kp_in"])
        imgs.append(preprocess(img))
        hm, mask = make_target(kp, IN_W, IN_H)
        hms.append(hm)
        masks.append(mask)
    return (torch.stack(imgs).to(device), torch.from_numpy(np.stack(hms)).to(device),
            torch.from_numpy(np.stack(masks)).to(device))


def _eval(model, val, device, conf) -> tuple[list[float], int]:
    grid = _court_grid()
    errs = []
    model.eval()
    with torch.no_grad():
        for s in val:
            hm = model(preprocess(s["img"]).unsqueeze(0).to(device))[0].cpu().numpy()
            H_hat = solve_from_keypoints(decode(hm, s["w"], s["h"], conf)[0])
            pts = _visible(s["H_gt"], grid, s["w"], s["h"])
            if H_hat is None or len(pts) < 6:
                continue
            e = reprojection_error(H_hat, s["H_gt"], pts)
            if math.isfinite(e):                           # drop degenerate-H solves (imperfect keypoints)
                errs.append(e)
    return errs, len(errs)                                 # `solved` = registrations that actually resolved


def run(args) -> dict:
    inst = [it for it in load_instants(args.data_dir) if it["image"]]
    arenas = sorted({_arena(it["image"]) for it in inst})
    val_arenas = set(arenas[:: args.val_every])                    # hold out unseen cameras
    tr = _prep([it for it in inst if _arena(it["image"]) not in val_arenas])
    va = _prep([it for it in inst if _arena(it["image"]) in val_arenas])
    floor = np.median(evaluate(None, [it for it in inst if _arena(it["image"]) in val_arenas]))

    torch.manual_seed(args.seed)
    if args.device == "mps" and hasattr(torch, "mps"):
        torch.mps.manual_seed(args.seed)
    model = CourtKeypointNet(pretrained=True).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    rng = np.random.default_rng(args.seed)

    best, curve, t0 = math.inf, [], time.time()
    for ep in range(args.epochs):
        model.train()
        perm = rng.permutation(len(tr))
        losses = []
        for i in range(0, len(tr), args.batch):
            x, hm, mask = _batch([tr[j] for j in perm[i:i + args.batch]], args.device,
                                 rng if args.augment else None)
            loss = (((model(x) - hm) ** 2).mean(dim=(2, 3)) * mask).sum() / (mask.sum() + 1e-6)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        ml = float(np.mean(losses))
        if not math.isfinite(ml):
            raise RuntimeError(f"non-finite loss at epoch {ep} — MPS guard (inc-04): AMP off, lower lr/batch.")
        if ep % args.eval_every == 0 or ep == args.epochs - 1:
            errs, solved = _eval(model, va, args.device, args.conf)
            med = float(np.median(errs)) if errs else math.inf
            curve.append({"epoch": ep, "loss": round(ml, 5), "val_reproj_median_px": round(med, 2),
                          "val_solve_rate": round(solved / max(len(va), 1), 3)})
            print(f"  epoch {ep:3d}  loss {ml:.5f}  val reproj median {med:.1f}px  "
                  f"solve {solved}/{len(va)}", flush=True)
            if med < best:
                best = med
                Path(args.weights).parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), args.weights)

    errs, solved = _eval(model, va, args.device, args.conf)
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "increment": "court-keypoint-detector",
        "stage": "homography front-end — learned court-keypoint detector (KaliCalib-lite)",
        "dataset": {"name": "DeepSportradar", "n_instants": len(inst), "n_train": len(tr), "n_val": len(va),
                    "split": "by arena (held-out cameras)", "val_arenas": sorted(val_arenas)},
        "model": {"arch": "resnet18(pretrained) encoder + upsample decoder -> 7 heatmaps",
                  "n_params": sum(p.numel() for p in model.parameters()),
                  "input": [288, 384], "keypoints": 7},
        "training": {"epochs": args.epochs, "batch": args.batch, "lr": args.lr, "device": args.device,
                     "augment": bool(args.augment), "seconds": round(time.time() - t0, 1), "amp": False,
                     "curve": curve},
        "results": {
            "trivial_floor_median_px": round(float(floor), 2),
            "detector_val_reproj_median_px": round(float(np.median(errs)), 2) if errs else None,
            "detector_val_reproj_mean_px": round(float(np.mean(errs)), 2) if errs else None,
            "detector_val_solve_rate": round(solved / max(len(va), 1), 3),
            "best_val_reproj_median_px": round(float(best), 2),
        },
        "provenance": {"seed": args.seed, "weights": args.weights, "platform": platform.platform()},
        "notes": "Beats the 503px floor iff the detector's keypoints yield a lower-reproj H on UNSEEN arenas. "
        "DeepSportradar is fixed arena cameras, not moving broadcast (stated boundary). Reported as-is.",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the court-keypoint detector (increment: homography front-end)")
    ap.add_argument("--data-dir", default="data/deepsport")
    ap.add_argument("--out-dir", default="eval_results")
    ap.add_argument("--weights", default="weights/court_keypoints.pt")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--val-every", type=int, default=5)     # every 5th arena -> val (~16%, 3 held-out arenas)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--augment", type=int, default=1)      # perspective + photometric domain augmentation
    args = ap.parse_args()

    report = run(args)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    (out / f"court_keypoints_detector_{stamp}.json").write_text(json.dumps(report, indent=2))
    r = report["results"]
    print(f"\nWrote court_keypoints_detector_{stamp}.json")
    print(f"  floor {r['trivial_floor_median_px']}px -> detector val reproj median "
          f"{r['detector_val_reproj_median_px']}px (solve {r['detector_val_solve_rate']}) "
          f"best {r['best_val_reproj_median_px']}px")


if __name__ == "__main__":
    main()
