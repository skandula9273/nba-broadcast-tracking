# Increment-10 — one demonstrated end-to-end path (frames → retrieval), fully honest

**Goal.** Wire and *run* a single clip through the entire path — broadcast frames → detect → track → court
coords → possession tensor → embedding → retrieval — and be explicit about every crutch. **Not** to produce a
clean number; one clip, honestly, is the deliverable. Scoped deliberately: no homography front-end, no re-ID,
no generalization to many clips.

Run: `python -m hooptrack.retrieve.from_tracks --checkpoint weights/retrieve/<arch>_<ts>.pt`
(commit: `eval_results/end2end_oneclip_*.json`; clip `v_00HRwkvvjtQ_c007`, window frames 1–48.)

## What actually ran

1. **frames → detect → track (real).** The shared `Pipeline` (fine-tuned YOLOv8m athlete detector → ByteTrack)
   decoded 60 frames of the clip and produced tracks; the window 1–48 held **10 tracks**.
2. **court coords (hand-clicked H).** Foot-points projected through a homography fit to 4 hand-clicked
   correspondences (`tests/fixtures/court_correspondences_v_00HRwkvvjtQ_c007.json`), normalized to the 94×50 court.
3. **possession tensor.** `from_tracks.track_result_to_tensor` → a `(T=48, 11, 2)` tensor (entity 0 = ball,
   zeroed; 10 players in slots 1–10).
4. **embedding.** Encoded with the saved inc-06b checkpoint (`trajectory_transformer_20260730T025617Z.pt`,
   the H6 checkpointing) — so the query and the gallery use the *same* model.
5. **retrieval.** Cosine top-5 against the SportVU 2015-16 gallery (2,612 possessions).

## What was hand-supplied / substituted (every crutch)

- **Homography — HAND-CLICKED, and crude.** 4 correspondences eyeballed on frame 1. Their own reprojection
  error is **0 ft by construction** (4 point-pairs fully determine an 8-DOF homography — a tautology, not
  accuracy). On **held-out** check points the reprojection error is **30.5 ft** — on a 94×50 court, off by a
  third of its length. It's a single static H for a *moving* broadcast camera, and the clip is an **NCAA**
  court (narrower lane/3pt) mapped into the **NBA** coordinate frame the encoder was trained on. This is a
  manual stand-in for the (unbuilt) homography front-end, and it is badly inaccurate.
- **re-ID — SUBSTITUTED by an ordering rule.** Track IDs are **not** player identities. The 10 longest tracks
  in the window are ordered by **(track length desc, then first-frame x asc)** and dropped into slots 1–10.
  This is deterministic but **arbitrary** — it is *exactly* the player-slot permutation / order corruption the
  degradation study (inc-07/08/09) models, and that study showed the order-sensitive encoder is **fragile** to
  it (combined realistic recall@1 0.68, and 0.27 once the measured re-ID rate is folded in).
- **No ball.** The detector is athlete-only, so entity 0 (ball) is zeroed — out-of-distribution for a slot the
  encoder always saw filled.
- **Gaps / track count.** Per-track linear interpolation bridges gaps; fewer than 10 tracks would zero-pad.
- **Cross-corpus gallery.** The query is a crude NCAA-clip reconstruction; the gallery is NBA SportVU (different
  league/season). "Similar" across that gap is not well-defined.

## The top-5 (as it came out)

| rank | cosine | SportVU game | event |
|---|---|---|---|
| 1 | 0.775 | 01.02.2016.HOU.at.SAS | ev244 |
| 2 | 0.708 | 01.02.2016.MIL.at.MIN | ev508 |
| 3 | 0.679 | 01.01.2016.NYK.at.CHI | ev140 |
| 4 | 0.658 | 01.01.2016.NYK.at.CHI | ev63 |
| 5 | 0.637 | 01.02.2016.BKN.at.BOS | ev580 |

## Honest assessment — the neighbours are not meaningful, and that is the result

The path **runs end to end**, but the retrieved neighbours are **not demonstrably similar plays**. The
similarities (0.64–0.78) sit in the encoder's generic, poorly-separated band — the same 0.3–0.9 range it
produces for most pairs — and the five hits are unrelated SportVU possessions from random games with no
recognizable relationship to the query clip. There is no labelled "correct" neighbour to score against, and
given the crutches there is no reason to trust these as similar plays.

This is the **empirical version of the degradation study's prediction**, not a bug:
- the homography is ~30 ft off, so the court coordinates are badly wrong;
- the re-ID substitute assigns players to **arbitrary slots**, which the study measured as the single most
  damaging corruption for this order-sensitive encoder;
- the ball is absent.

A crude reconstruction feeding an order-sensitive encoder returns arbitrary neighbours — which is precisely
what inc-07/08/09 said would happen. A clean-looking top-5 here would have been *less* honest than this. The
levers that would make this real are the ones already identified and (mostly) built or characterized elsewhere:
a real homography front-end (needs broadcast court GT), re-ID (jersey OCR resolves ~a quarter of tracks on
broadcast; stitching recovers a third of the lost coverage), and ball tracking — each measured in its own
increment, none faked here.
