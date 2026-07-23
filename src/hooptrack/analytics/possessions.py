"""Analytics on the reconstructed tracks: possession segmentation, shots, spacing, shot quality. (V1)

Feeds both the demo and the degradation study (analytics on reconstructed vs GT tracks).
"""
from __future__ import annotations

from ..pipeline import TrackResult


def segment_possessions(result: TrackResult):
    raise NotImplementedError("analytics: segment possessions from tracks. See design-doc.")
