"""Player identity: jersey-number OCR + appearance re-ID. (V1)

Plan: OSNet appearance embeddings + a scene-text jersey-number recognizer (DBNet + PARSeq) or a temporal
transformer over player crops. Fills Track.player_id. NOTE: least reliable stage (blur/occlusion) — report
identity accuracy honestly.
"""
from __future__ import annotations

from ..pipeline import Track


class ReIDIdentifier:
    def identify(self, tracks: list[Track], frames) -> list[Track]:
        raise NotImplementedError("re-ID: implement OSNet + jersey OCR. See design-doc.")
