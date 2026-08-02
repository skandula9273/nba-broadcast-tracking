"""Fragment stitching — spatiotemporal gap-closing logic, numpy only, CI-safe."""

from hoopvec.pipeline import Track
from hoopvec.reid.stitch import stitch_fragments


def _seg(tid, frames, cx, cy=0.0, w=20.0, h=40.0):
    return [Track(track_id=tid, frame=f, cls="athlete",
                  xyxy=(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)) for f in frames]


def _ids(tracks):
    return {t.track_id for t in tracks}


def test_links_same_player_across_a_short_gap():
    # id 1 ends at frame 5 near x=100; id 2 starts at frame 8 near x=105 -> same player, should merge.
    tracks = _seg(1, range(1, 6), cx=100.0) + _seg(2, range(8, 13), cx=105.0)
    out = stitch_fragments(tracks, max_gap=30, max_dist_factor=2.0)
    assert _ids(out) == {1}                       # merged to the canonical (smallest) id


def test_does_not_link_spatially_distant_fragments():
    # a gap in time but far apart (x=100 vs x=1000) -> different players, stay separate.
    tracks = _seg(1, range(1, 6), cx=100.0) + _seg(2, range(8, 13), cx=1000.0)
    out = stitch_fragments(tracks, max_gap=30, max_dist_factor=2.0)
    assert _ids(out) == {1, 2}


def test_does_not_merge_co_present_teammates():
    # two ids present in the SAME frames (temporally overlapping) are never the same player, even if close.
    tracks = _seg(1, range(1, 10), cx=100.0) + _seg(2, range(1, 10), cx=112.0)
    out = stitch_fragments(tracks, max_gap=30, max_dist_factor=5.0)
    assert _ids(out) == {1, 2}


def test_gap_too_long_is_not_linked():
    tracks = _seg(1, range(1, 6), cx=100.0) + _seg(2, range(60, 66), cx=100.0)
    out = stitch_fragments(tracks, max_gap=30, max_dist_factor=2.0)
    assert _ids(out) == {1, 2}


def test_chains_three_fragments_into_one():
    tracks = (_seg(3, range(1, 5), cx=100.0) + _seg(1, range(7, 11), cx=104.0)
              + _seg(2, range(13, 17), cx=108.0))
    out = stitch_fragments(tracks, max_gap=30, max_dist_factor=2.0)
    assert _ids(out) == {1}                       # all three chain -> canonical min id = 1
