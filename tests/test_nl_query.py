"""NL play query — keyword parse + semantic filter, numpy only, CI-safe."""

import numpy as np

from hoopvec.retrieve.nl_query import parse_query, query_corpus


def test_parse_query_maps_and_composes_keywords():
    assert parse_query("show me transition plays") == {"transition": "transition"}
    assert parse_query("half court sets on the left") == {"transition": "halfcourt", "initiation_side": "left"}
    assert parse_query("isolation on the right") == {"handler_change": "handler_low", "initiation_side": "right"}
    assert parse_query("completely unrelated text") == {}


def test_query_corpus_filters_by_transition():
    thr = {"transition": {"advance_frames": 10, "advance_thresh": 0.05}}
    T = 48
    trans = np.zeros((T, 11, 2))
    trans[:, 0, 0] = np.linspace(0.0, 0.5, T)   # ball advances -> transition
    half = np.zeros((T, 11, 2))
    half[:, 0, 0] = 0.1                          # ball static -> halfcourt
    corpus = np.stack([trans, half])
    r = query_corpus(corpus, thr, "a transition play")
    assert r["constraints"] == {"transition": "transition"} and r["results"] == [0]


def test_query_corpus_no_keyword_returns_empty():
    r = query_corpus(np.zeros((3, 48, 11, 2)), {}, "xyzzy")
    assert r["n_matches"] == 0 and r["results"] == []
