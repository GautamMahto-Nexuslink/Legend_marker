#!/usr/bin/env python3
"""
benchmark_icon_db.py
====================

Measure the known-icon ("blue") stage: leave-one-out over every icon in
``icons_glyph_db.npz``.

Each icon is used as a query against a database that EXCLUDES every exemplar
from its own source file (the icon itself), then we ask two questions:

  coverage  — how often does the stage produce a hit at all?
  accuracy  — of the hits, how often is the class right?

A stage that answers almost nothing is useless even at 100% accuracy, which is
the problem with a fixed Hamming cutoff; a stage that answers everything wrongly
is worse.  Both numbers are therefore reported for every strategy, plus the
"effective yield" (coverage x accuracy) = share of icons correctly named.

Strategies compared:

  hamming<=N      current behaviour: single pHash, fixed cutoff
  knn-vote        multi-hash prefilter + per-class vote, decided by margin
  glyph-rescore   the above, then re-score the shortlist with the pipeline's own
                  template+ORB matcher (needs the stored glyphs)
  glyph+groups    the same, but exemplars vote per SEMANTIC GROUP, so
                  "Restroom"/"Restrooms" stop splitting their votes

A prediction counts as correct when it lands in the right semantic group:
answering "Restrooms" for a "Restroom" icon is not an error, and the pipeline
prefers the legend's own wording anyway.  Pass ``--strict`` to demand the exact
class string instead.

Usage::

    python3 benchmark_icon_db.py                 # 800-icon sample, fast
    python3 benchmark_icon_db.py --limit 0       # every icon (slow)
    python3 benchmark_icon_db.py --degrade       # rescale/blur queries first
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

import legend_marker as lm
from legend_pipeline.containers import VisualSignature
from legend_pipeline.icon_db import IconDatabase
from legend_pipeline.synonyms import DEFAULT_SYNONYMS_PATH, SynonymMap

POPCOUNT = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(1)


def hamming(db_packed: np.ndarray, q_packed: np.ndarray) -> np.ndarray:
    return POPCOUNT[np.bitwise_xor(db_packed, q_packed[None, :])].sum(axis=1)


def degrade(glyph: np.ndarray, rng: random.Random) -> np.ndarray:
    """Simulate what a map icon looks like vs its legend/dataset exemplar."""
    g = glyph
    scale = rng.uniform(0.55, 0.85)
    small = cv2.resize(g, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    g = cv2.resize(small, (glyph.shape[1], glyph.shape[0]),
                   interpolation=cv2.INTER_CUBIC)
    if rng.random() < 0.6:
        g = cv2.GaussianBlur(g, (3, 3), 0.7)
    if rng.random() < 0.5:
        noise = rng.uniform(3, 9)
        g = np.clip(g.astype(np.float32)
                    + np.random.normal(0, noise, g.shape), 0, 255).astype(np.uint8)
    if rng.random() < 0.4:                     # JPEG-ish ringing
        ok, enc = cv2.imencode(".jpg", g, [cv2.IMWRITE_JPEG_QUALITY, rng.randint(45, 80)])
        if ok:
            g = cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)
    return g


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default="icons_glyph_db.npz")
    ap.add_argument("--limit", type=int, default=800,
                    help="Query sample size (0 = all icons).")
    ap.add_argument("--degrade", action="store_true",
                    help="Rescale/blur/JPEG the query glyph first (realistic).")
    ap.add_argument("--strict", action="store_true",
                    help="Require the exact class string, not just the group.")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args(argv)

    db = IconDatabase.load(args.db)
    if not db:
        print(f"could not load {args.db}", file=sys.stderr)
        return 1
    syn = SynonymMap.load(DEFAULT_SYNONYMS_PATH)
    grouped = IconDatabase.load(args.db, synonyms=syn)

    def same(pred: Optional[str], truth: str) -> bool:
        """Right answer? By semantic group unless --strict."""
        if pred is None:
            return False
        if args.strict:
            return pred == truth
        return (syn.canonical(pred) or pred.lower()) == \
               (syn.canonical(truth) or truth.lower())

    print(f"database: {len(db)} icons, {db.n_classes} classes, "
          f"{len(set(grouped.group_of(n) for n in db.names.tolist()))} semantic groups"
          f"  [judged {'strictly' if args.strict else 'by group'}]\n")

    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    idx = list(range(len(db)))
    if args.limit:
        idx = rng.sample(idx, min(args.limit, len(idx)))

    config = lm.PipelineConfig()
    matcher = lm.SignatureMatcher(config)
    builder = lm.SignatureBuilder(config)

    # ---- strategies -------------------------------------------------------
    CUTOFFS = [0, 10, 20, 40]
    results: Dict[str, List[Tuple[bool, bool]]] = {}      # name -> [(hit, correct)]
    for c in CUTOFFS:
        results[f"hamming<={c}"] = []
    results["knn-vote"] = []
    results["glyph-rescore"] = []
    results["glyph+groups"] = []

    t0 = time.perf_counter()
    for n, i in enumerate(idx):
        truth = db.names[i]
        query_glyph = degrade(db.glyphs[i], rng) if args.degrade else db.glyphs[i]

        # Leave-one-out: hide the exact exemplar we are querying with.
        mask = np.ones(len(db), dtype=bool)
        mask[i] = False

        qsig = VisualSignature()
        qsig.glyph = query_glyph
        try:
            kp, des = builder._orb.detectAndCompute(query_glyph, None)
            qsig.keypoints, qsig.orb_descriptors = kp, des
        except Exception:
            pass
        qph, qdh, qah = db.hash_query(query_glyph)

        # -- fixed-cutoff pHash (the current stage) -------------------------
        dists = hamming(db.phash[mask], qph)
        best = int(dists.argmin())
        best_name = db.names[mask][best]
        best_dist = int(dists[best])
        for c in CUTOFFS:
            hit = best_dist <= c
            results[f"hamming<={c}"].append((hit, hit and same(best_name, truth)))

        # -- multi-hash kNN + class vote ------------------------------------
        vote = db.match_hashes(qph, qdh, qah, mask=mask)
        results["knn-vote"].append((vote.name is not None, same(vote.name, truth)))

        # -- + glyph re-scoring with the pipeline's matcher ------------------
        full = db.match(qsig, matcher, mask=mask)
        results["glyph-rescore"].append((full.name is not None, same(full.name, truth)))

        # -- + votes aggregated per semantic group --------------------------
        best_group = grouped.match(qsig, matcher, mask=mask)
        results["glyph+groups"].append(
            (best_group.name is not None, same(best_group.name, truth)))

        if n and n % 200 == 0:
            print(f"  ...{n}/{len(idx)}", file=sys.stderr)

    elapsed = time.perf_counter() - t0

    # ---- report -----------------------------------------------------------
    print(f"{'strategy':<18}{'coverage':>10}{'accuracy':>10}{'yield':>10}"
          f"{'  (correct/answered/total)'}")
    print("-" * 74)
    for name, rows in results.items():
        answered = sum(1 for hit, _ in rows if hit)
        correct = sum(1 for _, ok in rows if ok)
        total = len(rows)
        cov = answered / total if total else 0
        acc = correct / answered if answered else 0
        print(f"{name:<18}{cov:>9.1%}{acc:>10.1%}{correct/total:>10.1%}"
              f"   {correct}/{answered}/{total}")
    print(f"\n{len(idx)} queries in {elapsed:.1f}s "
          f"({1000*elapsed/max(1,len(idx)):.1f} ms/query)"
          f"{'  [degraded queries]' if args.degrade else ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
