"""Known-icon database — the "blue" stage: identify a map icon from a curated set.

The old stage compared ONE 256-bit pHash against a fixed Hamming cutoff.  That
is a poor identifier for rendered map symbols: at ``hash_size=16`` a cutoff of 10
demands 96% identical bits, so a symbol that was rescaled, re-antialiased or
drawn over terrain misses — and raising the cutoff cannot help, because a single
distance says nothing about how much better the winner is than its rivals.

This module replaces it with the retrieval pipeline such problems actually call
for:

1. **Prefilter** — three cheap hashes (pHash, dHash, aHash) score every entry at
   once with vectorised popcounts, keeping the best ``prefilter_k`` candidates.
   pHash captures coarse structure, dHash horizontal gradients, aHash overall
   ink; a symbol that survives all three is worth the expensive test.
2. **Re-score** — the shortlist is re-scored with the pipeline's *own*
   ``SignatureMatcher`` (multi-scale template correlation + ORB inliers) on the
   stored 64x64 glyph.  This is the same measure the legend stage trusts, and it
   is tolerant of scale and alignment in a way no hash is.
3. **Vote** — a class usually has several exemplars in the database, so scores
   are aggregated per class (mean of its best two, which rewards a class that
   agrees with itself and damps a single lucky exemplar).  Aggregation happens
   per *semantic group* when a synonym map is supplied: "Restroom",
   "Restrooms" and "Public Restroom" are one icon wearing three names, and
   letting them compete would split their votes and destroy the margin below
   (measured: 64% -> 88% coverage at equal accuracy).
4. **Decide** — accept only when the winning group clears an absolute floor AND
   beats the runner-up *group* by a margin.  The margin is what makes a loose
   floor safe: "much better than anything else" is evidence a fixed distance
   cannot express.

``IconDatabase`` also loads the legacy ``{phash_hex: classname}`` JSON, in which
case there are no glyphs to re-score and only steps 1, 3 and 4 run.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .deps import LOGGER, Image, imagehash

#: popcount lookup for one byte — turns XOR into a Hamming distance.
_POPCOUNT = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(1)

#: Prefilter weights: pHash is the most reliable, aHash the crudest.
HASH_WEIGHTS = {"phash": 0.55, "dhash": 0.30, "ahash": 0.15}


@dataclass
class IconDbMatch:
    """Outcome of one database lookup.

    ``name`` is None when nothing convincing was found; ``nearest_name`` /
    ``nearest_hamming`` always describe the closest entry so a report can show
    how near the miss was.
    """
    name: Optional[str] = None
    score: float = 0.0
    margin: float = 0.0
    group: Optional[str] = None           # semantic group that won the vote
    runner_up: Optional[str] = None
    runner_up_score: float = 0.0
    votes: int = 0                        # exemplars backing the winner
    nearest_name: Optional[str] = None
    nearest_hamming: Optional[int] = None
    method: str = "none"                  # "hash" | "glyph" | "none"
    rejected: Optional[str] = None        # why a candidate was refused
    rows: List[Dict[str, Any]] = field(default_factory=list)   # per-group detail


class IconDatabase:
    """Loaded icon database + the matching strategy described above."""

    def __init__(
        self,
        names: np.ndarray,
        phash: np.ndarray,
        dhash: Optional[np.ndarray] = None,
        ahash: Optional[np.ndarray] = None,
        glyphs: Optional[np.ndarray] = None,
        files: Optional[np.ndarray] = None,
        meta: Optional[dict] = None,
        *,
        prefilter_k: int = 24,
        min_score: float = 0.55,
        margin: float = 0.06,
        hash_weight: float = 0.20,
        hash_shortcut_hamming: int = 6,
        hash_min_similarity: float = 0.82,
        hash_margin: float = 0.04,
        votes_per_class: int = 2,
        synonyms=None,
    ) -> None:
        self.names = names
        self.phash = phash
        self.dhash = dhash
        self.ahash = ahash
        self.glyphs = glyphs
        self.files = files
        self.meta = meta or {}
        self.hash_size = int(self.meta.get("hash_size", 16))
        self.n_bits = int(self.meta.get("hash_bits", self.hash_size ** 2))

        self.prefilter_k = prefilter_k
        self.min_score = min_score
        self.margin = margin
        self.hash_weight = hash_weight
        self.hash_shortcut_hamming = hash_shortcut_hamming
        self.hash_min_similarity = hash_min_similarity
        self.hash_margin = hash_margin
        self.votes_per_class = votes_per_class

        # ORB descriptors are recomputed lazily, only for shortlisted entries.
        self._sig_cache: Dict[int, Any] = {}
        # class name -> semantic group, resolved once for the whole DB.
        self._group_of: Dict[str, str] = {}
        self.set_synonyms(synonyms)

    def set_synonyms(self, synonyms) -> None:
        """Attach a :class:`SynonymMap` so votes aggregate per semantic group.

        Without it every class name votes for itself, and near-duplicate classes
        ("Restroom" vs "Restrooms") split their exemplars between two rivals —
        which is exactly what the runner-up margin then reads as "ambiguous".
        """
        self.synonyms = synonyms
        self._group_of = {}
        if synonyms and len(self):
            for name in set(self.names.tolist()):
                key = str(name)
                self._group_of[key] = synonyms.canonical(key) or key.lower()

    def group_of(self, class_name: str) -> str:
        """Semantic group of a class name (the name itself when unmapped)."""
        key = str(class_name)
        return self._group_of.get(key, key.lower())

    # -- construction ------------------------------------------------------
    def __bool__(self) -> bool:
        return self.names is not None and len(self.names) > 0

    def __len__(self) -> int:
        return 0 if self.names is None else int(len(self.names))

    @property
    def n_classes(self) -> int:
        return int(len(set(self.names.tolist()))) if len(self) else 0

    @property
    def has_glyphs(self) -> bool:
        return self.glyphs is not None and len(self.glyphs) == len(self)

    @classmethod
    def load(cls, path: str, **kwargs) -> "IconDatabase":
        """Load an ``.npz`` glyph database or a legacy ``{hash: class}`` JSON.

        Never raises: an unreadable/missing file yields an empty database, and
        the pipeline then behaves as if the stage were disabled.
        """
        empty = cls(np.array([]), np.zeros((0, 0), dtype=np.uint8), **kwargs)
        if not path or not os.path.isfile(path):
            if path:
                LOGGER.warning("Icon DB '%s' not found — blue stage disabled.", path)
            return empty
        try:
            if path.endswith(".npz"):
                return cls._load_npz(path, **kwargs)
            return cls._load_legacy_json(path, **kwargs)
        except Exception as exc:
            LOGGER.warning("Failed to load icon DB %s: %s", path, exc)
            return empty

    @classmethod
    def _load_npz(cls, path: str, **kwargs) -> "IconDatabase":
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["meta"])) if "meta" in data else {}
            db = cls(
                names=data["names"],
                phash=data["phash"],
                dhash=data["dhash"] if "dhash" in data else None,
                ahash=data["ahash"] if "ahash" in data else None,
                glyphs=data["glyphs"] if "glyphs" in data else None,
                files=data["files"] if "files" in data else None,
                meta=meta,
                **kwargs,
            )
        LOGGER.info("Loaded icon DB: %d icon(s), %d class(es), glyphs=%s (%s).",
                    len(db), db.n_classes, db.has_glyphs, path)
        return db

    @classmethod
    def _load_legacy_json(cls, path: str, **kwargs) -> "IconDatabase":
        """Legacy ``{phash_hex: classname}`` — hashes only, no glyph re-scoring."""
        with open(path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        names, packed = [], []
        for hex_str, class_name in raw.items():
            try:
                bits = imagehash.hex_to_hash(hex_str).hash
            except Exception:
                continue
            names.append(class_name)
            packed.append(np.packbits(np.asarray(bits, dtype=bool).ravel()))
        if not packed:
            raise ValueError("no usable hashes in the JSON")
        n_bits = int(np.asarray(imagehash.hex_to_hash(
            next(iter(raw))).hash).size)
        db = cls(names=np.array(names), phash=np.stack(packed),
                 meta={"hash_bits": n_bits,
                       "hash_size": int(round(n_bits ** 0.5))},
                 **kwargs)
        LOGGER.info("Loaded legacy icon DB: %d hash(es), %d class(es) (%s). "
                    "No glyphs stored — run build_icon_db.py for glyph "
                    "re-scoring.", len(db), db.n_classes, path)
        return db

    # -- hashing a query ---------------------------------------------------
    def hash_query(self, glyph: np.ndarray) -> Tuple[np.ndarray, ...]:
        """(pHash, dHash, aHash) of a query glyph, bit-packed like the DB."""
        pil = Image.fromarray(glyph)
        def pack(h):
            return np.packbits(np.asarray(h.hash, dtype=bool).ravel())
        return (pack(imagehash.phash(pil, hash_size=self.hash_size)),
                pack(imagehash.dhash(pil, hash_size=self.hash_size)),
                pack(imagehash.average_hash(pil, hash_size=self.hash_size)))

    def _hamming(self, table: np.ndarray, query: np.ndarray) -> np.ndarray:
        return _POPCOUNT[np.bitwise_xor(table, query[None, :])].sum(axis=1)

    def _hash_similarity(
        self, qph: np.ndarray, qdh: Optional[np.ndarray],
        qah: Optional[np.ndarray], mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Weighted multi-hash similarity in [0,1] for every (unmasked) entry.

        Returns ``(similarity, phash_hamming, indices)`` where ``indices`` maps
        back to database rows (identity unless ``mask`` hid some).
        """
        indices = np.nonzero(mask)[0] if mask is not None else np.arange(len(self))
        ph_dist = self._hamming(self.phash[indices], qph)
        total = HASH_WEIGHTS["phash"] * (ph_dist / self.n_bits)
        weight = HASH_WEIGHTS["phash"]
        if self.dhash is not None and qdh is not None:
            total += HASH_WEIGHTS["dhash"] * (
                self._hamming(self.dhash[indices], qdh) / self.n_bits)
            weight += HASH_WEIGHTS["dhash"]
        if self.ahash is not None and qah is not None:
            total += HASH_WEIGHTS["ahash"] * (
                self._hamming(self.ahash[indices], qah) / self.n_bits)
            weight += HASH_WEIGHTS["ahash"]
        similarity = 1.0 - (total / weight)
        return similarity, ph_dist, indices

    # -- per-class aggregation --------------------------------------------
    def _vote(self, names: np.ndarray, scores: np.ndarray) -> List[Dict[str, Any]]:
        """Aggregate per-exemplar scores into one score per group, best first.

        A group scores the mean of its best ``votes_per_class`` exemplars: two
        exemplars agreeing is much stronger evidence than one outlier, while
        still not punishing a group with only one exemplar in the shortlist.
        The row's ``name`` is the highest-scoring real class name inside the
        group, so the caller still gets a concrete label to print.
        """
        buckets: Dict[str, List[Tuple[float, str]]] = {}
        for name, score in zip(names, scores):
            buckets.setdefault(self.group_of(name), []).append(
                (float(score), str(name)))
        rows = []
        for group, values in buckets.items():
            values.sort(reverse=True)
            top = [v[0] for v in values[:max(1, self.votes_per_class)]]
            rows.append({"name": values[0][1], "group": group,
                         "score": float(np.mean(top)), "best": values[0][0],
                         "votes": len(values)})
        rows.sort(key=lambda r: -r["score"])
        return rows

    def _decide(self, rows: List[Dict[str, Any]], floor: float, margin: float,
                ) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]],
                           Optional[str]]:
        """Apply the floor + runner-up margin. Returns (winner, runner_up, why_not)."""
        if not rows:
            return None, None, "no candidates"
        best = rows[0]
        second = rows[1] if len(rows) > 1 else None
        second_score = second["score"] if second else 0.0
        gap = best["score"] - second_score
        if best["score"] < floor:
            return (None, second,
                    f"best {best['name']!r} score {best['score']:.3f} < floor {floor}")
        if second is not None and gap < margin:
            return (None, second,
                    f"{best['name']!r} {best['score']:.3f} vs "
                    f"{second['name']!r} {second_score:.3f} — margin "
                    f"{gap:.3f} < {margin}")
        return best, second, None

    def _finish(self, out: IconDbMatch, rows: List[Dict[str, Any]],
                floor: float, margin: float) -> IconDbMatch:
        """Run the decision over voted rows and fill in ``out``."""
        winner, runner, rejected = self._decide(rows, floor, margin)
        top = rows[0] if rows else None
        out.rows = rows[:5]
        out.rejected = rejected
        out.runner_up = runner["name"] if runner else None
        out.runner_up_score = runner["score"] if runner else 0.0
        out.score = top["score"] if top else 0.0
        out.margin = out.score - out.runner_up_score
        if winner is not None:
            out.name = winner["name"]
            out.group = winner["group"]
            out.votes = winner["votes"]
        return out

    # -- lookup: hashes only ----------------------------------------------
    def match_hashes(
        self, qph: np.ndarray, qdh: Optional[np.ndarray] = None,
        qah: Optional[np.ndarray] = None, mask: Optional[np.ndarray] = None,
    ) -> IconDbMatch:
        """Identify from hashes alone (used when the DB has no glyphs)."""
        if not self:
            return IconDbMatch()
        sim, ph_dist, indices = self._hash_similarity(qph, qdh, qah, mask)
        near = int(sim.argmax())
        out = IconDbMatch(
            nearest_name=str(self.names[indices[near]]),
            nearest_hamming=int(ph_dist[near]),
            method="hash",
        )
        # A near-exact pHash is decisive on its own — keep the cheap shortcut.
        if out.nearest_hamming <= self.hash_shortcut_hamming:
            out.name = out.nearest_name
            out.group = self.group_of(out.nearest_name)
            out.score = float(sim[near])
            out.votes = 1
            return out

        k = min(self.prefilter_k, len(sim))
        top = np.argpartition(-sim, k - 1)[:k]
        rows = self._vote(self.names[indices[top]], sim[top])
        return self._finish(out, rows, self.hash_min_similarity, self.hash_margin)

    # -- lookup: hashes + glyph re-scoring --------------------------------
    def match(
        self, query_sig, matcher, mask: Optional[np.ndarray] = None,
    ) -> IconDbMatch:
        """Full lookup: hash prefilter, glyph re-score, class vote, decide.

        ``query_sig`` is a ``VisualSignature`` (needs ``glyph``; ORB optional)
        and ``matcher`` the pipeline's ``SignatureMatcher`` — reusing it keeps
        this stage's notion of similarity identical to the legend stage's.
        Falls back to :meth:`match_hashes` when the DB holds no glyphs.
        """
        if not self or query_sig is None or query_sig.glyph is None:
            return IconDbMatch()

        qph, qdh, qah = self.hash_query(query_sig.glyph)
        if not self.has_glyphs:
            return self.match_hashes(qph, qdh, qah, mask)

        sim, ph_dist, indices = self._hash_similarity(qph, qdh, qah, mask)
        near = int(sim.argmax())
        out = IconDbMatch(
            nearest_name=str(self.names[indices[near]]),
            nearest_hamming=int(ph_dist[near]),
            method="glyph",
        )

        k = min(self.prefilter_k, len(sim))
        order = np.argpartition(-sim, k - 1)[:k]
        shortlist = indices[order]

        # Final score: mostly the glyph match, with a little hash similarity —
        # the hash term is a cheap tie-break that nudges apart two glyphs the
        # template correlation rates equally.
        scored: List[float] = []
        for row, hash_sim in zip(shortlist, sim[order]):
            score, _ = matcher.score(query_sig, self._signature(int(row)))
            scored.append((1.0 - self.hash_weight) * score
                          + self.hash_weight * float(hash_sim))

        rows = self._vote(self.names[shortlist], np.asarray(scored))
        return self._finish(out, rows, self.min_score, self.margin)

    # -- helpers -----------------------------------------------------------
    def _signature(self, row: int):
        """VisualSignature for a stored glyph (ORB computed once, then cached)."""
        cached = self._sig_cache.get(row)
        if cached is not None:
            return cached
        from .containers import VisualSignature      # local: avoid import cycle
        from .deps import cv2
        sig = VisualSignature()
        sig.glyph = self.glyphs[row]
        try:
            orb = getattr(self, "_orb", None)
            if orb is None:
                orb = cv2.ORB_create(nfeatures=300, fastThreshold=5)
                self._orb = orb
            sig.keypoints, sig.orb_descriptors = orb.detectAndCompute(sig.glyph, None)
        except Exception:
            pass
        self._sig_cache[row] = sig
        return sig
