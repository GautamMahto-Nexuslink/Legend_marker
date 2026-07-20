"""Step 4 & 6: visual signatures (glyph + ORB + pHash) and weighted matching."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .config import PipelineConfig
from .containers import VisualSignature
from .deps import LOGGER, Image, _require, cv2, imagehash


class SignatureBuilder:
    """Turns an icon crop into a background-free, canonically-sized *glyph*.

    Rendered map symbols are consistent up to scale/compression, but their
    surrounding pixels are not (flat badge in the legend vs. terrain on the
    map).  We therefore estimate the background from the crop border, keep only
    the pixels that differ from it (the symbol), tight-crop to that, and resize
    to a fixed square.  Every downstream feature is computed on this glyph.
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        _require(cv2, "opencv-python", "pip install opencv-python")
        # ORB on a fixed glyph is deterministic (same pixels -> same output).
        self._orb = cv2.ORB_create(nfeatures=300, fastThreshold=5)

    # -- Foreground segmentation -----------------------------------------
    def _segment_glyph(self, crop: np.ndarray) -> np.ndarray:
        """Return the tight, background-free grayscale glyph at canonical size."""
        size = self.config.glyph_size
        if crop is None or crop.size == 0:
            return np.zeros((size, size), dtype=np.uint8)

        bgr = crop if crop.ndim == 3 else cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
        h, w = bgr.shape[:2]

        # 1) Estimate the background colour from a thin border ring (robust
        #    median) — the frame around a symbol is almost always background.
        b = max(1, min(h, w) // 10)
        border = np.concatenate([
            bgr[:b, :, :].reshape(-1, 3), bgr[-b:, :, :].reshape(-1, 3),
            bgr[:, :b, :].reshape(-1, 3), bgr[:, -b:, :].reshape(-1, 3),
        ], axis=0)
        bg = np.median(border, axis=0)

        # 2) Foreground = pixels whose colour distance from bg exceeds tolerance.
        dist = np.linalg.norm(bgr.astype(np.float32) - bg[None, None, :], axis=2)
        mask = (dist > self.config.seg_bg_tolerance).astype(np.uint8)

        # 3) Clean tiny speckles with a morphological opening.
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1
        )

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        ys, xs = np.where(mask > 0)
        # If segmentation found too little, fall back to the whole crop so we
        # never return an empty glyph (better a noisy glyph than nothing).
        if xs.size < self.config.seg_min_fg_ratio * h * w:
            glyph_region = gray
        else:
            x1, x2 = xs.min(), xs.max() + 1
            y1, y2 = ys.min(), ys.max() + 1
            glyph_region = gray[y1:y2, x1:x2]
            # Blank the background inside the box to neutral grey so correlation
            # keys on the symbol, not leftover background texture.
            region_mask = mask[y1:y2, x1:x2]
            glyph_region = np.where(region_mask > 0, glyph_region, 127).astype(np.uint8)

        return self._pad_resize(glyph_region, size)

    @staticmethod
    def _pad_resize(gray: np.ndarray, size: int) -> np.ndarray:
        """Pad to square (neutral grey) then resize to size×size, keeping aspect."""
        if gray.size == 0:
            return np.zeros((size, size), dtype=np.uint8)
        gh, gw = gray.shape[:2]
        side = max(gh, gw)
        canvas = np.full((side, side), 127, dtype=np.uint8)
        y0, x0 = (side - gh) // 2, (side - gw) // 2
        canvas[y0:y0 + gh, x0:x0 + gw] = gray
        interp = cv2.INTER_AREA if side > size else cv2.INTER_CUBIC
        return cv2.resize(canvas, (size, size), interpolation=interp)

    # -- pHash (reporting only) ------------------------------------------
    def _compute_phash(self, glyph_gray: np.ndarray) -> Tuple[Any, str]:
        _require(imagehash, "imagehash", "pip install imagehash")
        _require(Image, "Pillow", "pip install pillow")
        pil = Image.fromarray(glyph_gray)
        algo = {
            "phash": imagehash.phash,
            "dhash": imagehash.dhash,
            "ahash": imagehash.average_hash,
            "whash": imagehash.whash,
        }.get(self.config.hash_algorithm, imagehash.phash)
        h = algo(pil, hash_size=self.config.hash_size)
        return h, str(h)

    # -- Public API -------------------------------------------------------
    def build(self, crop: np.ndarray) -> VisualSignature:
        """Compute the glyph and its features for a crop, tolerating failures."""
        sig = VisualSignature()

        # Aspect ratio from the ORIGINAL crop (segmentation removes it).
        if crop is not None and crop.size > 0:
            h, w = crop.shape[:2]
            sig.aspect_ratio = float(w) / float(h) if h else 1.0

        sig.glyph = self._segment_glyph(crop)

        try:
            kp, des = self._orb.detectAndCompute(sig.glyph, None)
            sig.keypoints, sig.orb_descriptors = kp, des
        except Exception as exc:
            LOGGER.warning("ORB failed: %s", exc)

        try:
            sig.phash, sig.phash_hex = self._compute_phash(sig.glyph)
        except Exception as exc:
            LOGGER.warning("pHash failed: %s", exc)

        return sig


class SignatureMatcher:
    """Scores glyph similarity with multi-scale template correlation + ORB.

    Both sub-scores live in [0, 1] and are computed on the background-free
    glyph.  The decision score is their (renormalised) weighted mean, so a
    missing sub-score (e.g. ORB found no keypoints) does not drag the total.
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        # knnMatch needs crossCheck OFF so we can apply Lowe's ratio test.
        self._bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    # -- Template correlation (scale-swept, alignment-tolerant) ----------
    def _score_template(self, a: VisualSignature, b: VisualSignature) -> Optional[float]:
        """Best normalised cross-correlation of glyph *b* against glyph *a*.

        The query glyph is padded so the (rescaled) legend glyph can slide a few
        pixels, absorbing minor misalignment; several scales absorb size drift.
        TM_CCOEFF_NORMED is mean-subtracted, so uniform brightness shifts don't
        matter — only the symbol's structure does.
        """
        if a.glyph is None or b.glyph is None:
            return None
        pad = self.config.template_search_pad
        # Neutral-grey border keeps the correlation focused on the glyph.
        search = cv2.copyMakeBorder(a.glyph, pad, pad, pad, pad,
                                    cv2.BORDER_CONSTANT, value=127)
        sh, sw = search.shape[:2]

        best = -1.0
        for s in self.config.template_scales:
            side = max(4, int(round(self.config.glyph_size * s)))
            if side > sh or side > sw:      # template must fit inside search.
                continue
            templ = cv2.resize(b.glyph, (side, side),
                               interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
            try:
                res = cv2.matchTemplate(search, templ, cv2.TM_CCOEFF_NORMED)
            except cv2.error:
                continue
            best = max(best, float(res.max()))

        if best < 0:
            return 0.0
        return max(0.0, min(1.0, best))     # clamp; negatives => no similarity.

    # -- ORB ratio-test inlier fraction ----------------------------------
    def _score_orb(self, a: VisualSignature, b: VisualSignature) -> Optional[float]:
        da, db = a.orb_descriptors, b.orb_descriptors
        if da is None or db is None or len(da) < 2 or len(db) < 2:
            return None
        try:
            knn = self._bf.knnMatch(da, db, k=2)
        except cv2.error:
            return None
        good = 0
        for pair in knn:
            if len(pair) < 2:
                continue
            m, n = pair
            # Lowe's ratio test keeps only confidently-unique matches.
            if m.distance < self.config.orb_ratio * n.distance:
                good += 1
        denom = min(len(da), len(db))
        return good / float(denom) if denom else 0.0

    def _score_phash(self, a: VisualSignature, b: VisualSignature) -> Optional[float]:
        """pHash similarity — reporting only, not part of the decision."""
        if a.phash is None or b.phash is None:
            return None
        distance = a.phash - b.phash
        n_bits = self.config.hash_size * self.config.hash_size
        return 1.0 - min(distance, n_bits) / float(n_bits)

    # -- Combined ---------------------------------------------------------
    def score(self, a: VisualSignature, b: VisualSignature) -> Tuple[float, Dict[str, float]]:
        """Renormalised weighted mean of the template + ORB sub-scores."""
        cfg = self.config
        parts: List[Tuple[float, Optional[float]]] = [
            (cfg.w_template, self._score_template(a, b)),
            (cfg.w_orb, self._score_orb(a, b)),
        ]
        names = ["template", "orb"]

        total, weight_sum = 0.0, 0.0
        breakdown: Dict[str, float] = {}
        for (weight, value), name in zip(parts, names):
            if weight <= 0 or value is None:
                continue
            total += weight * value
            weight_sum += weight
            breakdown[name] = round(value, 4)

        combined = total / weight_sum if weight_sum > 0 else 0.0
        return combined, breakdown

    @staticmethod
    def phash_hamming(a: VisualSignature, b: VisualSignature) -> Optional[int]:
        """Raw Hamming distance between two pHashes (None if either missing)."""
        if a.phash is None or b.phash is None:
            return None
        return int(a.phash - b.phash)

    def rank(
        self,
        query: VisualSignature,
        legend_sigs: List[Tuple[str, VisualSignature]],
    ) -> List[Dict[str, Any]]:
        """Score the query against every legend entry, sorted best-first.

        Each row carries both the raw pHash Hamming distance and the combined
        weighted similarity, so callers can report/inspect either.
        """
        rows: List[Dict[str, Any]] = []
        for name, sig in legend_sigs:
            score, breakdown = self.score(query, sig)
            hamming = self.phash_hamming(query, sig)
            rows.append(
                {
                    "name": name,
                    "hamming": hamming,
                    "phash_hex": sig.phash_hex,
                    "phash_similarity": self._score_phash(query, sig),
                    "score": score,
                    "breakdown": breakdown,
                }
            )
        # Sort by weighted score desc; break ties by smaller Hamming distance.
        rows.sort(
            key=lambda r: (-r["score"],
                           r["hamming"] if r["hamming"] is not None else 10**9)
        )
        return rows

    def best_match(
        self,
        query: VisualSignature,
        legend_sigs: List[Tuple[str, VisualSignature]],
    ) -> Tuple[Optional[str], float, Dict[str, float]]:
        """Return (best_name, best_score, breakdown) over all legend entries."""
        rows = self.rank(query, legend_sigs)
        if not rows:
            return None, -1.0, {}
        top = rows[0]
        return top["name"], top["score"], top["breakdown"]
