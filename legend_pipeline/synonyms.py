"""Semantic class-name reconciliation (``icon_class_synonyms.json``).

The pipeline gets a class name from two independent sources that use different
words for the same thing:

    pHash DB  -> "Observation Area"     (curated, cross-map, visual identity)
    legend OCR-> "Overlook"             (this map's own wording)

String comparison says they disagree; semantically they are one concept.  This
module loads the curated grouping (canonical name -> synonyms / alternate
spellings / search terms, built by ``build_class_synonyms.py``) and answers two
questions:

    canonical("Overlook")                      -> "observation area"
    same_concept("Observation Area", "overlook") -> True

so the pipeline can treat the two sources as *agreeing* and pick the label it
was told to prefer, instead of silently dropping one of them.

Matching is deliberately tolerant of what OCR actually returns — trailing
punctuation, stray case, doubled spaces, plurals ("overlooks"), and small
character errors ("0verlook") — via, in order: exact normalized lookup,
singular/plural folding, all-tokens containment, then a difflib ratio gate.
"""
from __future__ import annotations

import difflib
import json
import os
import re
from typing import Dict, List, Optional, Tuple

from .deps import LOGGER

# Repo root (…/legend_marker), i.e. the parent of this package.
_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_DIR = os.path.dirname(_PKG_DIR)

#: Default location of the grouping produced by ``build_class_synonyms.py``.
DEFAULT_SYNONYMS_PATH = os.path.join(_REPO_DIR, "icon_class_synonyms.json")
#: Neighbour graph (canonical -> nearby canonicals), same builder.  Looked up
#: beside the synonyms file, so pointing --synonyms elsewhere moves both.
RELATED_FILENAME = "icon_class_related.json"

_PUNCT_RE = re.compile(r"[^a-z0-9&\s:-]+")
_WS_RE = re.compile(r"\s+")


def normalize(name: str) -> str:
    """Fold a class name / OCR string to its comparison form.

    ``'Contact_Station'`` -> ``'contact station'``,
    ``'Campground Hike & Bike'`` -> ``'campground hike and bike'``,
    ``'Overlook. '`` -> ``'overlook'``.  Kept byte-compatible with the
    ``normalize()`` in ``build_class_synonyms.py`` for shared terms.
    """
    s = (name or "").strip().lower()
    s = s.replace("&", " and ")
    s = re.sub(r"[_/]+", " ", s)
    s = _PUNCT_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s)
    return s.strip(" -:")


def _singularize(token: str) -> str:
    """Crude plural fold — enough for 'cabins', 'beaches', 'restrooms'."""
    if len(token) > 3 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith("es") and token[-3] in "shxz":
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _stem_key(text: str) -> str:
    return " ".join(_singularize(t) for t in normalize(text).split())


#: Digits OCR commonly substitutes for letters in short labels.
_OCR_DIGITS = str.maketrans({"0": "o", "1": "l", "3": "e", "4": "a",
                             "5": "s", "6": "g", "8": "b", "9": "g"})


def _ocr_fold(text: str) -> str:
    """'0verlook' -> 'overlook' — only used as a fallback lookup key."""
    return text.translate(_OCR_DIGITS)


class SynonymMap:
    """Canonical-name lookup over the curated synonym groups.

    Construct with :meth:`load` (path -> cached instance) or directly from a
    ``{canonical: [terms...]}`` mapping.  An empty map is valid and makes every
    lookup return ``None``, so the pipeline degrades to its previous behaviour
    when the JSON is missing.
    """

    _cache: Dict[Tuple[str, float], "SynonymMap"] = {}

    def __init__(self, groups: Dict[str, List[str]], fuzzy_cutoff: float = 0.88,
                 related: Optional[Dict[str, List[str]]] = None) -> None:
        self.groups: Dict[str, List[str]] = groups or {}
        self.fuzzy_cutoff = fuzzy_cutoff
        # canonical -> neighbouring canonicals ("biking" ~ "motorbike trail").
        self.related_graph: Dict[str, frozenset] = {
            k: frozenset(v) for k, v in (related or {}).items()
        }
        self._exact: Dict[str, str] = {}
        self._stemmed: Dict[str, str] = {}
        self._token_sets: List[Tuple[frozenset, str]] = []

        for canonical in self.groups:
            self._exact.setdefault(normalize(canonical), canonical)
        for canonical, terms in self.groups.items():
            for term in terms:
                key = normalize(term)
                if key:
                    self._exact.setdefault(key, canonical)
        for key, canonical in self._exact.items():
            self._stemmed.setdefault(_stem_key(key), canonical)
            self._token_sets.append((frozenset(_stem_key(key).split()), canonical))
        # Longest term first so "picnic shelter" wins over "shelter".
        self._token_sets.sort(key=lambda kv: -len(kv[0]))
        self._keys = list(self._exact)

    # -- construction ------------------------------------------------------
    @classmethod
    def load(cls, path: str = DEFAULT_SYNONYMS_PATH,
             fuzzy_cutoff: float = 0.88,
             related_path: Optional[str] = None) -> "SynonymMap":
        """Load (and cache) the grouping JSON + its neighbour graph; never raises.

        ``related_path`` defaults to ``icon_class_related.json`` sitting next to
        ``path``; a missing neighbour file just disables the related-match
        second chance.
        """
        cache_key = (os.path.abspath(path) if path else "", fuzzy_cutoff)
        cached = cls._cache.get(cache_key)
        if cached is not None:
            return cached

        groups: Dict[str, List[str]] = {}
        if path and os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    raw = json.load(fh)
                groups = {str(k): [str(t) for t in v] for k, v in raw.items()}
                LOGGER.info("Loaded %d semantic class group(s) from %s.",
                            len(groups), path)
            except Exception as exc:
                LOGGER.warning("Failed to load synonym map %s: %s", path, exc)
        elif path:
            LOGGER.warning("Synonym map '%s' not found — semantic class "
                           "reconciliation disabled.", path)

        related: Dict[str, List[str]] = {}
        if groups:
            rel_path = related_path or os.path.join(
                os.path.dirname(os.path.abspath(path)), RELATED_FILENAME)
            if os.path.isfile(rel_path):
                try:
                    with open(rel_path, "r", encoding="utf-8") as fh:
                        raw_rel = json.load(fh)
                    related = {str(k): [str(t) for t in v]
                               for k, v in raw_rel.items() if k in groups}
                    LOGGER.info("Loaded neighbour graph (%d edge(s)) from %s.",
                                sum(len(v) for v in related.values()) // 2, rel_path)
                except Exception as exc:
                    LOGGER.warning("Failed to load related map %s: %s", rel_path, exc)
            else:
                LOGGER.warning("Neighbour graph '%s' not found — related-class "
                               "renaming disabled.", rel_path)

        inst = cls(groups, fuzzy_cutoff, related)
        cls._cache[cache_key] = inst
        return inst

    def __bool__(self) -> bool:
        return bool(self.groups)

    def __len__(self) -> int:
        return len(self.groups)

    # -- lookup ------------------------------------------------------------
    def canonical(self, name: Optional[str], fuzzy: bool = True) -> Optional[str]:
        """Return the canonical class name for ``name``, or ``None``.

        Tries, in order: exact normalized term, singular/plural fold, the
        longest group term whose tokens are all present in ``name`` (so
        "Scenic Overlook Trail" still resolves), then a difflib similarity
        gate for OCR character noise.
        """
        if not name or not self.groups:
            return None

        key = normalize(name)
        if not key:
            return None
        hit = self._exact.get(key)
        if hit:
            return hit

        stem = _stem_key(key)
        hit = self._stemmed.get(stem)
        if hit:
            return hit

        folded = _stem_key(_ocr_fold(key))
        if folded != stem:
            hit = self._exact.get(_ocr_fold(key)) or self._stemmed.get(folded)
            if hit:
                return hit

        # Longest fully-contained term wins ("Scenic Overlook Trail" ->
        # "scenic overlook").  A tie between different groups is ambiguous
        # ("Overlook Trail" -> observation area? hiking?) — refuse to guess.
        tokens = set(stem.split())
        if tokens:
            best_len, winners = 0, set()
            for term_tokens, canonical in self._token_sets:
                if len(term_tokens) < best_len:
                    break                      # sorted longest-first
                if term_tokens and term_tokens <= tokens:
                    if len(term_tokens) > best_len:
                        best_len, winners = len(term_tokens), {canonical}
                    else:
                        winners.add(canonical)
            if len(winners) == 1:
                return winners.pop()
            if winners:
                LOGGER.debug("Ambiguous class '%s' -> %s; no group assigned.",
                             name, sorted(winners))
                return None

        if fuzzy and self.fuzzy_cutoff < 1.0:
            close = difflib.get_close_matches(
                key, self._keys, n=1, cutoff=self.fuzzy_cutoff)
            if close:
                return self._exact[close[0]]
        return None

    def resolve(self, name: Optional[str], fuzzy: bool = True) -> Optional[str]:
        """:meth:`canonical` but falls back to the normalized input itself."""
        return self.canonical(name, fuzzy) or (normalize(name) or None)

    def same_concept(self, a: Optional[str], b: Optional[str],
                     fuzzy: bool = True) -> bool:
        """True when both names resolve to the same canonical group.

        Two names that are *identical* count as the same concept even when
        neither is in the map, so an unknown class still agrees with itself.
        """
        if not a or not b:
            return False
        ca, cb = self.canonical(a, fuzzy), self.canonical(b, fuzzy)
        if ca and cb:
            return ca == cb
        return normalize(a) == normalize(b)

    def related(self, name: Optional[str]) -> frozenset:
        """Canonical groups neighbouring ``name``'s group (never the group itself)."""
        canonical = self.canonical(name)
        return self.related_graph.get(canonical, frozenset()) if canonical else frozenset()

    def is_related(self, a: Optional[str], b: Optional[str]) -> bool:
        """True when ``a`` and ``b`` are *neighbouring* (not identical) concepts."""
        ca, cb = self.canonical(a), self.canonical(b)
        if not ca or not cb or ca == cb:
            return False
        return cb in self.related_graph.get(ca, frozenset())

    def relation(self, a: Optional[str], b: Optional[str]) -> Optional[str]:
        """``"same"`` | ``"related"`` | ``None`` — how two class names compare."""
        if self.same_concept(a, b):
            return "same"
        if self.is_related(a, b):
            return "related"
        return None

    def terms(self, name: Optional[str]) -> List[str]:
        """All synonyms/search terms of the group ``name`` belongs to."""
        canonical = self.canonical(name)
        return list(self.groups.get(canonical, [])) if canonical else []

    def pick_label(self, policy: str, legend_name: Optional[str],
                   db_name: Optional[str]) -> Optional[str]:
        """Choose the output label for two agreeing sources.

        ``policy`` is one of ``"legend"`` (the map's own wording),
        ``"canonical"`` (normalise everything to the group key) or ``"db"``
        (keep the curated pHash-DB class).  Falls back to whichever name
        exists when the preferred one is missing.
        """
        legend_name = (legend_name or "").strip() or None   # OCR keeps stray ws
        db_name = (db_name or "").strip() or None
        if policy == "canonical":
            return (self.canonical(db_name) or self.canonical(legend_name)
                    or db_name or legend_name)
        if policy == "db":
            return db_name or legend_name
        return legend_name or db_name          # "legend" (default)
