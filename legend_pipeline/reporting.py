"""Per-crop Hamming-distance report (.txt)."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence


def write_hamming_info(
    txt_path: str,
    *,
    title: str,
    bbox: Sequence[int],
    confidence: float,
    phash_hex: Optional[str],
    hash_size: int,
    rows: List[Dict[str, Any]],
    footer_lines: Sequence[str] = (),
) -> None:
    """Write a human-readable .txt describing one crop's Hamming distances.

    ``rows`` is a ranked list (nearest first) of dicts with keys
    ``name``, ``hamming``, ``phash_similarity`` and ``score`` — i.e. the output
    of :meth:`SignatureMatcher.rank`.  ``footer_lines`` carries the final
    decision (best match / rename verdict) appended verbatim at the bottom.
    """
    n_bits = hash_size * hash_size
    lines: List[str] = []
    lines.append(title)
    lines.append("=" * len(title))
    lines.append(f"Bounding box (x1,y1,x2,y2): {list(bbox)}")
    lines.append(f"Detection confidence      : {confidence:.4f}")
    lines.append(f"pHash (hex)               : {phash_hex}")
    lines.append(
        f"Hash size                 : {hash_size}x{hash_size} = {n_bits} bits "
        f"(max possible Hamming distance = {n_bits})"
    )
    lines.append("")
    lines.append("Hamming distance to each legend icon (nearest first):")
    header = f"  {'legend name':<34}{'hamming':>9}{'hash_sim':>10}{'weighted':>10}"
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    for r in rows:
        h = r.get("hamming")
        hs = r.get("phash_similarity")
        sc = r.get("score", 0.0)
        h_str = str(h) if h is not None else "n/a"
        hs_str = f"{hs:.3f}" if hs is not None else "n/a"
        name = str(r.get("name", ""))[:34]
        lines.append(f"  {name:<34}{h_str:>9}{hs_str:>10}{sc:>10.3f}")
    if footer_lines:
        lines.append("")
        lines.extend(footer_lines)
    with open(txt_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
