"""Per-crop report (.txt): Hamming table + what every stage called the icon."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from .visualization import METHOD_COLORS, color_for_method


def stage_report_lines(
    *,
    original_class: str,
    final_class: str,
    renamed: bool,
    match_method: Optional[str],
    # pHash DB stage
    db_enabled: bool,
    db_class: Optional[str],
    db_nearest_name: Optional[str],
    db_nearest_dist: Optional[int],
    db_max_hamming: int,
    # legend / OCR stage
    legend_name: Optional[str],
    legend_score: float,
    legend_margin: float,
    passes_floor: bool,
    passes_margin: bool,
    score_threshold: float,
    margin_threshold: float,
    # semantic stage
    syn_enabled: bool,
    relation: Optional[str],
    semantic_legend: Optional[str],
    semantic_score: Optional[float],
    canonical: Optional[str],
    naming: str,
    rescued: bool,
    n_legend_entries: int,
) -> List[str]:
    """The per-stage summary block: what each stage would have named this icon.

    Every stage is listed with the colour it draws on ``map_annotated.png``, the
    class name it produced (or why it produced none), and whether it won.  So a
    marina icon reads:

        orange  original detector class : Parking
        blue    pHash DB (icon hashes)  : Marina        HIT hamming=3 <= 10
        green   legend / OCR text       : Marina        score=0.71 PASS
        magenta hash class in legend    : Marina        same as legend 'Marina'
        FINAL   -> Marina               colour=magenta  method=phash_db+same
    """
    def stage(colour: str, label: str, value: str, note: str = "",
              won: bool = False) -> str:
        mark = " <== FINAL" if won else ""
        return f"  {colour:<8}{label:<26}: {value:<30}{note}{mark}"

    winning = color_for_method(match_method, renamed)[0]
    lines: List[str] = ["Stage results (colour on map_annotated.png -> class name)"]

    # -- orange: the raw detector class, used when nothing else wins ----------
    lines.append(stage("orange", "original detector class", original_class,
                       "kept when no stage below wins",
                       won=not renamed))

    # -- blue: known-icon pHash database --------------------------------------
    if not db_enabled:
        lines.append(stage("blue", "pHash DB (icon hashes)", "-",
                           "disabled (no pHash DB configured)"))
    elif db_class is not None:
        lines.append(stage("blue", "pHash DB (icon hashes)", db_class,
                           f"HIT hamming={db_nearest_dist} <= {db_max_hamming}",
                           won=match_method == "phash_db"))
    else:
        nearest = db_nearest_name or "-"
        lines.append(stage("blue", "pHash DB (icon hashes)", "-",
                           f"miss (nearest '{nearest}' "
                           f"hamming={db_nearest_dist} > {db_max_hamming})"))

    # -- green: legend / OCR text ---------------------------------------------
    if legend_name is None:
        lines.append(stage("green", "legend / OCR text", "-",
                           "no legend candidates"))
    else:
        gate = ("PASS" if passes_floor and passes_margin else
                f"FAIL ({'floor' if not passes_floor else 'margin'})")
        lines.append(stage(
            "green", "legend / OCR text", legend_name,
            f"score={legend_score:.3f} (>= {score_threshold}) "
            f"margin={legend_margin:.3f} (>= {margin_threshold}) {gate}",
            won=match_method in ("legend", "legend+synonym")))

    # -- magenta: hash class looked up in this map's legend -------------------
    if not syn_enabled:
        lines.append(stage("magenta", "hash class in legend", "-",
                           "disabled (no synonym map)"))
    elif relation is None:
        lines.append(stage("magenta", "hash class in legend", "-",
                           f"no same/nearby label among {n_legend_entries} "
                           f"legend entrie(s)"))
    else:
        note = (f"{relation} as legend '{semantic_legend}'"
                f" (group '{canonical}'"
                f"{f', score={semantic_score:.3f}' if semantic_score is not None else ''})"
                f"{' RESCUED below-gate' if rescued else ''}")
        lines.append(stage("magenta", "hash class in legend", semantic_legend or "-",
                           note,
                           won=match_method in ("phash_db+same", "phash_db+related",
                                                "synonym_agree")))

    lines.append("  " + "-" * 96)
    lines.append(stage("FINAL", "class saved to results", str(final_class),
                       f"colour={winning}  "
                       f"method={match_method or 'none (kept original)'}"))
    lines.append(f"  {'':8}{'canonical group':<26}: {canonical or '-':<30}"
                 f"naming policy={naming}")
    return lines


#: Legend for the colours used above / on the annotated map.
COLOR_LEGEND = "  ".join(
    f"{name}={method or 'kept original'}"
    for method, (name, _) in METHOD_COLORS.items()
)


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
