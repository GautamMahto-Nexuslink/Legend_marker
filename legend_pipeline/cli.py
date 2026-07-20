"""Command-line interface — argparse, config translation, validation, main()."""
from __future__ import annotations

import argparse
import json
import os
from typing import List, Optional, Sequence

from .config import PipelineConfig
from .deps import LOGGER
from .pipeline import LegendMarkerPipeline
from .utils import setup_logging


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Replace generic map-icon classes with real names from the legend.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Inputs
    p.add_argument("--map", dest="map_path", required=True,
                   help="Path to the original full map image.")
    p.add_argument("--legend", dest="legend_path", required=True,
                   help="Path to the cropped legend image.")

    # Roboflow (env-var fallbacks so keys need not be on the command line).
    p.add_argument("--api-key", default=os.environ.get("ROBOFLOW_API_KEY", ""),
                   help="Roboflow API key (or set ROBOFLOW_API_KEY).")
    p.add_argument("--workspace", default=os.environ.get("ROBOFLOW_WORKSPACE", ""),
                   help="Roboflow workspace id (or set ROBOFLOW_WORKSPACE).")
    p.add_argument("--project", required=False,
                   default=os.environ.get("ROBOFLOW_PROJECT", ""),
                   help="Roboflow project id (or set ROBOFLOW_PROJECT).")
    p.add_argument("--version", type=int,
                   default=int(os.environ.get("ROBOFLOW_VERSION", "1")),
                   help="Roboflow model version.")
    p.add_argument("--api-url", default="https://detect.roboflow.com",
                   help="Roboflow inference endpoint.")

    # OCR
    p.add_argument("--ocr-engine", choices=["tesseract", "easyocr", "paddleocr"],
                   default="tesseract")
    p.add_argument("--ocr-gpu", action="store_true", help="Use GPU for OCR.")

    # Thresholds
    p.add_argument("--match-threshold", type=float, default=0.60,
                   help="Absolute floor: min template+ORB score to rename.")
    p.add_argument("--match-margin", type=float, default=0.08,
                   help="Best match must beat the 2nd-best by this margin.")
    p.add_argument("--hash-algorithm",
                   choices=["phash", "dhash", "ahash", "whash"], default="phash",
                   help="pHash variant used for the Hamming .txt reports.")

    # Output / misc
    p.add_argument("--output-dir", default="output",
                   help="Where crops/JSON artefacts are written.")
    p.add_argument("--no-crops", action="store_true", help="Do not save crops.")
    p.add_argument("--no-viz", action="store_true",
                   help="Do not save annotated visualization images.")
    p.add_argument("--no-auto-rotate", action="store_true",
                   help="Disable automatic correction of sideways (90/180/270 deg) "
                        "legend/map images.")
    p.add_argument("-v", "--verbose", action="store_true", help="Debug logging.")
    return p


def config_from_args(args: argparse.Namespace) -> PipelineConfig:
    return PipelineConfig(
        api_key=args.api_key,
        workspace=args.workspace,
        project=args.project,
        version=args.version,
        api_url=args.api_url,
        ocr_engine=args.ocr_engine,
        ocr_gpu=args.ocr_gpu,
        hash_algorithm=args.hash_algorithm,
        match_score_threshold=args.match_threshold,
        match_margin=args.match_margin,
        output_dir=args.output_dir,
        save_crops=not args.no_crops,
        save_visualization=not args.no_viz,
        auto_rotate=not args.no_auto_rotate,
    )


def validate(args: argparse.Namespace) -> None:
    """Fail fast with actionable messages before doing any heavy work."""
    problems: List[str] = []
    if not os.path.isfile(args.map_path):
        problems.append(f"--map not found: {args.map_path}")
    if not os.path.isfile(args.legend_path):
        problems.append(f"--legend not found: {args.legend_path}")
    if not args.api_key:
        problems.append("Roboflow API key missing (--api-key / ROBOFLOW_API_KEY).")
    if not args.project:
        problems.append("Roboflow project missing (--project / ROBOFLOW_PROJECT).")
    if problems:
        for prob in problems:
            LOGGER.error(prob)
        raise SystemExit(2)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    setup_logging(verbose=args.verbose)
    validate(args)

    config = config_from_args(args)
    pipeline = LegendMarkerPipeline(config)

    try:
        results = pipeline.run(args.map_path, args.legend_path)
    except Exception as exc:
        LOGGER.exception("Pipeline failed: %s", exc)
        return 1

    # Print a concise summary to stdout (the full artefact is in results.json).
    print(json.dumps(results, indent=2, ensure_ascii=False))
    return 0
