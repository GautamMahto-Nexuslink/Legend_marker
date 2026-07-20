"""Step 1 & 5: Roboflow inference (local ``inference.get_model``)."""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import numpy as np

from .config import PipelineConfig
from .containers import Detection
from .deps import LOGGER
from .utils import safe_crop


class RoboflowDetector:
    """Wrapper around local Roboflow inference via the ``inference`` package.

    Uses ``inference.get_model`` to load the model in-process (no HTTP API), then
    calls ``model.infer(...)``.  The response is normalised into a plain dict
    with a ``predictions`` list so the rest of the pipeline is model-agnostic.
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self._model = self._load_model()

    def _load_model(self) -> Any:
        """Load the model with ``inference.get_model`` (raises if unavailable)."""
        try:
            from inference import get_model  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "The 'inference' package is required. Install it with: "
                "pip install inference"
            ) from exc

        LOGGER.info("Loading Roboflow model '%s' via inference.get_model.",
                    self._model_id)
        # get_model resolves & caches weights locally; api_key authorises the pull.
        return get_model(model_id=self._model_id, api_key=self.config.api_key)

    @property
    def _model_id(self) -> str:
        """Roboflow model id in the form 'project/version'."""
        return f"{self.config.project}/{self.config.version}"

    def infer(self, image_path: str) -> Dict[str, Any]:
        """Run local inference and return a normalised ``{'predictions': [...]}`` dict."""
        LOGGER.info("Running Roboflow inference (local get_model) on %s", image_path)
        # inference models accept a path/np.ndarray/PIL/URL. Confidence is a
        # 0-1 fraction here (unlike the 0-100 percent HTTP API convention).
        raw = self._model.infer(
            image_path,
            confidence=self.config.rf_confidence / 100.0,
        )
        return self._normalize_response(raw)

    @staticmethod
    def _normalize_response(raw: Any) -> Dict[str, Any]:
        """Coerce an inference-package response into a plain predictions dict.

        ``model.infer`` returns either a single response object or a list of
        them (one per input image).  Each may be a pydantic model or a dict; we
        handle all shapes and always emit ``{"predictions": [ {...}, ... ]}``.
        """
        # Unwrap a per-image list to its first element (we infer one image).
        if isinstance(raw, (list, tuple)):
            raw = raw[0] if raw else {}

        # Pydantic model -> dict (support both v1 .dict() and v2 .model_dump()).
        if hasattr(raw, "model_dump"):
            data = raw.model_dump()
        elif hasattr(raw, "dict"):
            data = raw.dict()
        elif isinstance(raw, dict):
            data = raw
        else:
            data = {"predictions": getattr(raw, "predictions", []) or []}

        preds = data.get("predictions", []) or []
        normalized: List[Dict[str, Any]] = []
        for p in preds:
            # Each prediction may itself be a model or a dict.
            if hasattr(p, "model_dump"):
                p = p.model_dump()
            elif hasattr(p, "dict"):
                p = p.dict()
            elif not isinstance(p, dict):
                p = {
                    "x": getattr(p, "x", None),
                    "y": getattr(p, "y", None),
                    "width": getattr(p, "width", None),
                    "height": getattr(p, "height", None),
                    "confidence": getattr(p, "confidence", 0.0),
                    "class": getattr(p, "class_name", getattr(p, "class", "icon")),
                    "points": getattr(p, "points", None),
                }
            # The inference package names the label 'class_name'; the rest of the
            # pipeline expects 'class'. Bridge the two without losing either.
            if "class" not in p and "class_name" in p:
                p["class"] = p["class_name"]
            normalized.append(p)

        data["predictions"] = normalized
        return data

    def detect(
        self,
        image_path: str,
        image: np.ndarray,
        raw_dump_path: Optional[str] = None,
    ) -> List[Detection]:
        """Run inference and convert the response into Detection objects (+crops).

        If ``raw_dump_path`` is given, the *unmodified* Roboflow JSON response is
        written there verbatim, so the raw inference output is always auditable.
        """
        raw = self.infer(image_path)

        if raw_dump_path is not None:
            try:
                with open(raw_dump_path, "w", encoding="utf-8") as fh:
                    json.dump(raw, fh, indent=2, ensure_ascii=False, default=str)
                LOGGER.info("Saved raw Roboflow response -> %s", raw_dump_path)
            except Exception as exc:  # Saving must never break the pipeline.
                LOGGER.warning("Could not save raw response to %s: %s",
                               raw_dump_path, exc)

        predictions = raw.get("predictions", []) or []
        LOGGER.info("Roboflow returned %d prediction(s).", len(predictions))

        detections: List[Detection] = []
        for pred in predictions:
            try:
                det = self._parse_prediction(pred, image)
            except Exception as exc:  # Never let one bad box kill the batch.
                LOGGER.warning("Skipping malformed prediction: %s (%s)", pred, exc)
                continue
            detections.append(det)
        return detections

    @staticmethod
    def _parse_prediction(pred: Dict[str, Any], image: np.ndarray) -> Detection:
        """Roboflow uses centre-x/centre-y/width/height in pixels."""
        cx = float(pred["x"])
        cy = float(pred["y"])
        w = float(pred["width"])
        h = float(pred["height"])
        x1 = int(round(cx - w / 2.0))
        y1 = int(round(cy - h / 2.0))
        x2 = int(round(cx + w / 2.0))
        y2 = int(round(cy + h / 2.0))

        polygon = None
        if "points" in pred and pred["points"]:
            polygon = [(float(p["x"]), float(p["y"])) for p in pred["points"]]

        bbox = (x1, y1, x2, y2)
        crop = safe_crop(image, bbox)
        return Detection(
            class_name=str(pred.get("class", "icon")),
            confidence=float(pred.get("confidence", 0.0)),
            bbox=bbox,
            polygon=polygon,
            crop=crop,
        )
