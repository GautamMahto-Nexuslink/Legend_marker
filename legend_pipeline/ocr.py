"""Step 2: OCR (Tesseract / EasyOCR / PaddleOCR) yielding cleaned OcrText."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .config import PipelineConfig
from .containers import OcrText
from .deps import LOGGER, cv2


class OcrEngine:
    """OCR wrapper (Tesseract / EasyOCR / PaddleOCR) yielding cleaned OcrText.

    Tesseract is the default: on the small, printed text of map legends it
    consistently reads far more real labels than EasyOCR (which tends to return
    icon-glyph garbage on low-resolution crops).
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self._reader = self._build_reader()

    def _build_reader(self) -> Any:
        engine = self.config.ocr_engine.lower()
        if engine == "tesseract":
            return self._build_tesseract()
        if engine == "easyocr":
            return self._build_easyocr()
        if engine == "paddleocr":
            return self._build_paddleocr()
        raise ValueError(f"Unknown OCR engine: {self.config.ocr_engine!r}")

    def _build_tesseract(self) -> Any:
        try:
            import pytesseract  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "pytesseract not installed. Install with: pip install pytesseract "
                "(and the tesseract binary: apt-get install tesseract-ocr)"
            ) from exc
        # Fail early with a clear message if the binary itself is missing.
        try:
            pytesseract.get_tesseract_version()
        except Exception as exc:
            raise RuntimeError(
                "The tesseract binary was not found. Install it, e.g. "
                "`sudo apt-get install tesseract-ocr`."
            ) from exc
        LOGGER.info("Using Tesseract OCR (lang=%s, config=%r).",
                    self.config.tesseract_lang, self.config.tesseract_config)
        return pytesseract

    def _build_easyocr(self) -> Any:
        try:
            import easyocr  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "easyocr not installed. Install with: pip install easyocr"
            ) from exc
        LOGGER.info("Initialising EasyOCR (langs=%s, gpu=%s)",
                    self.config.ocr_languages, self.config.ocr_gpu)
        return easyocr.Reader(list(self.config.ocr_languages), gpu=self.config.ocr_gpu)

    def _build_paddleocr(self) -> Any:
        try:
            from paddleocr import PaddleOCR  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "paddleocr not installed. Install with: pip install paddleocr"
            ) from exc
        LOGGER.info("Initialising PaddleOCR.")
        lang = self.config.ocr_languages[0] if self.config.ocr_languages else "en"
        return PaddleOCR(use_angle_cls=True, lang=lang, show_log=False)

    # -- Text cleaning ----------------------------------------------------
    @staticmethod
    def _clean_text(text: str) -> str:
        """Collapse whitespace and strip stray control characters/noise."""
        # Normalise all whitespace runs to a single space.
        cleaned = " ".join(str(text).split())
        # Remove characters that are almost always OCR noise at token edges.
        return cleaned.strip(" .:;|_-[](){}<>*=+~^`\"'").strip()

    @staticmethod
    def _is_real_word(text: str, min_letters: int) -> bool:
        """True if the token is genuine text (enough letters), not a number/symbol.

        Rejects pure numbers ("4", "0.91"), punctuation/symbols ("=", "@", "#")
        and icon-glyph noise, while keeping any real label (which always has
        several letters, even things like "Boat Docks 8 a.m.").
        """
        return sum(ch.isalpha() for ch in text) >= min_letters

    # -- Inference --------------------------------------------------------
    def _resize_for_ocr(
        self, image: np.ndarray, target_long_side: Optional[int] = None
    ) -> Tuple[np.ndarray, float]:
        """Resize a legend so its text is a good size for OCR.

        With ``target_long_side=None`` this keeps the original behaviour: only
        *upscale* small legends toward ``ocr_target_long_side`` (never shrink).

        With an explicit positive ``target_long_side`` the image is scaled toward
        that cap in *either* direction — so a large legend is DOWNSCALED, which
        the throwaway false-positive-scan pass uses to run much faster.

        Returns the (possibly resized) image and the scale factor applied, so
        detected boxes can be mapped back to the original coordinate system.
        """
        h, w = image.shape[:2]
        long_side = max(h, w)

        if target_long_side and target_long_side > 0:
            scale = target_long_side / float(long_side)
            if abs(scale - 1.0) <= 0.01:
                return image, 1.0
            if scale > 1.0:
                if not self.config.ocr_upscale:
                    return image, 1.0
                scale = min(scale, self.config.ocr_max_upscale)
            interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
        else:
            if not self.config.ocr_upscale:
                return image, 1.0
            if long_side >= self.config.ocr_target_long_side:
                return image, 1.0
            scale = min(self.config.ocr_max_upscale,
                        self.config.ocr_target_long_side / float(long_side))
            if scale <= 1.01:
                return image, 1.0
            interp = cv2.INTER_CUBIC

        out = cv2.resize(image, (int(round(w * scale)), int(round(h * scale))),
                         interpolation=interp)
        LOGGER.info("Resized legend %dx%d -> %dx%d (x%.2f) for OCR.",
                    w, h, out.shape[1], out.shape[0], scale)
        return out, scale

    def read(
        self, image: np.ndarray, target_long_side: Optional[int] = None
    ) -> List[OcrText]:
        """Run OCR and return cleaned, spatially-aware text tokens.

        ``target_long_side`` optionally caps the working resolution (see
        :meth:`_resize_for_ocr`); the FP-scan pass uses it to run faster.
        """
        proc, scale = self._resize_for_ocr(image, target_long_side)
        engine = self.config.ocr_engine.lower()
        if engine == "tesseract":
            raw = self._read_tesseract(proc)
        elif engine == "easyocr":
            raw = self._read_easyocr(proc)
        else:
            raw = self._read_paddleocr(proc)

        inv = 1.0 / scale if scale else 1.0
        results: List[OcrText] = []
        for text, conf, box in raw:
            # Map the box from the upscaled image back to original coordinates.
            if scale != 1.0:
                box = (int(round(box[0] * inv)), int(round(box[1] * inv)),
                       int(round(box[2] * inv)), int(round(box[3] * inv)))
            cleaned = self._clean_text(text)
            if not cleaned:
                continue
            if conf < self.config.ocr_min_confidence:
                LOGGER.debug("Dropping low-conf OCR %r (%.2f)", cleaned, conf)
                continue
            # Keep only real words — drop pure numbers / special characters.
            if not self._is_real_word(cleaned, self.config.text_min_letters):
                LOGGER.debug("Dropping non-word OCR %r (number/symbol).", cleaned)
                continue
            results.append(OcrText(text=cleaned, confidence=float(conf), bbox=box))
        LOGGER.info("OCR produced %d cleaned text token(s).", len(results))
        return results

    def _read_tesseract(
        self, image: np.ndarray
    ) -> List[Tuple[str, float, Tuple[int, int, int, int]]]:
        """Phrase-level tokens from Tesseract (best for small printed legends).

        Tesseract returns words; we group them back into phrases per text line,
        splitting a line wherever there is a big horizontal gap (a gap marks a
        column boundary or the jump from an icon glyph to its label).  So a
        label like "Picnic Area" stays one token, while single-character junk
        Tesseract reads over an icon becomes its own token and is filtered out.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        data = self._reader.image_to_data(
            gray,
            lang=self.config.tesseract_lang,
            config=self.config.tesseract_config,
            output_type=self._reader.Output.DICT,
        )

        # Collect words grouped by Tesseract's (block, paragraph, line) index.
        lines: Dict[Tuple[int, int, int], List[Tuple[str, float, int, int, int, int]]] = {}
        for i in range(len(data["text"])):
            text = data["text"][i].strip()
            try:
                conf = float(data["conf"][i])
            except (TypeError, ValueError):
                conf = -1.0
            if not text or conf < 0:
                continue
            # Drop pure symbol/number words (icon-glyph noise like "$", "|", "=")
            # BEFORE grouping, so they never get merged into a real label phrase.
            if not any(ch.isalpha() for ch in text):
                continue
            key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
            x, y = int(data["left"][i]), int(data["top"][i])
            w, h = int(data["width"][i]), int(data["height"][i])
            lines.setdefault(key, []).append((text, conf, x, y, x + w, y + h))

        out: List[Tuple[str, float, Tuple[int, int, int, int]]] = []
        for words in lines.values():
            words.sort(key=lambda t: t[2])            # left-to-right.
            med_h = float(np.median([w[5] - w[3] for w in words])) or 12.0
            gap_thresh = 1.5 * med_h                  # bigger gap => new phrase.
            run: List[Tuple[str, float, int, int, int, int]] = [words[0]]
            for wd in words[1:]:
                if wd[2] - run[-1][4] <= gap_thresh:
                    run.append(wd)
                else:
                    out.append(self._merge_word_run(run))
                    run = [wd]
            out.append(self._merge_word_run(run))
        return out

    @staticmethod
    def _merge_word_run(
        run: List[Tuple[str, float, int, int, int, int]]
    ) -> Tuple[str, float, Tuple[int, int, int, int]]:
        """Join a run of words into one phrase token (union box, mean conf)."""
        text = " ".join(w[0] for w in run)
        conf = float(np.mean([w[1] for w in run])) / 100.0   # 0-100 -> 0-1.
        x1 = min(w[2] for w in run)
        y1 = min(w[3] for w in run)
        x2 = max(w[4] for w in run)
        y2 = max(w[5] for w in run)
        return text, conf, (x1, y1, x2, y2)

    def _read_easyocr(
        self, image: np.ndarray
    ) -> List[Tuple[str, float, Tuple[int, int, int, int]]]:
        # EasyOCR returns [ [box(4 pts)], text, confidence ].
        # Keep grouping tight: EasyOCR's default width_ths (0.5) merges
        # horizontally-close boxes, which glues an icon's glyph (e.g. the
        # "H<tent>B" symbol misread as "HAE") onto its neighbouring label
        # ("Hike & Bike Campground"). Once merged, filter_text_on_icons can't
        # strip the glyph because the combined box sits mostly OFF the icon.
        # Emitting finer boxes lets the glyph become its own token that the
        # on-icon filter drops; a label's real words are re-joined later by
        # _contiguous_tokens / _merge_texts, so tighter splitting is safe.
        detections = self._reader.readtext(image, width_ths=0.1, paragraph=False)
        out = []
        for box, text, conf in detections:
            xs = [p[0] for p in box]
            ys = [p[1] for p in box]
            bbox = (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))
            out.append((text, float(conf), bbox))
        return out

    def _read_paddleocr(
        self, image: np.ndarray
    ) -> List[Tuple[str, float, Tuple[int, int, int, int]]]:
        # PaddleOCR returns [[ [box(4pts)], (text, conf) ], ...] per image.
        result = self._reader.ocr(image, cls=True)
        out = []
        # result is a list (one entry per image); guard for None/empty.
        for page in result or []:
            for line in page or []:
                box, (text, conf) = line
                xs = [p[0] for p in box]
                ys = [p[1] for p in box]
                bbox = (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))
                out.append((text, float(conf), bbox))
        return out
