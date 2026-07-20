"""Optional third-party dependencies + shared logger.

We import them lazily/defensively so that the package can at least be imported
(and ``--help`` shown) even if an optional engine is missing.  Each import error
is turned into a clear, actionable message at the point of use via
:func:`_require`.
"""
from __future__ import annotations

import logging
from typing import Any

try:
    import cv2  # OpenCV — image IO, colour, ORB, resizing.
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore

try:
    from PIL import Image  # Pillow — required by imagehash.
except ImportError:  # pragma: no cover
    Image = None  # type: ignore

try:
    import imagehash  # Perceptual hashing (pHash / dHash / …) — reporting only.
except ImportError:  # pragma: no cover
    imagehash = None  # type: ignore


LOGGER = logging.getLogger("legend_marker")


def _require(module: Any, name: str, install_hint: str) -> None:
    """Raise a clear error if an optional dependency is missing."""
    if module is None:
        raise RuntimeError(
            f"The '{name}' library is required for this step but is not "
            f"installed. Install it with: {install_hint}"
        )
