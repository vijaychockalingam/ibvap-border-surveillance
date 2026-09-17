"""
Basic ANPR (Automatic Number Plate Recognition).

For a 1-day prototype we don't run a separate plate-detector model - we just
run OCR on the cropped vehicle bounding box and keep whichever detected text
looks plate-shaped (mostly uppercase letters/digits, 4-10 characters). This
is exactly the "demo it on sample footage, don't claim production-grade"
scope the problem statement calls for.

Requires easyocr (see requirements.txt). If it's not installed, ANPR is
silently disabled so the rest of the app keeps working without it.
"""
import re

import config

_PLATE_PATTERN = re.compile(r"^[A-Z0-9]{4,10}$")

_reader = None
_available = False

if config.ENABLE_ANPR:
    try:
        import easyocr
        _reader = easyocr.Reader(["en"], gpu=False, verbose=False)
        _available = True
    except Exception as e:  # ImportError, or model download failure, etc.
        print(f"[IBVAP] ANPR disabled - could not load easyocr ({e}). "
              f"Run: pip install easyocr    to enable plate reading.")
        _available = False


def anpr_available():
    return _available


def read_plate(crop):
    """
    crop: a BGR image (numpy array) of just the vehicle's bounding box.
    Returns (plate_text, confidence) or (None, 0.0) if nothing plate-like
    was found.
    """
    if not _available or crop is None or crop.size == 0:
        return None, 0.0

    h, w = crop.shape[:2]
    if h < 10 or w < 10:
        # Too small to plausibly contain a readable plate - skip the OCR
        # call entirely rather than wasting CPU on a degenerate crop (this
        # was already exception-safe below, but no point paying the cost).
        return None, 0.0

    try:
        results = _reader.readtext(crop)
    except Exception:
        return None, 0.0

    best_text, best_conf = None, 0.0
    for (_bbox, text, conf) in results:
        cleaned = text.upper().replace(" ", "").replace("-", "")
        if _PLATE_PATTERN.match(cleaned) and conf > best_conf:
            best_text, best_conf = cleaned, conf

    return best_text, best_conf
