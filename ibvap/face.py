"""
Face DETECTION (not recognition/identification) - finds whether a face is
visible in a crop, using OpenCV's bundled Haar cascade. No extra pip
install needed - cv2.data ships with opencv-python.

Deliberately not doing identity matching here: that needs an enrolled photo
database and a much heavier embedding model, and accuracy on real CCTV
footage is genuinely unreliable - the original plan intentionally scoped
this out. This module answers "is there a face here", which is cheap and
useful as its own signal without overclaiming what it can do.
"""
import cv2

_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def has_face(crop):
    if crop is None or crop.size == 0:
        return False
    h, w = crop.shape[:2]
    # A crop thinner than the cascade's own minSize can trigger a native
    # OpenCV crash ("invalid vector<T> subscript") deep in its scale
    # pyramid - guard against it directly rather than relying on the
    # per-camera crash-recovery to paper over it every time.
    if h < 20 or w < 20:
        return False
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    try:
        faces = _cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))
    except cv2.error:
        return False
    return len(faces) > 0
