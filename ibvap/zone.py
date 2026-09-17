import json
import os

import cv2
import numpy as np


def load_zone(path="zone_config.json"):
    """Returns a list of [x, y] points, or None if no zone has been defined yet."""
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    return data.get("polygon")


def point_in_zone(point, polygon):
    if not polygon:
        return False
    contour = np.array(polygon, dtype=np.int32)
    return cv2.pointPolygonTest(contour, point, False) >= 0
