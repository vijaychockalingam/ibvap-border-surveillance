"""
Defines the set of cameras IBVAP knows about. Edit cameras.json to add more
- each camera needs a unique id, a display name, a video source (file path,
or an integer string like "0" for a webcam), and its own zone config file
(each camera can have a different restricted zone shape/position).

If cameras.json doesn't exist, falls back to a single camera using the
IBVAP_VIDEO env var (same behavior as the original single-camera version).
"""
import json
import os

CAMERAS_CONFIG_PATH = "cameras.json"


def load_cameras():
    if os.path.exists(CAMERAS_CONFIG_PATH):
        with open(CAMERAS_CONFIG_PATH) as f:
            return json.load(f)

    # Fallback: behave like the original single-camera setup
    return {
        "cam1": {
            "name": "Camera 1",
            "source": os.environ.get("IBVAP_VIDEO", "sample.mp4"),
            "zone_config": "zone_config.json",
        }
    }
