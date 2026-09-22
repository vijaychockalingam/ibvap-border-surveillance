"""
Defines and loads the camera configuration for IBVAP.
Supports 6-camera multi-camera configuration with logical processing node assignments.
"""
import json
import os

CAMERAS_CONFIG_PATH = "cameras.json"


def load_cameras():
    """Loads camera configurations from cameras.json with sensible defaults."""
    if os.path.exists(CAMERAS_CONFIG_PATH):
        with open(CAMERAS_CONFIG_PATH, "r", encoding="utf-8") as f:
            cams = json.load(f)
            # Ensure every camera has standard keys
            for idx, (cid, cam) in enumerate(cams.items()):
                cam.setdefault("name", f"Camera {cid}")
                cam.setdefault("source", f"sample{idx+1}.mp4")
                cam.setdefault("location", f"Sector {idx+1}")
                cam.setdefault("zone_config", f"zone_{cid}.json")
                cam.setdefault("enabled", True)
                cam.setdefault("processing_node", "node_1" if idx < 3 else "node_2")
            return cams

    # Fallback configuration for 6 cameras if file not present
    return {
        f"cam{i+1}": {
            "name": f"BOP Camera {i+1}",
            "source": f"sample{i+1}.mp4",
            "location": f"Sector {i+1} Perimeter",
            "zone_config": f"zone_cam{i+1}.json",
            "enabled": True,
            "processing_node": "node_1" if i < 3 else "node_2",
        }
        for i in range(6)
    }


def get_cameras_for_node(node_id):
    """Returns a dict of camera configurations assigned to a given node_id."""
    all_cams = load_cameras()
    return {cid: cam for cid, cam in all_cams.items() if cam.get("processing_node") == node_id and cam.get("enabled", True)}
