"""
IBVAP Configuration
Tuned for CPU-only multi-camera execution (16GB RAM).
Supports logical multi-node AI processing and dynamic stream management.
"""

RESIZE_WIDTH = 640            # Frame width for uniform processing and coordinate space
FRAME_SKIP = 3                 # Run AI detection every Nth frame (tracker fills in gaps)
CONF_THRESHOLD = 0.4           # YOLO detection confidence threshold
NIGHT_BRIGHTNESS_THRESHOLD = 60  # Average grayscale brightness below this triggers Night Mode
ALERT_COOLDOWN_SECONDS = 5     # Minimum seconds between duplicate alerts for the same object
MODEL_PATH = "yolo26n.pt"      # Local YOLO26 nano model path

# ANPR (Automatic Number Plate Recognition)
ENABLE_ANPR = True             # Enable/disable plate reading via EasyOCR
ANPR_MAX_ATTEMPTS = 8          # Maximum OCR attempts per vehicle object

# Face detection (Haar cascade bounding boxes - detection only, not identity recognition)
ENABLE_FACE_DETECTION = True
FACE_MAX_ATTEMPTS = 5          # Maximum face check attempts per person object

WATCHLIST_PATH = "watchlist.json"

# Suspicious activity (loitering in virtual restricted zone)
LOITERING_SECONDS = 8          # Continuous seconds in restricted zone before LOITERING alert

# Stream & Display settings
STREAM_FPS = 15                # Target frames per second for browser display feeds
AI_INFERENCE_FPS = 8           # Default AI inference cycle rate

# Adaptive AI Processing Limits
IDLE_FPS = 3
ACTIVE_FPS = 8
HIGH_ALERT_FPS = 12

# Logical Processing Nodes Definition
NODES_CONFIG = {
    "node_1": {
        "name": "AI Processing Node 1",
        "description": "Logical worker managing Cameras 1-3",
        "default_cameras": ["cam1", "cam2", "cam3"]
    },
    "node_2": {
        "name": "AI Processing Node 2",
        "description": "Logical worker managing Cameras 4-6",
        "default_cameras": ["cam4", "cam5", "cam6"]
    }
}

# Intelligence Layer Configurations

# Camera Topology Graph (which cameras can entities physically move between)
CAMERA_TOPOLOGY = {
    "cam1": ["cam2"],
    "cam2": ["cam1", "cam3", "cam4"],
    "cam3": ["cam2", "cam6"],
    "cam4": ["cam2", "cam5"],
    "cam5": ["cam4", "cam6"],
    "cam6": ["cam5", "cam3"]
}

# Zone Behavior Baselines (hours of the day when activity is EXPECTED)
# e.g., "cam1": (6, 22) means normal activity is between 6 AM and 10 PM.
# If activity happens outside these hours, it's flagged as unusual.
ZONE_BASELINES = {
    "cam1": (6, 22), 
    "cam2": (8, 18),
    "cam3": (0, 0),  # Restricted 24/7 (start == end means always restricted)
    "cam4": (6, 22),
    "cam5": (6, 22),
    "cam6": (8, 18)
}
