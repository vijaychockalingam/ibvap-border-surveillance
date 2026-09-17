# IBVAP config
# Tuned for CPU-only inference (integrated graphics, 16GB RAM).
# If the video feed feels laggy on your machine, raise FRAME_SKIP first,
# then lower RESIZE_WIDTH. Defaults below are tuned for running several
# cameras simultaneously (each camera runs its own full pipeline in
# parallel, so CPU load scales with camera count) - if you're only running
# 1-2 cameras you can safely lower FRAME_SKIP back toward 2.
#
# NOTE: if you change RESIZE_WIDTH, redraw every zone with select_zone.py
# afterward - existing zone_cam*.json files were drawn at the current
# RESIZE_WIDTH and will be misaligned at a different one.

RESIZE_WIDTH = 640            # all frames are resized to this width before
                               # detection AND before zone selection, so the
                               # two stay in the same coordinate space
FRAME_SKIP = 3                 # run detection every Nth frame (tracker fills
                               # in the gaps), raise to 3-4 if it's laggy
CONF_THRESHOLD = 0.4           # detection confidence threshold
NIGHT_BRIGHTNESS_THRESHOLD = 60  # avg grayscale pixel value below this = night
ALERT_COOLDOWN_SECONDS = 5     # don't re-alert on the same tracked object
                                 # more often than this
MODEL_PATH = "yolov8n.pt"      # nano model - required for CPU speed

# ANPR (number plate reading)
ENABLE_ANPR = True             # set False to disable plate reading entirely
ANPR_MAX_ATTEMPTS = 8          # give up trying to read a given vehicle's
                                 # plate after this many attempts (it may be
                                 # too far away / angled / blurry)

# Face detection (bounding boxes only - NOT identity recognition/matching).
# See README for why this prototype deliberately stops at detection.
ENABLE_FACE_DETECTION = True
FACE_MAX_ATTEMPTS = 5          # give up checking a given person for a
                                 # visible face after this many attempts

WATCHLIST_PATH = "watchlist.json"

# Suspicious activity (loitering) - a person continuously inside the zone
# past this many seconds gets a separate LOITERING event on top of the
# immediate INTRUSION alert. Kept short by default so it's demoable on
# short test clips; a real deployment would likely use ~60s (the number the
# original problem-statement planning used).
LOITERING_SECONDS = 8

# How many times per second each camera's video_feed route checks the
# background thread's buffer for a new frame to stream to the browser.
STREAM_FPS = 12
