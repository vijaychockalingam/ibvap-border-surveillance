"""
Draw OR edit a camera's "restricted zone" polygon.

    python select_zone.py path/to/your_video.mp4 [output_zone_file]

If output_zone_file already exists, its points are loaded automatically so
you can edit them instead of starting over:
  - Left-click empty space              = add a new point
  - Left-click + drag an existing point = move it
  - Right-click an existing point       = delete it
  - r = clear all points and start fresh
  - q = quit

The file auto-saves after every add/move/delete (as long as you have 3+
points), so it's always up to date on disk - you don't need to remember to
press a save key, and it's safe to close the window with the X button.

If you're setting up multiple cameras (see cameras.json), pass the matching
zone_config filename for that camera, e.g.:

    python select_zone.py sample1.mp4 zone_cam1.json
    python select_zone.py sample2.mp4 zone_cam2.json

output_zone_file defaults to zone_config.json if omitted (single-camera setup).
"""
import json
import math
import sys

import cv2
import numpy as np

import config
from zone import load_zone

points = []
dragging_idx = None
output_path = "zone_config.json"
DRAG_RADIUS = 12  # pixels - how close a click needs to be to grab/delete a point


def _nearest_point_idx(x, y):
    if not points:
        return None
    dists = [math.dist((x, y), p) for p in points]
    idx = int(np.argmin(dists))
    return idx if dists[idx] <= DRAG_RADIUS else None


def _autosave():
    """Writes the current points to disk immediately, as long as they form
    a valid polygon. Called after every edit so the file is never stale -
    no explicit save step to forget."""
    if len(points) >= 3:
        with open(output_path, "w") as f:
            json.dump({"polygon": points}, f, indent=2)
        print(f"Saved {output_path} ({len(points)} points)")


def click_event(event, x, y, flags, param):
    global dragging_idx

    if event == cv2.EVENT_LBUTTONDOWN:
        idx = _nearest_point_idx(x, y)
        if idx is not None:
            dragging_idx = idx  # start dragging this existing point
        else:
            points.append([x, y])
            print(f"Point added: ({x}, {y})")
            _autosave()

    elif event == cv2.EVENT_MOUSEMOVE:
        if dragging_idx is not None:
            points[dragging_idx] = [x, y]

    elif event == cv2.EVENT_LBUTTONUP:
        if dragging_idx is not None:
            dragging_idx = None
            _autosave()

    elif event == cv2.EVENT_RBUTTONDOWN:
        idx = _nearest_point_idx(x, y)
        if idx is not None:
            removed = points.pop(idx)
            print(f"Point removed: {removed}")
            _autosave()


def main():
    global points, output_path

    if len(sys.argv) < 2:
        print("Usage: python select_zone.py <video_path> [output_zone_file]")
        return

    video_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "zone_config.json"
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"Could not read a frame from: {video_path}")
        return

    # Resize to the SAME width the live processor uses, so the polygon
    # points line up correctly during the actual demo.
    h, w = frame.shape[:2]
    scale = config.RESIZE_WIDTH / w
    frame = cv2.resize(frame, (config.RESIZE_WIDTH, int(h * scale)))
    clone = frame.copy()

    existing = load_zone(output_path)
    if existing:
        points = [list(p) for p in existing]
        print(f"Loaded {len(points)} existing point(s) from {output_path} - "
              f"drag to move, right-click to delete, left-click empty space to add.")

    window = "Edit restricted zone (auto-saves) - drag=move, right-click=delete, R=reset, Q=quit"
    cv2.namedWindow(window)
    cv2.setMouseCallback(window, click_event)

    while True:
        display = clone.copy()
        for i, p in enumerate(points):
            color = (0, 165, 255) if i == dragging_idx else (0, 0, 255)
            cv2.circle(display, tuple(p), 5, color, -1)
        if len(points) > 1:
            cv2.polylines(display, [np.array(points)], True, (0, 255, 255), 2)

        saved_state = "SAVED" if len(points) >= 3 else "need 3+ points to save"
        status = f"Points: {len(points)} ({saved_state})  |  drag=move  right-click=delete  R=reset  Q=quit"
        cv2.rectangle(display, (0, 0), (display.shape[1], 24), (0, 0, 0), -1)
        cv2.putText(display, status, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow(window, display)

        key = cv2.waitKey(1) & 0xFF
        key_char = chr(key).lower() if key != 255 else None

        # Also treat the window being closed via the X button as "quit" -
        # cv2.waitKey keeps returning -1 forever otherwise and the script
        # would hang invisibly.
        try:
            window_closed = cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1
        except cv2.error:
            window_closed = True

        if key_char == "r":
            points = []
            print("Points cleared. (File on disk keeps your last saved zone until you draw a new one.)")
        elif key_char == "q" or window_closed:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
