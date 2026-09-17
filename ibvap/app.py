import os
import sys
import time
import warnings

# Suppress PyTorch and OpenCV noisy console deprecation warnings
warnings.filterwarnings("ignore")
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"
os.environ["PYTHONWARNINGS"] = "ignore"

from flask import Flask, Response, jsonify, render_template

import config
import database as db
from cameras import load_cameras
from processor import IBVAPProcessor

app = Flask(__name__)

CAMERAS = load_cameras()
db.init_db()

# Every camera starts and runs continuously in its own background thread -
# all of them are always live, not just whichever one you're currently
# looking at. A camera whose video file is missing/bad is skipped (and
# marked offline in the UI) instead of taking down the other cameras.
processors = {}
for cam_id, cam in CAMERAS.items():
    try:
        proc = IBVAPProcessor(
            cam["source"],
            zone_path=cam.get("zone_config", f"zone_{cam_id}.json"),
            camera_id=cam_id,
            camera_name=cam.get("name", cam_id),
        )
        proc.start()
        processors[cam_id] = proc
        print(f"[IBVAP] {cam.get('name', cam_id)} started.")
    except RuntimeError as e:
        print(f"[IBVAP] WARNING: could not start {cam.get('name', cam_id)}: {e}")

if len(processors) > 1:
    print(f"[IBVAP] {len(processors)} cameras running simultaneously. If playback "
          f"feels slow, raise config.FRAME_SKIP or lower config.RESIZE_WIDTH - "
          f"each camera runs its own full detection pipeline in parallel.")

_STREAM_INTERVAL = 1.0 / config.STREAM_FPS


@app.route("/")
def index():
    return render_template("index.html", cameras=CAMERAS, online={cid: (cid in processors) for cid in CAMERAS})


@app.route("/snapshot/<camera_id>")
def snapshot(camera_id):
    processor = processors.get(camera_id)
    if processor is None:
        return "Camera not found", 404
    frame_bytes = processor.get_latest_jpeg()
    if frame_bytes is None:
        return "", 204
    return Response(frame_bytes, mimetype="image/jpeg")


@app.route("/video_feed/<camera_id>")
def video_feed(camera_id):
    processor = processors.get(camera_id)
    if processor is None:
        return "Camera unavailable (failed to start - check the source in cameras.json)", 404

    def gen():
        try:
            while True:
                frame_bytes = processor.get_latest_jpeg()
                if frame_bytes is not None:
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
                time.sleep(_STREAM_INTERVAL)
        except (GeneratorExit, Exception):
            pass

    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")




@app.route("/api/cameras")
def api_cameras():
    out = {}
    for cid, cam in CAMERAS.items():
        proc = processors.get(cid)
        alive = proc.is_alive() if proc else False
        last_event = proc.get_last_event() if proc else None
        out[cid] = {
            "name": cam.get("name", cid),
            "online": alive,
            "last_event_type": last_event[0] if last_event else None,
            "last_event_severity": last_event[1] if last_event else None,
            "last_event_time": last_event[2] if last_event else None,
        }
    return jsonify(out)


@app.route("/api/events")
def api_events():
    return jsonify(db.get_recent_events(50))


@app.route("/api/stats")
def api_stats():
    total_persons = sum(len(p.tracked_persons) for p in processors.values())
    total_vehicles = sum(len(p.tracked_vehicles) for p in processors.values())
    db_stats = db.get_stats()
    return jsonify({
        "persons": total_persons,
        "vehicles": total_vehicles,
        "alerts": db_stats.get("alerts", 0),
        "intrusions": db_stats.get("intrusions", 0),
        "loitering": db_stats.get("loitering", 0),
    })



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
