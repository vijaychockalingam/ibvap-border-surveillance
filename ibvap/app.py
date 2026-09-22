import os
import sys
import time
import warnings
from flask import Flask, Response, jsonify, render_template, request

# Suppress PyTorch and OpenCV noisy console deprecation warnings
warnings.filterwarnings("ignore")
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"
os.environ["PYTHONWARNINGS"] = "ignore"

import config
import database as db
from cameras import load_cameras
from stream_manager import StreamManager
from incident_engine import IncidentEngine
from ai_node import AINode

app = Flask(__name__)

CAMERAS = load_cameras()
db.init_db()

# Initialize Stream Manager
stream_manager = StreamManager(CAMERAS)
stream_manager.start_all()

# Initialize Incident Engine
incident_engine = IncidentEngine()

# Initialize AI Nodes
ai_nodes = {}
for node_id, node_config in config.NODES_CONFIG.items():
    node = AINode(node_id, node_config, stream_manager, CAMERAS, incident_engine)
    node.start()
    ai_nodes[node_id] = node

_STREAM_INTERVAL = 1.0 / config.STREAM_FPS

def get_node_for_camera(camera_id):
    cam_config = CAMERAS.get(camera_id)
    if cam_config:
        node_id = cam_config.get("processing_node")
        return ai_nodes.get(node_id)
    return None

@app.route("/")
def index():
    return render_template("index.html", cameras=CAMERAS, nodes=config.NODES_CONFIG)

@app.route("/snapshot/<camera_id>")
def snapshot(camera_id):
    node = get_node_for_camera(camera_id)
    if node is None:
        return "Camera or Node not found", 404
    frame_bytes = node.get_jpeg(camera_id)
    if frame_bytes is None:
        return "", 204
    return Response(frame_bytes, mimetype="image/jpeg")

@app.route("/video_feed/<camera_id>")
def video_feed(camera_id):
    node = get_node_for_camera(camera_id)
    if node is None:
        return "Camera unavailable", 404
    def gen():
        try:
            while True:
                frame_bytes = node.get_jpeg(camera_id)
                if frame_bytes is not None:
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
                time.sleep(_STREAM_INTERVAL)
        except (GeneratorExit, Exception):
            pass
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/cameras")
def api_cameras():
    out = {}
    stream_statuses = stream_manager.get_all_statuses()
    for cid, cam in CAMERAS.items():
        node = get_node_for_camera(cid)
        stream_status = stream_statuses.get(cid, {})
        online = stream_status.get("online", False)
        
        last_event = node.get_last_event(cid) if node else None
        out[cid] = {
            "name": cam.get("name", cid),
            "location": cam.get("location", "Perimeter"),
            "node_id": cam.get("processing_node"),
            "online": online,
            "status": stream_status.get("status", "OFFLINE"),
            "health_warnings": stream_status.get("health_warnings", []),
            "last_event_type": last_event[0] if last_event else None,
            "last_event_severity": last_event[1] if last_event else None,
            "last_event_time": last_event[2] if last_event else None,
        }
    return jsonify(out)

@app.route("/api/nodes")
def api_nodes():
    return jsonify({nid: n.get_node_status() for nid, n in ai_nodes.items()})

@app.route("/api/health")
def api_health():
    return jsonify({
        "status": "ok",
        "nodes": {nid: n.is_alive() for nid, n in ai_nodes.items()}
    })

@app.route("/api/events")
def api_events():
    return jsonify(db.get_recent_events(50))

@app.route("/api/incidents")
def api_incidents():
    return jsonify(db.get_active_incidents(20))

@app.route("/api/incidents/<incident_id>")
def api_incident_detail(incident_id):
    inc = db.get_incident(incident_id)
    if inc:
        return jsonify(inc)
    return jsonify({"error": "Not found"}), 404

@app.route("/api/incidents/<incident_id>/close", methods=["POST"])
def api_incident_close(incident_id):
    db.update_incident(incident_id, status="CLOSED")
    return jsonify({"status": "closed"})

@app.route("/api/stats")
def api_stats():
    total_persons = 0
    total_vehicles = 0
    for node in ai_nodes.values():
        for cid, ctx in node.cameras.items():
            total_persons += len(ctx.tracked_persons)
            total_vehicles += len(ctx.tracked_vehicles)
            
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
