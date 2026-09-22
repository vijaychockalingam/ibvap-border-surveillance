"""
IBVAP Logical AI Processing Node
Responsible for:
- Managing AI/CV analytics for its assigned subset of cameras.
- Local YOLOv8 inference (offline capable).
- Independent per-camera object tracking (CentroidTracker instances).
- Face detection, ANPR, Zones.
- Failure-isolated worker execution.
- *NEW*: Adaptive FPS and Integration with Incident Engine.
"""
import os
import threading
import time
import traceback
import cv2
import numpy as np

import anpr
import config
import database as db
import face
from detector import Detector
from tracker import CentroidTracker
from zone import load_zone, point_in_zone

SNAPSHOT_DIR = "snapshots"
os.makedirs(SNAPSHOT_DIR, exist_ok=True)


def _load_watchlist(path):
    if not os.path.exists(path):
        return set()
    try:
        import json
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {p.upper().replace(" ", "") for p in data.get("plates", [])}
    except Exception:
        return set()


class CameraAIContext:
    def __init__(self, camera_id, config_dict, node_id):
        self.camera_id = camera_id
        self.config = config_dict
        self.name = config_dict.get("name", camera_id)
        self.location = config_dict.get("location", "Perimeter")
        self.node_id = node_id
        
        zone_path = config_dict.get("zone_config", f"zone_{camera_id}.json")
        self.zone_polygon = load_zone(zone_path)
        
        self.tracker = CentroidTracker()
        
        self.last_alert_time = {}       
        self.anpr_state = {}            
        self.face_state = {}            
        self.zone_entry_time = {}       
        self.loitering_logged = set()   
        self.logged_initial = set()     
        self.intrusion_logged = set()   
        
        self.tracked_persons = set()
        self.tracked_vehicles = set()
        
        self.last_event = None          
        
        self.latest_jpeg = None
        self.jpeg_lock = threading.Lock()
        
        self.frame_count = 0
        self.ai_inference_count = 0
        self.processing_fps = 0.0

    def get_jpeg(self):
        with self.jpeg_lock:
            return self.latest_jpeg

    def set_jpeg(self, jpeg_bytes):
        with self.jpeg_lock:
            self.latest_jpeg = jpeg_bytes

    def get_last_event(self):
        return self.last_event


class AINode:
    def __init__(self, node_id, node_config, stream_manager, camera_configs, incident_engine):
        self.node_id = node_id
        self.name = node_config.get("name", f"AI Node {node_id}")
        self.description = node_config.get("description", "")
        self.stream_manager = stream_manager
        self.incident_engine = incident_engine
        
        self.cameras = {}
        for cid, cfg in camera_configs.items():
            if cfg.get("processing_node") == node_id and cfg.get("enabled", True):
                self.cameras[cid] = CameraAIContext(cid, cfg, self.node_id)
                
        self.detector = Detector(model_path=config.MODEL_PATH, conf_threshold=config.CONF_THRESHOLD)
        self.watchlist = _load_watchlist(config.WATCHLIST_PATH)
        
        self.stop_event = threading.Event()
        self.thread = None
        self.status = "INITIALIZING"
        self.fps = 0.0

    def start(self):
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._node_worker, name=f"AINodeWorker-{self.node_id}", daemon=True)
        self.thread.start()
        self.status = "RUNNING"
        print(f"[AINode:{self.node_id}] Started with {len(self.cameras)} cameras.")

    def stop(self):
        self.stop_event.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        self.status = "STOPPED"

    def is_alive(self):
        return self.thread is not None and self.thread.is_alive()

    def _node_worker(self):
        cycle_count = 0
        start_time = time.time()

        while not self.stop_event.is_set():
            cycle_start = time.time()
            
            active_objects = sum(len(ctx.tracker.objects) for ctx in self.cameras.values())
            if active_objects > 0:
                target_interval = 1.0 / config.ACTIVE_FPS
            else:
                target_interval = 1.0 / config.IDLE_FPS
            
            for cid, ctx in list(self.cameras.items()):
                try:
                    self._process_camera_cycle(cid, ctx)
                except Exception as e:
                    print(f"[AINode:{self.node_id}] Error in camera {cid} pipeline: {e}")
                    traceback.print_exc()

            cycle_count += 1
            elapsed = time.time() - start_time
            if elapsed >= 3.0:
                self.fps = round(cycle_count / elapsed, 1)
                start_time = time.time()
                cycle_count = 0

            spent = time.time() - cycle_start
            sleep_time = max(0.01, target_interval - spent)
            time.sleep(sleep_time)

    def _process_camera_cycle(self, cid, ctx):
        raw_frame, frame_time = self.stream_manager.get_frame(cid)
        if raw_frame is None:
            return

        ctx.frame_count += 1
        h, w = raw_frame.shape[:2]
        scale = config.RESIZE_WIDTH / w
        frame = cv2.resize(raw_frame, (config.RESIZE_WIDTH, int(h * scale)))

        is_night = float(np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))) < config.NIGHT_BRIGHTNESS_THRESHOLD

        run_detection = (ctx.frame_count % config.FRAME_SKIP == 0)
        if run_detection:
            try:
                detections = self.detector.detect(frame)
                tracked = ctx.tracker.update(detections)
                ctx.ai_inference_count += 1
            except Exception as e:
                print(f"[AINode:{self.node_id}] Detection error on {cid}: {e}")
                tracked = ctx.tracker.objects
        else:
            tracked = ctx.tracker.objects

        node_label = f"[{self.node_id.upper()}] {ctx.name} - {ctx.location}"
        cv2.putText(frame, node_label, (10, frame.shape[0] - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)

        if ctx.zone_polygon:
            pts = np.array(ctx.zone_polygon, dtype=np.int32)
            cv2.polylines(frame, [pts], True, (0, 255, 255), 2)
            cv2.putText(frame, "RESTRICTED ZONE", tuple(ctx.zone_polygon[0]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        alert_banner = None

        for obj_id, obj in tracked.items():
            x1, y1, x2, y2 = obj["bbox"]
            centroid = obj["centroid"]
            label, obj_type, confidence = obj["label"], obj["type"], obj["confidence"]

            if obj_type == "person":
                ctx.tracked_persons.add(obj_id)
            elif obj_type == "vehicle":
                ctx.tracked_vehicles.add(obj_id)

            in_zone = point_in_zone(centroid, ctx.zone_polygon)

            plate = None
            if obj_type == "vehicle" and run_detection:
                plate = self._try_anpr(ctx, obj_id, frame, (x1, y1, x2, y2))
            elif obj_type == "vehicle":
                plate = ctx.anpr_state.get(obj_id, {}).get("plate")

            on_watchlist = bool(plate) and (plate in self.watchlist)

            if on_watchlist:
                color = (0, 0, 255)
                event_type = "WATCHLIST_MATCH"
            elif obj_type == "animal":
                color = (200, 200, 200) # Gray for environmental filters
                event_type = "ANIMAL_DETECTED"
            elif in_zone:
                color = (0, 0, 255)
                event_type = "INTRUSION"
            elif obj_type == "person" and is_night:
                color = (0, 165, 255)
                event_type = "NIGHT_MOVEMENT"
            elif obj_type == "person":
                color = (0, 200, 0)
                event_type = "PERSON_DETECTED"
            else:
                color = (255, 200, 0)
                event_type = "VEHICLE_DETECTED"

            display_label = f"{label} #{obj_id}"
            if plate:
                display_label += f" [{plate}]"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, display_label, (x1, max(15, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            if obj_type == "person" and run_detection:
                face_found = self._try_face(ctx, obj_id, frame, (x1, y1, x2, y2))
                if face_found:
                    cv2.putText(frame, "FACE", (x1, min(frame.shape[0] - 5, y2 + 14)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)

            if obj_type == "person":
                loiter_banner = self._check_loitering(ctx, obj_id, in_zone, frame, label, confidence)
                if loiter_banner:
                    event_type = "LOITERING"
                    alert_banner = loiter_banner

            # Only trigger IncidentEngine if cool down has passed for this specific obj_id
            if self._can_alert(ctx, obj_id, obj_type, in_zone, event_type, is_night):
                snapshot_path = None
                if event_type in ["INTRUSION", "LOITERING", "WATCHLIST_MATCH", "NIGHT_MOVEMENT"]:
                    snapshot_path = self._save_snapshot(frame, event_type, obj_id)
                
                # Send to incident engine
                incident_id, severity, desc = self.incident_engine.process_event(
                    camera_id=cid,
                    camera_name=ctx.name,
                    event_type=event_type,
                    object_type=obj_type,
                    local_obj_id=obj_id,
                    confidence=confidence,
                    snapshot_path=snapshot_path,
                    plate_number=plate,
                    is_night=is_night,
                    in_zone=in_zone
                )
                
                ctx.last_event = (event_type, severity, time.time(), desc)
                if severity in ["HIGH", "CRITICAL"] and not alert_banner:
                    alert_banner = desc

        if is_night:
            cv2.putText(frame, "NIGHT MODE", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        if alert_banner:
            cv2.rectangle(frame, (0, frame.shape[0] - 30), (frame.shape[1], frame.shape[0]), (0, 0, 255), -1)
            cv2.putText(frame, alert_banner, (10, frame.shape[0] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        ok, buffer = cv2.imencode(".jpg", frame)
        if ok:
            ctx.set_jpeg(buffer.tobytes())

    def _can_alert(self, ctx, object_id, obj_type, in_zone, event_type, is_night):
        now = time.time()
        
        if event_type == "ANIMAL_DETECTED":
            # Very aggressive cooldown for animals to avoid log spam
            cooldown = config.ALERT_COOLDOWN_SECONDS * 5
        elif event_type in ["INTRUSION", "LOITERING", "WATCHLIST_MATCH"]:
            cooldown = config.ALERT_COOLDOWN_SECONDS
        else:
            # Normal detections get a longer cooldown so we don't spam the timeline
            cooldown = config.ALERT_COOLDOWN_SECONDS * 2
            
        if now - ctx.last_alert_time.get(object_id, 0) >= cooldown:
            ctx.last_alert_time[object_id] = now
            return True
        return False

    def _save_snapshot(self, frame, event_type, object_id):
        filename = f"{event_type}_{object_id}_{int(time.time())}.jpg"
        path = os.path.join(SNAPSHOT_DIR, filename)
        cv2.imwrite(path, frame)
        return path

    def _try_anpr(self, ctx, object_id, frame, bbox):
        state = ctx.anpr_state.setdefault(object_id, {"attempts": 0, "plate": None, "logged_watchlist": False})
        if state["plate"] is None and state["attempts"] < config.ANPR_MAX_ATTEMPTS and anpr.anpr_available():
            x1, y1, x2, y2 = bbox
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if (x2 - x1) >= 50 and (y2 - y1) >= 25:
                crop = frame[y1:y2, x1:x2]
                plate, _conf = anpr.read_plate(crop)
                state["attempts"] += 1
                if plate:
                    state["plate"] = plate
        return state["plate"]

    def _try_face(self, ctx, object_id, frame, bbox):
        if not config.ENABLE_FACE_DETECTION:
            return False
        state = ctx.face_state.setdefault(object_id, {"attempts": 0, "found": False, "logged": False})
        if not state["found"] and state["attempts"] < config.FACE_MAX_ATTEMPTS:
            x1, y1, x2, y2 = bbox
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            crop = frame[y1:y2, x1:x2]
            state["attempts"] += 1
            if face.has_face(crop):
                state["found"] = True
        return state["found"]

    def _check_loitering(self, ctx, object_id, in_zone, frame, label, confidence):
        now = time.time()
        if not in_zone:
            ctx.zone_entry_time.pop(object_id, None)
            return None

        first_seen = ctx.zone_entry_time.setdefault(object_id, now)
        dwell = now - first_seen
        if dwell >= config.LOITERING_SECONDS:
            return f"LOITERING - {label} #{object_id} ({int(dwell)}s in zone)"
        return None

    def get_camera_context(self, camera_id):
        return self.cameras.get(camera_id)

    def get_jpeg(self, camera_id):
        ctx = self.cameras.get(camera_id)
        return ctx.get_jpeg() if ctx else None

    def get_last_event(self, camera_id):
        ctx = self.cameras.get(camera_id)
        return ctx.get_last_event() if ctx else None

    def get_node_status(self):
        return {
            "node_id": self.node_id,
            "name": self.name,
            "description": self.description,
            "status": self.status,
            "online": self.is_alive(),
            "assigned_cameras": list(self.cameras.keys()),
            "fps": self.fps,
        }
