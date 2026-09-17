import json
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


def _resolve_source(video_source):
    """Lets cameras.json use "0" for a webcam, or a file path, or an
    rtsp://... / http://... URL for a real live camera - all as plain
    strings in the config file."""
    if isinstance(video_source, str) and video_source.isdigit():
        return int(video_source)
    return video_source


def _load_watchlist(path):
    if not os.path.exists(path):
        return set()
    with open(path) as f:
        data = json.load(f)
    return {p.upper().replace(" ", "") for p in data.get("plates", [])}


class IBVAPProcessor:
    """
    Owns one camera end-to-end: reads frames, runs detection/tracking/zone/
    ANPR/face/loitering checks, and continuously writes the latest annotated
    JPEG into a thread-safe buffer via start(). Runs in its own background
    thread so all cameras can run genuinely simultaneously - the Flask
    routes just read whatever's currently in the buffer, they don't drive
    the capture themselves. This also means there's no more "only one
    camera active" switching logic, and no more races from releasing a
    capture out from under an in-progress read - each processor owns its
    capture for its whole lifetime.
    """

    def __init__(self, video_source, zone_path="zone_config.json", loop=True,
                 camera_id="cam1", camera_name="Camera 1"):
        self._resolved_source = _resolve_source(video_source)
        self.cap = cv2.VideoCapture(self._resolved_source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video source: {video_source}")

        self.camera_id = camera_id
        self.camera_name = camera_name
        self.loop = loop
        self.detector = Detector()
        self.tracker = CentroidTracker()
        self.zone_polygon = load_zone(zone_path)
        self.watchlist = _load_watchlist(config.WATCHLIST_PATH)
        self.last_alert_time = {}    # object_id -> last alert timestamp
        self.anpr_state = {}         # object_id -> {"attempts", "plate", "logged"}
        self.face_state = {}         # object_id -> {"attempts", "found", "logged"}
        self.zone_entry_time = {}    # object_id -> time first seen continuously in zone
        self.loitering_logged = set()  # object_ids already logged as loitering
        self.frame_count = 0
        db.init_db()

        self._jpeg = None
        self._jpeg_lock = threading.Lock()
        self._last_event = None      # most recent (event_type, severity, time) for this camera
        self._stop_flag = threading.Event()
        self._thread = None
        self._healthy = True         # False once retries are exhausted - reflected in /api/cameras

        self.tracked_persons = set()
        self.tracked_vehicles = set()
        self.logged_initial = set()  # object_ids whose first detection has been logged
        self.intrusion_logged = set() # object_ids logged as intrusion

        if self.zone_polygon is None:
            print(f"[IBVAP] WARNING: no zone file found for {camera_name} ({zone_path}) - "
                  f"run select_zone.py first, otherwise intrusion detection is disabled.")


    # --- lifecycle -----------------------------------------------------

    def start(self):
        """Launches the capture+detect+encode loop in a background thread."""
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_flag.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        self.cap.release()

    def is_alive(self):
        """True if this camera's thread is running and hasn't given up
        after repeated crashes. Reflects real current status, not just
        whether start() was ever called successfully."""
        return self._healthy and self._thread is not None and self._thread.is_alive()

    def get_latest_jpeg(self):
        with self._jpeg_lock:
            return self._jpeg

    def get_last_event(self):
        """(event_type, severity, unix_time) of the most recent alert-worthy
        event on this camera, or None. Used to drive the per-tile alert
        glow on the dashboard without hitting the DB on every poll."""
        return self._last_event

    def _run_loop(self):
        max_retries = 5
        retries = 0
        while not self._stop_flag.is_set():
            try:
                for jpeg_bytes in self._frames():
                    with self._jpeg_lock:
                        self._jpeg = jpeg_bytes
                    if self._stop_flag.is_set():
                        return
                    retries = 0  # a successful frame resets the retry counter
                return  # _frames() ended normally (loop=False and video finished)
            except Exception:
                retries += 1
                print(f"[IBVAP] ERROR: {self.camera_name} ({self.camera_id}) crashed "
                      f"(attempt {retries}/{max_retries}):")
                traceback.print_exc()

                if retries >= max_retries:
                    self._healthy = False
                    print(f"[IBVAP] {self.camera_name} ({self.camera_id}) gave up after "
                          f"{max_retries} crashes - marking offline. This usually means "
                          f"something is wrong with that specific video file/stream.")
                    return

                time.sleep(1)
                try:
                    self.cap.release()
                except Exception:
                    pass
                self.cap = cv2.VideoCapture(self._resolved_source)
                if not self.cap.isOpened():
                    print(f"[IBVAP] {self.camera_name} ({self.camera_id}) could not "
                          f"reopen its video source, retrying...")

    # --- per-frame pipeline ---------------------------------------------

    def _is_night(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return float(np.mean(gray)) < config.NIGHT_BRIGHTNESS_THRESHOLD

    def _can_alert(self, object_id):
        now = time.time()
        if now - self.last_alert_time.get(object_id, 0) >= config.ALERT_COOLDOWN_SECONDS:
            self.last_alert_time[object_id] = now
            return True
        return False

    def _save_snapshot(self, frame, event_type, object_id):
        filename = f"{event_type}_{object_id}_{int(time.time())}.jpg"
        path = os.path.join(SNAPSHOT_DIR, filename)
        cv2.imwrite(path, frame)
        return path

    def _log(self, event_type, obj_type, object_id, confidence, frame, severity, plate_number=None):
        snapshot_path = self._save_snapshot(frame, event_type, object_id)
        db.log_event(self.camera_id, self.camera_name, event_type, obj_type,
                     object_id, confidence, snapshot_path, severity, plate_number=plate_number)
        self._last_event = (event_type, severity, time.time())

    def _try_anpr(self, object_id, frame, bbox):
        state = self.anpr_state.setdefault(object_id, {"attempts": 0, "plate": None, "logged_watchlist": False})
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

    def _try_face(self, object_id, frame, bbox):
        if not config.ENABLE_FACE_DETECTION:
            return False
        state = self.face_state.setdefault(object_id, {"attempts": 0, "found": False, "logged": False})
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

    def _maybe_log_face(self, object_id, frame):
        state = self.face_state.get(object_id, {})
        if state.get("found") and not state.get("logged"):
            self._log("FACE_DETECTED", "person", object_id, 0.9, frame, "LOW")
            state["logged"] = True

    def _check_loitering(self, object_id, in_zone, frame, label, confidence):
        now = time.time()
        if not in_zone:
            self.zone_entry_time.pop(object_id, None)
            self.loitering_logged.discard(object_id)
            return None

        first_seen = self.zone_entry_time.setdefault(object_id, now)
        dwell = now - first_seen
        if dwell >= config.LOITERING_SECONDS and object_id not in self.loitering_logged:
            self._log("LOITERING", "person", object_id, confidence, frame, "HIGH")
            self.loitering_logged.add(object_id)
            return f"LOITERING - {label} #{object_id} ({int(dwell)}s in zone)"
        return None

    def _frames(self):
        while True:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                if self.loop:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = self.cap.read()
                    if not ret or frame is None:
                        try:
                            self.cap.release()
                        except Exception:
                            pass
                        self.cap = cv2.VideoCapture(self._resolved_source)
                        ret, frame = self.cap.read()
                        if not ret or frame is None:
                            time.sleep(0.05)
                            continue
                else:
                    return


            self.frame_count += 1
            run_detection = (self.frame_count % config.FRAME_SKIP == 0)

            h, w = frame.shape[:2]
            scale = config.RESIZE_WIDTH / w
            frame = cv2.resize(frame, (config.RESIZE_WIDTH, int(h * scale)))

            is_night = self._is_night(frame)

            if run_detection:
                try:
                    detections = self.detector.detect(frame)
                    tracked = self.tracker.update(detections)
                except Exception as e:
                    print(f"[IBVAP] WARNING: detection error on {self.camera_name}, skipping frame: {e}")
                    tracked = self.tracker.objects
            else:
                tracked = self.tracker.objects

            cv2.putText(frame, self.camera_name, (10, frame.shape[0] - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            if self.zone_polygon:
                pts = np.array(self.zone_polygon, dtype=np.int32)
                cv2.polylines(frame, [pts], True, (0, 255, 255), 2)
                cv2.putText(frame, "RESTRICTED ZONE", tuple(self.zone_polygon[0]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            alert_banner = None

            for obj_id, obj in tracked.items():
                x1, y1, x2, y2 = obj["bbox"]
                centroid = obj["centroid"]
                label, obj_type, confidence = obj["label"], obj["type"], obj["confidence"]

                if obj_type == "person":
                    self.tracked_persons.add(obj_id)
                elif obj_type == "vehicle":
                    self.tracked_vehicles.add(obj_id)

                in_zone = point_in_zone(centroid, self.zone_polygon)

                plate = None
                if obj_type == "vehicle" and run_detection:
                    plate = self._try_anpr(obj_id, frame, (x1, y1, x2, y2))
                elif obj_type == "vehicle":
                    plate = self.anpr_state.get(obj_id, {}).get("plate")

                on_watchlist = bool(plate) and (plate in self.watchlist)

                # Determine display colors and status
                if on_watchlist:
                    color = (0, 0, 255)
                    event_type = "WATCHLIST_MATCH"
                    severity = "CRITICAL"
                elif in_zone:
                    color = (0, 0, 255)
                    event_type = "INTRUSION"
                    severity = "HIGH"
                elif obj_type == "person" and is_night:
                    color = (0, 165, 255)
                    event_type = "NIGHT_MOVEMENT"
                    severity = "MEDIUM"
                elif obj_type == "person":
                    color = (0, 200, 0)
                    event_type = None
                    severity = "LOW"
                else:  # vehicle
                    color = (255, 200, 0)
                    event_type = None
                    severity = "LOW"

                display_label = f"{label} #{obj_id}"
                if plate:
                    display_label += f" [{plate}]"

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, display_label, (x1, max(15, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Face detection for persons
                if obj_type == "person" and run_detection:
                    face_found = self._try_face(obj_id, frame, (x1, y1, x2, y2))
                    if face_found:
                        cv2.putText(frame, "FACE", (x1, min(frame.shape[0] - 5, y2 + 14)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)

                # Check loitering for persons in zone
                if obj_type == "person":
                    loiter_banner = self._check_loitering(obj_id, in_zone, frame, label, confidence)
                    if loiter_banner:
                        alert_banner = loiter_banner

                # Event-based recording: Only log actionable security alerts to DB and disk
                if in_zone and obj_id not in self.intrusion_logged and self._can_alert(obj_id):
                    self._log("INTRUSION", obj_type, obj_id, confidence, frame, "HIGH", plate_number=plate)
                    self.intrusion_logged.add(obj_id)
                    if not alert_banner:
                        alert_banner = f"INTRUSION - {label} #{obj_id}"

                elif on_watchlist and not self.anpr_state.get(obj_id, {}).get("logged_watchlist"):
                    self._log("WATCHLIST_MATCH", obj_type, obj_id, confidence, frame, "CRITICAL", plate_number=plate)
                    self.anpr_state[obj_id]["logged_watchlist"] = True
                    alert_banner = f"WATCHLIST MATCH - {label} #{obj_id} [{plate}]"

                elif obj_type == "person" and is_night and obj_id not in self.logged_initial and self._can_alert(obj_id):
                    self._log("NIGHT_MOVEMENT", obj_type, obj_id, confidence, frame, "MEDIUM")
                    self.logged_initial.add(obj_id)
                    if not alert_banner:
                        alert_banner = f"NIGHT_MOVEMENT - {label} #{obj_id}"

                if (in_zone or on_watchlist) and not alert_banner:
                    alert_banner = f"{event_type} - {label} #{obj_id}"


            if is_night:
                cv2.putText(frame, "NIGHT MODE", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            if alert_banner:
                cv2.rectangle(frame, (0, frame.shape[0] - 30), (frame.shape[1], frame.shape[0]), (0, 0, 255), -1)
                cv2.putText(frame, alert_banner, (10, frame.shape[0] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

            ok, buffer = cv2.imencode(".jpg", frame)
            if not ok:
                continue
            yield buffer.tobytes()

