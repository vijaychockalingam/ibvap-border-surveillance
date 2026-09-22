"""
IBVAP Stream Manager
Responsible for:
- Opening and managing RTSP/IP camera streams, webcams, and video files.
- Managing multiple camera connections in parallel with full failure isolation.
- Automatic background reconnection with exponential backoff on disconnect.
- Maintaining connection health, FPS, and status per camera stream.
- Delivering raw frames to AI processing nodes.
"""
import os
import threading
import time
import cv2


def resolve_video_source(source):
    """Parses integer string '0' -> 0 (webcam), or returns file path / RTSP URL."""
    if isinstance(source, str) and source.isdigit():
        return int(source)
    return source


class CameraStream:
    """Manages an individual video/RTSP stream with isolated lifecycle and auto-recovery."""

    def __init__(self, camera_id, config):
        self.camera_id = camera_id
        self.config = config
        self.name = config.get("name", camera_id)
        self.source = resolve_video_source(config.get("source", f"sample_{camera_id}.mp4"))
        self.enabled = config.get("enabled", True)
        self.location = config.get("location", "Perimeter")
        self.node_id = config.get("processing_node", "node_1")

        self.cap = None
        self.status = "INITIALIZING"  # ONLINE, RECONNECTING, OFFLINE, DISABLED, ERROR
        self.last_frame = None
        self.last_frame_time = 0
        self.frame_lock = threading.Lock()
        
        self.stop_event = threading.Event()
        self.thread = None
        
        self.fps = 0.0
        self.frame_count = 0
        self.reconnect_count = 0
        self.error_message = None
        self.health_warnings = set()
        self.last_gray = None
        self.frozen_frames = 0

    def start(self):
        if not self.enabled:
            self.status = "DISABLED"
            return
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._stream_worker, name=f"StreamWorker-{self.camera_id}", daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.5)
        self._release_cap()
        self.status = "OFFLINE"

    def _release_cap(self):
        try:
            if self.cap is not None:
                self.cap.release()
        except Exception:
            pass
        self.cap = None

    def _open_stream(self):
        self._release_cap()
        try:
            # For RTSP, set lower buffer size if supported
            if isinstance(self.source, str) and self.source.startswith("rtsp://"):
                os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"
            self.cap = cv2.VideoCapture(self.source)
            if self.cap is not None and self.cap.isOpened():
                self.status = "ONLINE"
                self.error_message = None
                return True
            else:
                self.status = "OFFLINE"
                self.error_message = f"Could not open source: {self.source}"
                return False
        except Exception as e:
            self.status = "ERROR"
            self.error_message = str(e)
            return False

    def _stream_worker(self):
        retry_delay = 1.0
        max_retry_delay = 10.0

        if not self._open_stream():
            print(f"[StreamManager] Initial connection failed for {self.name} ({self.camera_id}). Will retry in background.")

        fps_start_time = time.time()
        fps_frame_count = 0

        while not self.stop_event.is_set():
            if self.cap is None or not self.cap.isOpened():
                self.status = "RECONNECTING"
                self.reconnect_count += 1
                time.sleep(retry_delay)
                retry_delay = min(retry_delay * 1.5, max_retry_delay)
                if self._open_stream():
                    print(f"[StreamManager] Reconnected {self.name} ({self.camera_id}) successfully.")
                    retry_delay = 1.0
                continue

            try:
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    # If it's a file, loop back to start
                    if isinstance(self.source, str) and not self.source.startswith(("rtsp://", "http://")):
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret, frame = self.cap.read()
                        if not ret or frame is None:
                            self._open_stream()
                            time.sleep(0.05)
                            continue
                    else:
                        # RTSP stream dropped
                        self.status = "RECONNECTING"
                        self._release_cap()
                        time.sleep(1.0)
                        continue

                now = time.time()
                with self.frame_lock:
                    self.last_frame = frame
                    self.last_frame_time = now

                self.frame_count += 1
                fps_frame_count += 1
                self.status = "ONLINE"
                
                if self.frame_count % 30 == 0:
                    try:
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
                        brightness = gray.mean()
                        
                        if blur_score < 15.0:
                            self.health_warnings.add("BLUR")
                        else:
                            self.health_warnings.discard("BLUR")
                            
                        if brightness < 15.0:
                            self.health_warnings.add("DARK")
                        else:
                            self.health_warnings.discard("DARK")
                            
                        if self.last_gray is not None:
                            diff = cv2.absdiff(gray, self.last_gray).mean()
                            if diff < 1.0:
                                self.frozen_frames += 1
                            else:
                                self.frozen_frames = 0
                        self.last_gray = gray
                        
                        if self.frozen_frames > 3:
                            self.health_warnings.add("FROZEN")
                        else:
                            self.health_warnings.discard("FROZEN")
                            
                    except Exception as e:
                        pass

                # Calculate measured FPS
                elapsed = now - fps_start_time
                if elapsed >= 2.0:
                    self.fps = round(fps_frame_count / elapsed, 1)
                    fps_start_time = now
                    fps_frame_count = 0

                # Slight yield to avoid spinning 100% CPU on loopback files
                time.sleep(0.005)

            except Exception as e:
                self.status = "ERROR"
                self.error_message = str(e)
                print(f"[StreamManager] Stream read error on {self.name}: {e}")
                self._release_cap()
                time.sleep(1.0)

        self._release_cap()

    def get_latest_frame(self):
        """Returns (frame, timestamp) or (None, 0)."""
        with self.frame_lock:
            if self.last_frame is not None:
                return self.last_frame.copy(), self.last_frame_time
            return None, 0

    def get_status(self):
        """Returns current status summary dict."""
        now = time.time()
        is_fresh = (now - self.last_frame_time) < 3.0 if self.last_frame_time > 0 else False
        return {
            "camera_id": self.camera_id,
            "name": self.name,
            "location": self.location,
            "node_id": self.node_id,
            "enabled": self.enabled,
            "status": self.status if is_fresh or self.status != "ONLINE" else "OFFLINE",
            "online": (self.status == "ONLINE" and is_fresh),
            "fps": self.fps,
            "frame_count": self.frame_count,
            "reconnect_count": self.reconnect_count,
            "error": self.error_message,
            "health_warnings": list(self.health_warnings)
        }


class StreamManager:
    """Manages collection of all CameraStream instances."""

    def __init__(self, cameras_config=None):
        self.streams = {}
        self.lock = threading.Lock()
        if cameras_config:
            self.load_from_config(cameras_config)

    def load_from_config(self, cameras_config):
        with self.lock:
            for cid, cfg in cameras_config.items():
                self.streams[cid] = CameraStream(cid, cfg)

    def start_all(self):
        print(f"[StreamManager] Starting {len(self.streams)} camera streams...")
        for cid, stream in self.streams.items():
            stream.start()

    def stop_all(self):
        print("[StreamManager] Stopping all camera streams...")
        for cid, stream in self.streams.items():
            stream.stop()

    def get_stream(self, camera_id):
        return self.streams.get(camera_id)

    def get_frame(self, camera_id):
        stream = self.streams.get(camera_id)
        if stream:
            return stream.get_latest_frame()
        return None, 0

    def get_all_statuses(self):
        return {cid: s.get_status() for cid, s in self.streams.items()}
