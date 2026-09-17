import sqlite3
from datetime import datetime

DB_PATH = "ibvap.db"


def get_connection():
    conn = sqlite3.connect(DB_PATH, timeout=15.0)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


def init_db():
    conn = get_connection()
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            camera_id TEXT,
            camera_name TEXT,
            event_type TEXT,
            object_type TEXT,
            object_id INTEGER,
            timestamp TEXT,
            confidence REAL,
            snapshot_path TEXT,
            severity TEXT,
            plate_number TEXT,
            status TEXT DEFAULT 'UNRESOLVED'
        )
    """)

    # Auto-migrate a DB created by an earlier version of IBVAP, which
    # didn't have these columns yet.
    c.execute("PRAGMA table_info(events)")
    existing_cols = {row[1] for row in c.fetchall()}
    if "camera_id" not in existing_cols:
        c.execute("ALTER TABLE events ADD COLUMN camera_id TEXT")
    if "camera_name" not in existing_cols:
        c.execute("ALTER TABLE events ADD COLUMN camera_name TEXT")
    if "plate_number" not in existing_cols:
        c.execute("ALTER TABLE events ADD COLUMN plate_number TEXT")

    conn.commit()
    conn.close()


def log_event(camera_id, camera_name, event_type, object_type, object_id,
               confidence, snapshot_path, severity, plate_number=None):
    try:
        conn = get_connection()
        c = conn.cursor()
        c.execute("""
            INSERT INTO events (camera_id, camera_name, event_type, object_type, object_id,
                                 timestamp, confidence, snapshot_path, severity, plate_number)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            camera_id, camera_name, event_type, object_type, object_id,
            datetime.now().isoformat(timespec="seconds"),
            confidence, snapshot_path, severity, plate_number,
        ))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[DB ERROR] Failed to log event: {e}")


def make_summary(row):
    event_type = row.get("event_type") or "EVENT"
    obj_type = (row.get("object_type") or "object").lower()
    obj_id = row.get("object_id", "")
    cam_name = row.get("camera_name") or row.get("camera_id") or "Camera"
    plate = row.get("plate_number")

    if event_type == "INTRUSION":
        return f"Unauthorized {obj_type} #{obj_id} entered Restricted Zone at {cam_name}"
    elif event_type == "LOITERING":
        return f"Suspicious loitering: {obj_type} #{obj_id} lingering inside restricted perimeter"
    elif event_type == "WATCHLIST_MATCH":
        return f"CRITICAL MATCH: Flagged plate [{plate}] detected on vehicle #{obj_id}"
    elif event_type == "NIGHT_MOVEMENT":
        return f"Night-time activity: {obj_type} #{obj_id} detected under low-light surveillance"
    elif event_type == "VEHICLE_DETECTED":
        plate_str = f" with plate [{plate}]" if plate else ""
        return f"Vehicle #{obj_id} tracked moving past {cam_name}{plate_str}"
    elif event_type == "PERSON_DETECTED":
        return f"Pedestrian #{obj_id} detected in {cam_name} field of view"
    elif event_type == "FACE_DETECTED":
        return f"Facial feature detected for person #{obj_id}"
    return f"{event_type} - {obj_type} #{obj_id} at {cam_name}"


def get_recent_events(limit=50):
    try:
        conn = get_connection()
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute("SELECT * FROM events ORDER BY id DESC LIMIT ?", (limit,))
        rows = []
        for r in c.fetchall():
            d = dict(r)
            d["summary"] = make_summary(d)
            rows.append(d)
        conn.close()
        return rows
    except Exception as e:
        print(f"[DB ERROR] get_recent_events failed: {e}")
        return []


def get_stats():
    try:
        conn = get_connection()
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM events WHERE object_type='person'")
        persons = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM events WHERE object_type='vehicle'")
        vehicles = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM events WHERE severity IN ('HIGH','CRITICAL')")
        alerts = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM events WHERE event_type='INTRUSION'")
        intrusions = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM events WHERE event_type='LOITERING'")
        loitering = c.fetchone()[0]
        conn.close()
        return {
            "persons": persons,
            "vehicles": vehicles,
            "alerts": alerts,
            "intrusions": intrusions,
            "loitering": loitering,
        }
    except Exception as e:
        print(f"[DB ERROR] get_stats failed: {e}")
        return {"persons": 0, "vehicles": 0, "alerts": 0, "intrusions": 0, "loitering": 0}


