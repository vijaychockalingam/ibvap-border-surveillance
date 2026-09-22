import sqlite3
import json
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
            node_id TEXT,
            location TEXT,
            event_type TEXT,
            object_type TEXT,
            object_id INTEGER,
            timestamp TEXT,
            confidence REAL,
            snapshot_path TEXT,
            severity TEXT,
            plate_number TEXT,
            description TEXT,
            status TEXT DEFAULT 'UNRESOLVED'
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS incidents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            incident_id TEXT UNIQUE,
            severity TEXT,
            status TEXT DEFAULT 'ACTIVE',
            created_at TEXT,
            closed_at TEXT,
            primary_event_type TEXT,
            primary_camera_id TEXT,
            description TEXT,
            reasons TEXT, 
            timeline TEXT
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS incident_evidence (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            incident_id TEXT,
            camera_id TEXT,
            snapshot_path TEXT,
            timestamp TEXT
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS entities (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entity_id TEXT UNIQUE,
            object_type TEXT,
            first_seen TEXT,
            last_seen TEXT,
            status TEXT
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS entity_movements (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entity_id TEXT,
            camera_id TEXT,
            zone TEXT,
            direction TEXT,
            timestamp TEXT,
            confidence REAL
        )
    """)

    # Auto-migrate any existing DB tables with missing columns for older events table
    c.execute("PRAGMA table_info(events)")
    existing_cols = {row[1] for row in c.fetchall()}
    columns_to_add = {
        "camera_id": "TEXT",
        "camera_name": "TEXT",
        "node_id": "TEXT",
        "location": "TEXT",
        "plate_number": "TEXT",
        "description": "TEXT",
        "status": "TEXT DEFAULT 'UNRESOLVED'",
    }
    for col_name, col_type in columns_to_add.items():
        if col_name not in existing_cols:
            try:
                c.execute(f"ALTER TABLE events ADD COLUMN {col_name} {col_type}")
            except Exception as e:
                print(f"[DB MIGRATE] Column {col_name} migration note: {e}")

    conn.commit()
    conn.close()


def make_summary(row):
    event_type = row.get("event_type") or "EVENT"
    obj_type = (row.get("object_type") or "object").lower()
    obj_id = row.get("object_id", "")
    cam_name = row.get("camera_name") or row.get("camera_id") or "Camera"
    location = row.get("location") or ""
    loc_str = f" at {location}" if location else f" at {cam_name}"
    plate = row.get("plate_number")

    if event_type == "INTRUSION":
        return f"Unauthorized {obj_type} #{obj_id} entered Restricted Zone{loc_str}"
    elif event_type == "LOITERING":
        return f"Suspicious loitering: {obj_type} #{obj_id} lingering in restricted perimeter{loc_str}"
    elif event_type == "WATCHLIST_MATCH":
        return f"CRITICAL MATCH: Flagged vehicle plate [{plate}] detected on vehicle #{obj_id}{loc_str}"
    elif event_type == "NIGHT_MOVEMENT":
        return f"Night-time movement: {obj_type} #{obj_id} detected under low-light surveillance{loc_str}"
    elif event_type == "VEHICLE_DETECTED":
        plate_str = f" [Plate: {plate}]" if plate else ""
        return f"Vehicle #{obj_id} tracked moving past {cam_name}{plate_str}"
    elif event_type == "PERSON_DETECTED":
        return f"Pedestrian #{obj_id} detected in {cam_name} field of view"
    elif event_type == "FACE_DETECTED":
        return f"Facial feature detected for person #{obj_id}{loc_str}"
    elif event_type == "ANPR":
        return f"ANPR capture: Plate [{plate}] recorded for vehicle #{obj_id}{loc_str}"
    return f"{event_type} - {obj_type} #{obj_id}{loc_str}"


def log_event(camera_id, camera_name, event_type, object_type, object_id,
              confidence, snapshot_path, severity, plate_number=None,
              node_id=None, location=None, description=None):
    try:
        conn = get_connection()
        c = conn.cursor()
        if not description:
            description = make_summary({
                "event_type": event_type,
                "object_type": object_type,
                "object_id": object_id,
                "camera_name": camera_name,
                "camera_id": camera_id,
                "location": location,
                "plate_number": plate_number,
            })

        c.execute("""
            INSERT INTO events (camera_id, camera_name, node_id, location, event_type, object_type,
                                 object_id, timestamp, confidence, snapshot_path, severity, plate_number, description)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            camera_id, camera_name, node_id, location, event_type, object_type,
            object_id, datetime.now().isoformat(timespec="seconds"),
            confidence, snapshot_path, severity, plate_number, description
        ))
        event_db_id = c.lastrowid
        conn.commit()
        conn.close()
        return event_db_id
    except Exception as e:
        print(f"[DB ERROR] Failed to log event: {e}")
        return None

def get_recent_events(limit=50, camera_id=None, event_type=None, severity=None):
    try:
        conn = get_connection()
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        
        query = "SELECT * FROM events WHERE 1=1"
        params = []
        if camera_id:
            query += " AND camera_id = ?"
            params.append(camera_id)
        if event_type:
            query += " AND event_type = ?"
            params.append(event_type)
        if severity:
            query += " AND severity = ?"
            params.append(severity)
            
        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        
        c.execute(query, tuple(params))
        rows = []
        for r in c.fetchall():
            d = dict(r)
            if not d.get("description"):
                d["description"] = make_summary(d)
            d["summary"] = d["description"]
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
        c.execute("SELECT COUNT(*) FROM events WHERE event_type IN ('WATCHLIST_MATCH', 'ANPR')")
        anpr_count = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM events WHERE event_type='NIGHT_MOVEMENT'")
        night_movement = c.fetchone()[0]
        conn.close()
        return {
            "persons": persons,
            "vehicles": vehicles,
            "alerts": alerts,
            "intrusions": intrusions,
            "loitering": loitering,
            "anpr_events": anpr_count,
            "night_movement": night_movement,
        }
    except Exception as e:
        print(f"[DB ERROR] get_stats failed: {e}")
        return {
            "persons": 0, "vehicles": 0, "alerts": 0, "intrusions": 0,
            "loitering": 0, "anpr_events": 0, "night_movement": 0,
        }

# --- Incident Engine DB Methods ---

def create_incident(incident_id, severity, primary_event_type, primary_camera_id, description, reasons, timeline):
    try:
        conn = get_connection()
        c = conn.cursor()
        c.execute("""
            INSERT INTO incidents (incident_id, severity, created_at, primary_event_type, primary_camera_id, description, reasons, timeline)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            incident_id, severity, datetime.now().isoformat(timespec="seconds"),
            primary_event_type, primary_camera_id, description, json.dumps(reasons), json.dumps(timeline)
        ))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[DB ERROR] create_incident failed: {e}")

def update_incident(incident_id, status=None, timeline=None, new_evidence=None):
    try:
        conn = get_connection()
        c = conn.cursor()
        
        if status:
            if status == 'CLOSED':
                c.execute("UPDATE incidents SET status=?, closed_at=? WHERE incident_id=?", 
                          (status, datetime.now().isoformat(timespec="seconds"), incident_id))
            else:
                c.execute("UPDATE incidents SET status=? WHERE incident_id=?", (status, incident_id))
        
        if timeline:
            # We fetch current timeline, append, and save
            c.execute("SELECT timeline FROM incidents WHERE incident_id=?", (incident_id,))
            row = c.fetchone()
            if row:
                current_timeline = json.loads(row[0]) if row[0] else []
                current_timeline.extend(timeline)
                c.execute("UPDATE incidents SET timeline=? WHERE incident_id=?", (json.dumps(current_timeline), incident_id))
        
        if new_evidence:
            for ev in new_evidence:
                c.execute("""
                    INSERT INTO incident_evidence (incident_id, camera_id, snapshot_path, timestamp)
                    VALUES (?, ?, ?, ?)
                """, (incident_id, ev['camera_id'], ev['snapshot_path'], datetime.now().isoformat(timespec="seconds")))

        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[DB ERROR] update_incident failed: {e}")


def get_active_incidents(limit=20):
    try:
        conn = get_connection()
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute("SELECT * FROM incidents WHERE status='ACTIVE' ORDER BY id DESC LIMIT ?", (limit,))
        rows = [dict(r) for r in c.fetchall()]
        
        for r in rows:
            if r.get('reasons'): r['reasons'] = json.loads(r['reasons'])
            if r.get('timeline'): r['timeline'] = json.loads(r['timeline'])
            
            c.execute("SELECT * FROM incident_evidence WHERE incident_id=? ORDER BY id ASC", (r['incident_id'],))
            r['evidence'] = [dict(ev) for ev in c.fetchall()]
            
        conn.close()
        return rows
    except Exception as e:
        print(f"[DB ERROR] get_active_incidents failed: {e}")
        return []

def get_incident(incident_id):
    try:
        conn = get_connection()
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute("SELECT * FROM incidents WHERE incident_id=?", (incident_id,))
        row = c.fetchone()
        if not row:
            conn.close()
            return None
            
        incident = dict(row)
        if incident.get('reasons'): incident['reasons'] = json.loads(incident['reasons'])
        if incident.get('timeline'): incident['timeline'] = json.loads(incident['timeline'])
        
        c.execute("SELECT * FROM incident_evidence WHERE incident_id=? ORDER BY id ASC", (incident_id,))
        incident['evidence'] = [dict(ev) for ev in c.fetchall()]
        
        conn.close()
        return incident
    except Exception as e:
        print(f"[DB ERROR] get_incident failed: {e}")
        return None
