import os
import time
import json
import uuid
import threading
import shutil
from datetime import datetime

from context_engine import ContextEngine
from entity_memory import EntityMemory
import database as db

class IncidentEngine:
    def __init__(self):
        self.entity_memory = EntityMemory()
        self.lock = threading.Lock()
        
        self.active_incidents = {} 
        
        self.evidence_dir = "evidence_packages"
        os.makedirs(self.evidence_dir, exist_ok=True)
        
    def process_event(self, camera_id, camera_name, event_type, object_type, local_obj_id, 
                      confidence, snapshot_path, plate_number=None, is_night=False, in_zone=False):
        
        with self.lock:
            # 1. Correlate Entity
            eid, is_new, cross_camera = self.entity_memory.register_or_correlate(
                camera_id, object_type, local_obj_id, plate_number
            )
            
            # 2. Context Evaluation
            is_unusual_time = ContextEngine.is_unusual_time(camera_id)
            watchlist_match = (event_type == "WATCHLIST_MATCH")
            
            severity, reasons = ContextEngine.evaluate_priority(
                event_type, object_type, is_night, in_zone, is_unusual_time, cross_camera, watchlist_match
            )
            
            # 3. Create or Update Incident
            incident_id = self.active_incidents.get(eid)
            
            description = f"{event_type} - {object_type.upper()} {eid} at {camera_name}"
            if plate_number:
                description += f" [{plate_number}]"
                
            timeline_entry = {
                "time": datetime.now().isoformat(timespec="seconds"),
                "camera_id": camera_id,
                "event": event_type,
                "desc": description
            }
            
            if incident_id:
                # Update existing incident
                new_evidence = []
                if snapshot_path:
                    new_evidence.append({'camera_id': camera_id, 'snapshot_path': snapshot_path})
                    
                db.update_incident(incident_id, timeline=[timeline_entry], new_evidence=new_evidence)
                self._update_evidence_package(incident_id, snapshot_path, timeline_entry)
            else:
                # Promote to incident if warrants
                if severity in ["HIGH", "CRITICAL"] or event_type in ["INTRUSION", "LOITERING", "WATCHLIST_MATCH"]:
                    incident_id = f"INC-{datetime.now().strftime('%Y%m%d')}-{str(uuid.uuid4())[:6].upper()}"
                    self.active_incidents[eid] = incident_id
                    
                    if cross_camera:
                        history = self.entity_memory.memory[eid]['history']
                        cam_seq = " -> ".join([h['camera_id'] for h in history])
                        description = f"Cross-camera movement: {cam_seq}. " + description
                        
                    db.create_incident(
                        incident_id=incident_id,
                        severity=severity,
                        primary_event_type=event_type,
                        primary_camera_id=camera_id,
                        description=description,
                        reasons=reasons,
                        timeline=[timeline_entry]
                    )
                    
                    new_evidence = []
                    if snapshot_path:
                        new_evidence.append({'camera_id': camera_id, 'snapshot_path': snapshot_path})
                        
                    if new_evidence:
                        db.update_incident(incident_id, new_evidence=new_evidence)
                        
                    self._create_evidence_package(incident_id, snapshot_path, timeline_entry, reasons, description)
            
            # 4. Standard Event Log
            db.log_event(
                camera_id, camera_name, event_type, object_type, local_obj_id, 
                confidence, snapshot_path, severity, plate_number, description=description
            )
            
            return incident_id, severity, description

    def _create_evidence_package(self, incident_id, snapshot_path, timeline_entry, reasons, description):
        try:
            pkg_dir = os.path.join(self.evidence_dir, incident_id)
            os.makedirs(pkg_dir, exist_ok=True)
            
            if snapshot_path and os.path.exists(snapshot_path):
                shutil.copy(snapshot_path, os.path.join(pkg_dir, os.path.basename(snapshot_path)))
                
            meta = {
                "incident_id": incident_id,
                "description": description,
                "reasons": reasons,
                "timeline": [timeline_entry]
            }
            with open(os.path.join(pkg_dir, "metadata.json"), "w") as f:
                json.dump(meta, f, indent=2)
        except Exception as e:
            print(f"[Evidence Error] {e}")

    def _update_evidence_package(self, incident_id, snapshot_path, timeline_entry):
        try:
            pkg_dir = os.path.join(self.evidence_dir, incident_id)
            if not os.path.exists(pkg_dir):
                return
                
            if snapshot_path and os.path.exists(snapshot_path):
                shutil.copy(snapshot_path, os.path.join(pkg_dir, os.path.basename(snapshot_path)))
                
            meta_file = os.path.join(pkg_dir, "metadata.json")
            if os.path.exists(meta_file):
                with open(meta_file, "r") as f:
                    meta = json.load(f)
                meta["timeline"].append(timeline_entry)
                with open(meta_file, "w") as f:
                    json.dump(meta, f, indent=2)
        except Exception as e:
            print(f"[Evidence Error] {e}")
