import time
import uuid
from context_engine import ContextEngine

class EntityMemory:
    """
    Maintains a rolling memory of entities to track them across cameras heuristically.
    """
    def __init__(self, memory_ttl_seconds=60):
        self.memory = {}  
        self.memory_ttl = memory_ttl_seconds
        
    def _cleanup(self):
        now = time.time()
        to_delete = []
        for eid, data in self.memory.items():
            if now - data['last_seen'] > self.memory_ttl:
                to_delete.append(eid)
        for eid in to_delete:
            del self.memory[eid]

    def register_or_correlate(self, camera_id, object_type, local_obj_id, plate_number=None):
        self._cleanup()
        now = time.time()
        
        best_match = None
        best_delta = float('inf')
        
        for eid, data in self.memory.items():
            if data['type'] == object_type:
                
                # Hard match on plate number if available
                if plate_number and data.get('plate_number') == plate_number:
                    best_match = eid
                    break
                    
                time_delta = now - data['last_seen']
                if time_delta > 0 and ContextEngine.is_valid_transition(data['last_camera'], camera_id):
                    # Same object continuing on same camera
                    if data['last_camera'] == camera_id and data.get('local_obj_id') == local_obj_id:
                        best_match = eid
                        break
                    # Possible cross-camera jump
                    elif data['last_camera'] != camera_id:
                        if time_delta < best_delta:
                            best_delta = time_delta
                            best_match = eid

        if best_match:
            eid = best_match
            self.memory[eid]['last_seen'] = now
            self.memory[eid]['last_camera'] = camera_id
            self.memory[eid]['local_obj_id'] = local_obj_id
            if plate_number:
                self.memory[eid]['plate_number'] = plate_number
                
            # Avoid appending to history if it's the exact same camera in a short burst
            last_history = self.memory[eid]['history'][-1]
            if last_history['camera_id'] != camera_id or (now - last_history['time']) > 5:
                self.memory[eid]['history'].append({'camera_id': camera_id, 'time': now})
                
            is_new = False
            cross_camera = len(set(h['camera_id'] for h in self.memory[eid]['history'])) > 1
        else:
            eid = f"{object_type[0].upper()}-{str(uuid.uuid4())[:6].upper()}"
            self.memory[eid] = {
                'type': object_type,
                'first_seen': now,
                'last_seen': now,
                'last_camera': camera_id,
                'local_obj_id': local_obj_id,
                'plate_number': plate_number,
                'history': [{'camera_id': camera_id, 'time': now}]
            }
            is_new = True
            cross_camera = False

        return eid, is_new, cross_camera
