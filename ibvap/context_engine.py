import config
from datetime import datetime

class ContextEngine:
    @staticmethod
    def evaluate_priority(event_type, object_type, is_night, in_zone, is_unusual_time, cross_camera, watchlist_match):
        reasons = []
        score = 0
        
        if event_type == "WATCHLIST_MATCH" or watchlist_match:
            score += 100
            reasons.append("Flagged vehicle matched against watchlist")
            
        if in_zone:
            score += 50
            reasons.append("Intrusion into restricted zone")
            
        if event_type == "LOITERING":
            score += 40
            reasons.append("Extended loitering in restricted perimeter")
            
        if is_unusual_time:
            score += 30
            reasons.append("Activity detected during normally inactive hours")
            
        if cross_camera:
            score += 20
            reasons.append("Entity movement corroborated across multiple cameras")
            
        if is_night:
            score += 20
            reasons.append("Night-time movement detected")
            
        # Determine string severity
        if score >= 100:
            severity = "CRITICAL"
        elif score >= 50:
            severity = "HIGH"
        elif score >= 30:
            severity = "MEDIUM"
        else:
            severity = "LOW"
            
        if not reasons:
            if object_type == "person":
                reasons.append("Standard pedestrian detection")
            elif object_type == "vehicle":
                reasons.append("Standard vehicle tracking")
            else:
                reasons.append("Routine object detection")
                
        return severity, reasons

    @staticmethod
    def is_unusual_time(camera_id, current_time=None):
        if not current_time:
            current_time = datetime.now()
            
        baseline = config.ZONE_BASELINES.get(camera_id)
        if not baseline:
            return False
            
        start_hour, end_hour = baseline
        if start_hour == end_hour:
            return True # 24/7 restricted
            
        h = current_time.hour
        if start_hour <= end_hour:
            is_normal = (start_hour <= h < end_hour)
        else:
            is_normal = (h >= start_hour or h < end_hour)
            
        return not is_normal

    @staticmethod
    def is_valid_transition(from_camera_id, to_camera_id):
        if from_camera_id == to_camera_id:
            return True
        allowed = config.CAMERA_TOPOLOGY.get(from_camera_id, [])
        return to_camera_id in allowed
