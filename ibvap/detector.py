from ultralytics import YOLO

import config

# COCO class ids we care about for border surveillance.
# (Everything else - dogs, chairs, traffic lights, etc. - is filtered out
# at detection time so we don't waste CPU cycles or clutter the demo.)
CLASSES_OF_INTEREST = {
    0: ("person", "person"),
    2: ("car", "vehicle"),
    3: ("motorcycle", "vehicle"),
    5: ("bus", "vehicle"),
    7: ("truck", "vehicle"),
}


class Detector:
    def __init__(self, model_path=config.MODEL_PATH, conf_threshold=config.CONF_THRESHOLD):
        # First run downloads yolov8n.pt automatically (~6MB) - needs internet once.
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold

    def detect(self, frame):
        results = self.model.predict(
            source=frame,
            conf=self.conf_threshold,
            classes=list(CLASSES_OF_INTEREST.keys()),
            verbose=False,
        )[0]

        detections = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            if cls_id not in CLASSES_OF_INTEREST:
                continue
            label, obj_type = CLASSES_OF_INTEREST[cls_id]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append({
                "label": label,
                "type": obj_type,
                "confidence": conf,
                "bbox": (x1, y1, x2, y2),
            })
        return detections
