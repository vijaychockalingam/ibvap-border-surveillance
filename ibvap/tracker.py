import math


class CentroidTracker:
    """
    Minimal object tracker: matches new detections to existing tracked
    objects by nearest centroid. No neural re-identification, so it's
    essentially free on CPU - good enough to give each person/vehicle a
    stable ID across frames for a demo.
    """

    def __init__(self, max_disappeared=15, max_distance=80):
        self.next_object_id = 0
        self.objects = {}       # object_id -> {centroid, bbox, label, type, confidence}
        self.disappeared = {}   # object_id -> frames since last matched
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    @staticmethod
    def _centroid(bbox):
        x1, y1, x2, y2 = bbox
        return (int((x1 + x2) / 2), int((y1 + y2) / 2))

    def _register(self, det, centroid):
        self.objects[self.next_object_id] = {
            "centroid": centroid,
            "bbox": det["bbox"],
            "label": det["label"],
            "type": det["type"],
            "confidence": det["confidence"],
        }
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def _forget_stale(self, obj_id):
        self.disappeared[obj_id] += 1
        if self.disappeared[obj_id] > self.max_disappeared:
            self.objects.pop(obj_id, None)
            self.disappeared.pop(obj_id, None)

    def update(self, detections):
        if len(detections) == 0:
            for obj_id in list(self.disappeared.keys()):
                self._forget_stale(obj_id)
            return self.objects

        input_centroids = [self._centroid(d["bbox"]) for d in detections]

        if len(self.objects) == 0:
            for i, det in enumerate(detections):
                self._register(det, input_centroids[i])
            return self.objects

        object_ids = list(self.objects.keys())
        object_centroids = [self.objects[oid]["centroid"] for oid in object_ids]

        # Build all (distance, existing_idx, new_idx) pairs, greedily assign
        # closest first - simple and fast enough for a handful of objects.
        pairs = []
        for r, oc in enumerate(object_centroids):
            for c, ic in enumerate(input_centroids):
                pairs.append((math.dist(oc, ic), r, c))
        pairs.sort(key=lambda p: p[0])

        used_rows, used_cols = set(), set()
        for dist, r, c in pairs:
            if r in used_rows or c in used_cols:
                continue
            if dist > self.max_distance:
                continue
            obj_id = object_ids[r]
            det = detections[c]
            self.objects[obj_id].update({
                "centroid": input_centroids[c],
                "bbox": det["bbox"],
                "label": det["label"],
                "type": det["type"],
                "confidence": det["confidence"],
            })
            self.disappeared[obj_id] = 0
            used_rows.add(r)
            used_cols.add(c)

        for r in set(range(len(object_ids))) - used_rows:
            self._forget_stale(object_ids[r])

        for c in set(range(len(input_centroids))) - used_cols:
            self._register(detections[c], input_centroids[c])

        return self.objects
