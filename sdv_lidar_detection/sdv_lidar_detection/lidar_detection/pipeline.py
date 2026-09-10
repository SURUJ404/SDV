import time
import numpy as np
import yaml
from .types import Detection3D, DetectionResult
from .preprocess import validate_points, range_filter, voxel_downsample
from .ground import remove_ground
from .clustering import dbscan, clusters_from_labels
from .geometry import oriented_bbox, physical_filter
from .tracking import MultiObjectTracker

class LidarDetector:
    def __init__(self, cfg):
        self.cfg = cfg
        self.p, self.g = cfg["preprocess"], cfg["ground"]
        self.c, self.o = cfg["clustering"], cfg["objects"]
        t = cfg.get("tracking", {})
        self.tracker = (MultiObjectTracker(
            max_age=t.get("max_age",5),
            max_match_distance=t.get("max_match_distance_m",3.0),
            velocity_smoothing=t.get("velocity_smoothing",0.65))
            if t.get("enabled", True) else None)
        self.last_timestamp = None

    @classmethod
    def from_config(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            return cls(yaml.safe_load(f))

    def detect(self, points, timestamp=None):
        raw = validate_points(points)
        filtered = range_filter(raw, self.p["min_range_m"], self.p["max_range_m"],
                                self.p["min_z_m"], self.p["max_z_m"])
        filtered = voxel_downsample(filtered, self.p["voxel_size_m"])

        if self.g.get("enabled", True):
            nonground, _, _ = remove_ground(
                filtered,
                threshold=self.g["distance_threshold_m"],
                iterations=self.g["iterations"],
                min_inliers=self.g["min_inliers"],
                max_tilt_deg=self.g["max_tilt_deg"])
        else:
            nonground = filtered

        labels = dbscan(nonground, self.c["eps_m"], self.c["min_points"])
        clusters = clusters_from_labels(nonground, labels, self.c["max_points"])
        raw_dets = []

        for cluster in clusters:
            center, dims, yaw = oriented_bbox(cluster)
            if not physical_filter(dims, self.o):
                continue
            density = len(cluster) / max(float(np.prod(dims)), 1e-3)
            confidence = float(np.clip(
                0.35 + 0.15*np.log1p(len(cluster)) + 0.03*np.log1p(density),
                0.0, 0.99))
            raw_dets.append({
                "center": center, "dimensions": dims, "yaw": yaw,
                "point_count": len(cluster), "confidence": confidence})

        dt = 0.1
        if timestamp is not None and self.last_timestamp is not None:
            dt = float(np.clip(timestamp-self.last_timestamp, 0.02, 1.0))
        self.last_timestamp = timestamp if timestamp is not None else time.time()

        if self.tracker is not None:
            tracks = self.tracker.update(raw_dets, dt)
            detections = []
            for t in tracks:
                nearest = min(raw_dets,
                              key=lambda d: np.linalg.norm(d["center"]-t.center),
                              default=None)
                detections.append(Detection3D(
                    t.track_id, t.center.copy(), t.dimensions.copy(), t.yaw,
                    nearest["point_count"] if nearest else 0,
                    nearest["confidence"] if nearest else 0.25,
                    t.velocity.copy()))
        else:
            detections = [
                Detection3D(i+1,d["center"],d["dimensions"],d["yaw"],
                            d["point_count"],d["confidence"])
                for i,d in enumerate(raw_dets)]

        return DetectionResult(timestamp, detections, len(raw),
                               len(filtered), len(nonground))
