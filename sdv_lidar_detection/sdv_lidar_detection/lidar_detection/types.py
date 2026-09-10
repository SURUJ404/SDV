from dataclasses import dataclass, field
from typing import Optional
import numpy as np

@dataclass
class Detection3D:
    track_id: int
    center: np.ndarray
    dimensions: np.ndarray
    yaw: float
    point_count: int
    confidence: float
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))

    @property
    def distance_m(self):
        return float(np.linalg.norm(self.center[:2]))

    @property
    def x(self):
        return float(self.center[0])

    @property
    def y(self):
        return float(self.center[1])

@dataclass
class DetectionResult:
    timestamp: Optional[float]
    detections: list
    raw_points: int
    filtered_points: int
    nonground_points: int

    def closest_in_path(self, forward_min=1.0, forward_max=60.0, half_width=2.0):
        candidates = [
            d for d in self.detections
            if forward_min <= d.x <= forward_max and abs(d.y) <= half_width
        ]
        return min(candidates, key=lambda d: d.x, default=None)
