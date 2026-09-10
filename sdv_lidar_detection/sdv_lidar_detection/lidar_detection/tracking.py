from dataclasses import dataclass
import numpy as np
from scipy.optimize import linear_sum_assignment

@dataclass
class _Track:
    track_id: int
    center: np.ndarray
    velocity: np.ndarray
    dimensions: np.ndarray
    yaw: float
    age: int = 0
    hits: int = 1

class MultiObjectTracker:
    def __init__(self, max_age=5, max_match_distance=3.0, velocity_smoothing=0.65):
        self.max_age = max_age
        self.max_match_distance = max_match_distance
        self.alpha = velocity_smoothing
        self.tracks = {}
        self.next_id = 1

    def update(self, detections, dt=0.1):
        if not self.tracks:
            for d in detections:
                self._new(d)
            return list(self.tracks.values())

        tids = list(self.tracks)
        predicted = np.array([
            self.tracks[t].center + self.tracks[t].velocity * dt for t in tids
        ])
        observed = np.array([d["center"] for d in detections], dtype=np.float32)
        matched_t, matched_d = set(), set()

        if len(predicted) and len(observed):
            cost = np.linalg.norm(
                predicted[:, None, :2] - observed[None, :, :2], axis=2
            )
            rows, cols = linear_sum_assignment(cost)
            for r, c in zip(rows, cols):
                if cost[r, c] <= self.max_match_distance:
                    matched_t.add(r); matched_d.add(c)
                    tr = self.tracks[tids[r]]
                    new_v = (observed[c] - tr.center) / max(dt, 1e-3)
                    tr.velocity = self.alpha * new_v + (1-self.alpha) * tr.velocity
                    tr.center = observed[c]
                    tr.dimensions = detections[c]["dimensions"]
                    tr.yaw = detections[c]["yaw"]
                    tr.age = 0; tr.hits += 1

        for r, tid in enumerate(tids):
            if r not in matched_t:
                self.tracks[tid].age += 1
                self.tracks[tid].center += self.tracks[tid].velocity * dt

        for i, d in enumerate(detections):
            if i not in matched_d:
                self._new(d)

        for tid in list(self.tracks):
            if self.tracks[tid].age > self.max_age:
                del self.tracks[tid]
        return list(self.tracks.values())

    def _new(self, d):
        tid = self.next_id; self.next_id += 1
        self.tracks[tid] = _Track(
            tid, d["center"].copy(), np.zeros(3, dtype=np.float32),
            d["dimensions"].copy(), float(d["yaw"])
        )
