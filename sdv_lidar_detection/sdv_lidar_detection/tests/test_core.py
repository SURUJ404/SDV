from pathlib import Path
import numpy as np
from lidar_detection import LidarDetector
from lidar_detection.sdv_adapter import decode_telemetry_lidar

ROOT = Path(__file__).resolve().parents[1]

def test_empty_cloud():
    d = LidarDetector.from_config(ROOT/"config/default.yaml")
    r = d.detect(np.empty((0,3), dtype=np.float32), timestamp=0.0)
    assert r.raw_points == 0
    assert not r.detections

def test_adapter():
    p = decode_telemetry_lidar([[1,2,3],[2,3,4]])
    assert p.shape == (2,3)

def test_nan_filter():
    d = LidarDetector.from_config(ROOT/"config/default.yaml")
    p = np.array([[5,0,0.5],[np.nan,0,0]], dtype=np.float32)
    r = d.detect(p, timestamp=0.0)
    assert r.raw_points == 1
