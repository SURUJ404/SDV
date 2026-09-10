from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lidar_detection import LidarDetector

def make_cloud(seed=3):
    rng = np.random.default_rng(seed)
    n = 5000
    ground = np.column_stack([
        rng.uniform(0,60,n), rng.uniform(-12,12,n),
        rng.normal(0,0.025,n)])
    objs = []
    for cx,cy,sx,sy,h,count in [
        (12,-1.2,2.0,1.0,1.6,350),
        (25,2.5,4.2,1.7,2.0,500),
        (40,-4.0,1.2,0.8,1.5,220)]:
        objs.append(np.column_stack([
            rng.uniform(-sx/2,sx/2,count)+cx,
            rng.uniform(-sy/2,sy/2,count)+cy,
            rng.uniform(0.05,h,count)]))
    return np.vstack([ground,*objs]).astype(np.float32)

if __name__ == "__main__":
    detector = LidarDetector.from_config(
        Path(__file__).resolve().parents[1]/"config/default.yaml")
    result = detector.detect(make_cloud(), timestamp=0.0)
    print("raw:", result.raw_points,
          "filtered:", result.filtered_points,
          "nonground:", result.nonground_points)
    for d in result.detections:
        print("id=%d center=%s dims=%s yaw=%.2f conf=%.2f" %
              (d.track_id, np.round(d.center,2), np.round(d.dimensions,2),
               d.yaw, d.confidence))
