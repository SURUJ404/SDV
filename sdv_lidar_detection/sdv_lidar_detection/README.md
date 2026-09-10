# SDV LiDAR Detection

A from-scratch, dependency-light LiDAR perception stack designed to plug into SURUJ404/SDV.

Pipeline:
point cloud -> validation/range filtering -> voxel downsampling -> RANSAC ground removal
-> DBSCAN clustering -> 3-D bounding boxes -> multi-object tracking -> path obstacle query.

It does not require ROS, PCL, OpenPCDet, or a specific LiDAR vendor.

## SDV compatibility

The current SDV repository is a Python/Socket.IO Udacity-simulator stack using
U-Net, CNN-LSTM, YOLOv8 and MPC/PID. Its own README states that it has no LiDAR/radar
fusion. This project therefore makes LiDAR an optional channel: if telemetry contains
a valid `lidar` field, the detector uses it; otherwise the existing camera/YOLO path
can continue unchanged.

The standard Udacity simulator does not inherently provide a real LiDAR stream, so
actual LiDAR data must come from a simulator extension, replay source, or physical
sensor bridge. The detection core is independent of that source.

## Coordinate convention

Input is Nx3 or Nx4 float data:
`x,y,z[,intensity]`

Default convention:
+X forward, +Y left, +Z up.

Convert vendor/simulator coordinates once at the adapter boundary rather than allowing
the detector to guess.

## Install

```bash
pip install -r requirements.txt
```

## Demo

```bash
python examples/demo_synthetic.py
```

## SDV integration

Copy `lidar_detection/` into the SDV repository. Add:

```python
import time
from lidar_detection import LidarDetector
from lidar_detection.sdv_adapter import decode_telemetry_lidar
```

Create once during startup:

```python
lidar_detector = LidarDetector.from_config(
    "lidar_detection/config/default.yaml"
)
```

Inside `telemetry()`:

```python
lidar_obstacle_distance_m = None
try:
    lidar_points = decode_telemetry_lidar(data.get("lidar"))
    if lidar_points is not None:
        result = lidar_detector.detect(lidar_points, timestamp=time.time())
        closest = result.closest_in_path(1.0, 60.0, 2.0)
        if closest is not None:
            lidar_obstacle_distance_m = closest.x
except Exception as exc:
    print(f"[LiDAR] frame skipped: {exc}")
```

Then conservatively fuse with the existing YOLO obstacle distance:

```python
if lidar_obstacle_distance_m is not None:
    obstacle_distance_m = (
        lidar_obstacle_distance_m if obstacle_distance_m is None
        else min(obstacle_distance_m, lidar_obstacle_distance_m)
    )
```

The supplied `integration/drive_v3_lidar_patch.py` contains the same patch in one place.

## Calibration

`config/calibration.yaml` stores K, distortion, and the rigid LiDAR-to-camera
transform `p_camera = R @ p_lidar + t`. The calibration module validates R before use
and can project points into the image. This keeps calibration separate from detection
and is suitable for later camera-LiDAR fusion.

Do not reuse calibration values from another vehicle.

## Robustness

- Explicit coordinate conventions; no silent guessing.
- Safe handling of empty, NaN and Inf clouds.
- Ground-plane RANSAC with tilt validation and fallback.
- Physical-size rejection of implausible clusters.
- Deterministic clustering and tracking for fixed inputs.
- Missed-frame tolerant constant-velocity tracking.
- LiDAR exceptions are isolated from SDV's camera control loop.
- Geometric confidence is clearly not semantic class confidence.

For semantic vehicle/pedestrian/cyclist classes, add a learned classifier later;
the current detector intentionally reports geometry-only objects.
