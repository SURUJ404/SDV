# Conservative patch for SURUJ404/SDV drive_v3.py.
#
# Add:
#   import time
#   from lidar_detection import LidarDetector
#   from lidar_detection.sdv_adapter import decode_telemetry_lidar
#
# Create during startup:
#   lidar_detector = LidarDetector.from_config(
#       "lidar_detection/config/default.yaml")
#
# In telemetry(), after reading speed:
#
#   lidar_obstacle_distance_m = None
#   try:
#       lidar_points = decode_telemetry_lidar(data.get("lidar"))
#       if lidar_points is not None:
#           result = lidar_detector.detect(
#               lidar_points, timestamp=time.time())
#           closest = result.closest_in_path(1.0, 60.0, 2.0)
#           if closest is not None:
#               lidar_obstacle_distance_m = closest.x
#   except Exception as exc:
#       print(f"[LiDAR] frame skipped: {exc}")
#
# Fuse with existing YOLO distance:
#
#   if lidar_obstacle_distance_m is not None:
#       if obstacle_distance_m is None:
#           obstacle_distance_m = lidar_obstacle_distance_m
#       else:
#           obstacle_distance_m = min(
#               obstacle_distance_m, lidar_obstacle_distance_m)
#
# LiDAR failures must never terminate the camera control loop.
# Do not fabricate a LiDAR stream when the simulator does not provide one.
