"""
Full pipeline driver — supersedes drive_v2.py.

Per-frame pipeline:
  1. CNN+LSTM proposes a steering angle (steering_model/)
  2. U-Net (optional) segments the lane for a lateral-offset estimate
  3. YOLO (optional) detects in-lane obstacles and their distance
  4. MPC takes the CNN-LSTM's steering as a prior / the U-Net's lateral
     offset as the tracking target, and the YOLO distance as a hard
     safety constraint, and outputs the final steering + acceleration
  5. A PID loop converts the MPC's acceleration target into a throttle
     command (acceleration -> throttle is not 1:1 in the simulator, so
     PID closes that loop rather than assuming it)

If --lane_model or --object_model are omitted, the MPC falls back to
tracking the CNN-LSTM's steering angle directly with no obstacle
constraint — i.e. it degrades gracefully to "MPC as a smoothing/limiting
layer on top of the learned model" rather than requiring the full stack.
"""

import argparse
import base64
from collections import deque
from io import BytesIO

import cv2
import eventlet
import numpy as np
import socketio
import tensorflow as tf
from flask import Flask
from PIL import Image

from lane_segmentation.unet_model import dice_coef, bce_dice_loss
from steering_model.cnn_lstm_model import SEQUENCE_LENGTH
from steering_model.data_augmentation import crop_and_resize
from planning.pid_controller import build_default_speed_controller
from planning.mpc_controller import MPCController

sio = socketio.Server()
app = Flask(__name__)

steering_model = None
lane_model = None
yolo_detector = None
mpc = MPCController()
speed_pid = build_default_speed_controller()
frame_buffer = deque(maxlen=SEQUENCE_LENGTH)

TARGET_CRUISE_SPEED_MS = 8.0  # ~18 mph, conservative default


def predict_cnn_lstm_steering():
    if len(frame_buffer) < SEQUENCE_LENGTH:
        return 0.0
    sequence = np.expand_dims(np.stack(frame_buffer, axis=0), axis=0)
    return float(steering_model.predict(sequence, verbose=0)[0][0])


def estimate_lateral_offset_from_mask(mask):
    """
    Crude but effective: find the centroid of lane-mask pixels in the
    bottom third of the frame (closest to the car) and compare it to
    image-center. Positive = lane center is to the right of the car.
    """
    h, w = mask.shape
    band = mask[int(h * 0.7):, :]
    ys, xs = np.where(band > 0)
    if len(xs) == 0:
        return 0.0
    centroid_x = xs.mean()
    image_center_x = w / 2
    pixels_per_meter = w / 3.7  # assume lane width fills roughly the frame width
    return float((centroid_x - image_center_x) / pixels_per_meter)


@sio.on("telemetry")
def telemetry(sid, data):
    if not data:
        return

    speed = float(data["speed"])
    image = Image.open(BytesIO(base64.b64decode(data["image"])))
    raw_frame_rgb = np.asarray(image)
    raw_frame_bgr = cv2.cvtColor(raw_frame_rgb, cv2.COLOR_RGB2BGR)

    frame_buffer.append(crop_and_resize(raw_frame_rgb))
    cnn_steering = predict_cnn_lstm_steering()

    # --- Lane offset target (U-Net if available, else trust the CNN's implicit target) ---
    lateral_offset_target = 0.0
    if lane_model is not None:
        resized = cv2.resize(raw_frame_bgr, (320, 160))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        mask = lane_model.predict(np.expand_dims(rgb.astype(np.float32), 0), verbose=0)[0, ..., 0]
        mask = (mask > 0.5).astype(np.uint8) * 255
        lateral_offset_target = estimate_lateral_offset_from_mask(mask)

    # --- Obstacle distance (YOLO if available) ---
    obstacle_distance_m = None
    if yolo_detector is not None:
        detections = yolo_detector.detect(raw_frame_rgb)
        closest = yolo_detector.closest_in_path(detections)
        if closest is not None:
            obstacle_distance_m = closest.distance_m

    # --- MPC: fuse CNN-LSTM prior + lane offset target + obstacle constraint ---
    current_state = np.array([lateral_offset_target, cnn_steering * 0.3, speed])
    steering_angle, target_accel = mpc.solve(
        current_state,
        target_lateral_offset=0.0,  # always steer back toward lane center
        target_speed=TARGET_CRUISE_SPEED_MS,
        obstacle_distance_m=obstacle_distance_m,
    )

    # Blend: MPC provides safety/smoothness, CNN-LSTM keeps the model's
    # learned road-reading — weighted average rather than fully overriding it.
    final_steering = float(np.clip(0.6 * steering_angle + 0.4 * cnn_steering, -1.0, 1.0))

    # --- PID converts the MPC's target acceleration into throttle, closing
    # the loop against measured speed rather than assuming accel==throttle ---
    target_speed_from_accel = speed + target_accel * 0.5  # short-horizon speed target
    throttle = speed_pid.step(setpoint=target_speed_from_accel, measurement=speed)

    send_control(final_steering, throttle)


@sio.on("connect")
def connect(sid, environ):
    print("Simulator connected:", sid)
    frame_buffer.clear()
    speed_pid.reset()
    send_control(0.0, 0.0)


def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={"steering_angle": str(steering_angle), "throttle": str(throttle)},
        skip_sid=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("steering_model_path")
    parser.add_argument("--lane_model", default=None)
    parser.add_argument("--object_model", default="yolov8n.pt")
    parser.add_argument("--disable_yolo", action="store_true")
    parser.add_argument("--cruise_speed", type=float, default=TARGET_CRUISE_SPEED_MS)
    args = parser.parse_args()

    TARGET_CRUISE_SPEED_MS = args.cruise_speed
    steering_model = tf.keras.models.load_model(args.steering_model_path, compile=False)
    print(f"CNN+LSTM steering model loaded ({SEQUENCE_LENGTH}-frame sequences)")

    if args.lane_model:
        lane_model = tf.keras.models.load_model(
            args.lane_model, custom_objects={"dice_coef": dice_coef, "bce_dice_loss": bce_dice_loss}
        )
        print("Lane U-Net loaded")

    if not args.disable_yolo:
        from object_detection.yolo_detector import YoloDetector
        yolo_detector = YoloDetector(weights=args.object_model)
        print("YOLO object detector loaded")

    wrapped_app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(("", 4567)), wrapped_app)
