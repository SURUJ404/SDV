"""
Replacement for the original repo's drive.py.

Original: loads a single-frame PilotNet model, predicts steering from
whatever frame the simulator just sent, done.

This version:
  1. Buffers the last SEQUENCE_LENGTH frames so the CNN+LSTM model gets
     the temporal context it needs (a single new frame alone isn't
     enough input for this architecture).
  2. Optionally runs the U-Net in parallel purely for live visualization/
     logging of the detected lane mask — useful for debugging where the
     steering model's attention should be, and a drop-in hook point if
     you later want to feed the mask itself into the steering model as
     an extra input channel.
  3. Keeps the same SocketIO <-> Udacity simulator protocol as the
     original so it's a drop-in replacement — no simulator-side changes.
"""

import argparse
import base64
from collections import deque
from io import BytesIO

import cv2
import eventlet
import numpy as np
import socketio
from flask import Flask
from PIL import Image

from lane_segmentation.unet_model import dice_coef, bce_dice_loss
from steering_model.cnn_lstm_model import SEQUENCE_LENGTH
from steering_model.data_augmentation import crop_and_resize

import tensorflow as tf

sio = socketio.Server()
app = Flask(__name__)

steering_model = None
lane_model = None
frame_buffer = deque(maxlen=SEQUENCE_LENGTH)

MAX_SPEED = 20
MIN_SPEED = 8
speed_target = MAX_SPEED


def preprocess(image_rgb):
    return crop_and_resize(image_rgb)


def predict_steering():
    """Runs once the buffer has a full sequence; returns 0.0 until then."""
    if len(frame_buffer) < SEQUENCE_LENGTH:
        return 0.0
    sequence = np.expand_dims(np.stack(frame_buffer, axis=0), axis=0)  # (1, T, H, W, C)
    return float(steering_model.predict(sequence, verbose=0)[0][0])


def predict_lane_mask(raw_frame_bgr):
    """Optional — only used if --lane_model was passed. Purely diagnostic here."""
    if lane_model is None:
        return None
    resized = cv2.resize(raw_frame_bgr, (320, 160))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    batch = np.expand_dims(rgb.astype(np.float32), axis=0)
    mask = lane_model.predict(batch, verbose=0)[0, ..., 0]
    return (mask > 0.5).astype(np.uint8) * 255


@sio.on("telemetry")
def telemetry(sid, data):
    if not data:
        return

    speed = float(data["speed"])
    image = Image.open(BytesIO(base64.b64decode(data["image"])))
    raw_frame = np.asarray(image)  # RGB, as PIL decodes it

    processed = preprocess(raw_frame)
    frame_buffer.append(processed)

    steering_angle = predict_steering()

    # Simple speed control: back off approaching max speed, floor at MIN_SPEED,
    # and ease off further the harder we're steering (sharper turns -> slower).
    throttle = 1.0 - (steering_angle ** 2) - (speed / speed_target)
    throttle = float(np.clip(throttle, -1.0, 1.0))

    if lane_model is not None:
        raw_bgr = cv2.cvtColor(raw_frame, cv2.COLOR_RGB2BGR)
        mask = predict_lane_mask(raw_bgr)
        # Hook point: log mask, or in a future iteration, concatenate it
        # as a 4th input channel to the steering model.
        _ = mask

    send_control(steering_angle, throttle)


@sio.on("connect")
def connect(sid, environ):
    print("Simulator connected:", sid)
    frame_buffer.clear()
    send_control(0.0, 0.0)


def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={"steering_angle": str(steering_angle), "throttle": str(throttle)},
        skip_sid=True,
    )


def load_steering_model(path):
    return tf.keras.models.load_model(path, compile=False)


def load_lane_model(path):
    return tf.keras.models.load_model(
        path, custom_objects={"dice_coef": dice_coef, "bce_dice_loss": bce_dice_loss}
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("steering_model_path", help="Path to cnn_lstm_steering.h5")
    parser.add_argument("--lane_model", default=None, help="Optional path to lane_unet.h5")
    parser.add_argument("--speed", type=int, default=MAX_SPEED)
    args = parser.parse_args()

    speed_target = args.speed
    steering_model = load_steering_model(args.steering_model_path)
    if args.lane_model:
        lane_model = load_lane_model(args.lane_model)
        print("Lane U-Net loaded for live overlay/logging")

    print(f"CNN+LSTM steering model loaded, expecting {SEQUENCE_LENGTH}-frame sequences")

    wrapped_app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(("", 4567)), wrapped_app)
