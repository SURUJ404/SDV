"""
Builds training batches of (sequence_of_frames, steering_angle) from the
simulator's driving_log.csv, applying multi-camera sampling and
augmentation per-frame, then stacking SEQUENCE_LENGTH consecutive frames
for the CNN+LSTM model.

driving_log.csv format (standard Udacity simulator output):
    center_path, left_path, right_path, steering, throttle, brake, speed
"""

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

from data_augmentation import augment, choose_camera, crop_and_resize, IMG_SHAPE
from cnn_lstm_model import SEQUENCE_LENGTH


def load_log(csv_path):
    df = pd.read_csv(
        csv_path,
        names=["center", "left", "right", "steering", "throttle", "brake", "speed"],
        header=0 if _has_header(csv_path) else None,
    )
    return df


def _has_header(csv_path):
    with open(csv_path) as f:
        first_line = f.readline()
    return "steering" in first_line.lower()


def read_image(path):
    img = cv2.imread(path.strip())
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


class SteeringSequenceGenerator(tf.keras.utils.Sequence):
    """
    Yields batches shaped (batch_size, SEQUENCE_LENGTH, 66, 200, 3).

    Sequences are built from consecutive rows in the log (the simulator
    logs frames in driving order), so a sequence is a short clip of real
    consecutive driving rather than randomly sampled unrelated frames —
    that temporal continuity is what the LSTM needs to learn from.
    """

    def __init__(self, df, batch_size=32, sequence_length=SEQUENCE_LENGTH,
                 augment_data=True, camera_correction=0.2):
        self.df = df.reset_index(drop=True)
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.augment_data = augment_data
        self.camera_correction = camera_correction
        # Valid starting indices leave room for a full sequence
        self.valid_starts = list(range(0, len(self.df) - sequence_length))

    def __len__(self):
        return int(np.ceil(len(self.valid_starts) / self.batch_size))

    def __getitem__(self, idx):
        batch_starts = self.valid_starts[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = np.zeros((len(batch_starts), self.sequence_length, *IMG_SHAPE), dtype=np.float32)
        batch_y = np.zeros((len(batch_starts),), dtype=np.float32)

        for i, start in enumerate(batch_starts):
            frames = []
            last_steering = 0.0
            for offset in range(self.sequence_length):
                row = self.df.iloc[start + offset]
                if self.augment_data:
                    img_path, steering = choose_camera(row, self.camera_correction)
                else:
                    img_path, steering = row["center"], row["steering"]

                image = read_image(img_path)
                image = crop_and_resize(image)

                if self.augment_data:
                    image, steering = augment(image, steering)

                frames.append(image)
                last_steering = steering  # label = steering at the LAST frame in the sequence

            batch_x[i] = np.stack(frames, axis=0)
            batch_y[i] = last_steering

        return batch_x, batch_y

    def on_epoch_end(self):
        if self.augment_data:
            np.random.shuffle(self.valid_starts)
