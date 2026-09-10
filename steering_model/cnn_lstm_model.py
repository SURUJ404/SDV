"""
CNN+LSTM steering model — replaces the original repo's single-frame
PilotNet (NVIDIA's 2016 "End to End Learning for Self-Driving Cars"
5-conv-layer CNN with no temporal component).

Why this is stronger:
  - PilotNet predicts steering from ONE frame, so it has no notion of
    motion, curvature rate-of-change, or where the car is mid-maneuver.
    It reacts, but can't anticipate.
  - This model runs a shared CNN feature extractor (TimeDistributed,
    so weights are shared across time) over a short sequence of frames
    (default 5, ~0.5-1s of driving at typical simulator FPS), then feeds
    that sequence of feature vectors into an LSTM. The LSTM lets the
    steering decision depend on recent trajectory, which smooths output
    and helps through sustained curves where a single frame is ambiguous.

The CNN body keeps PilotNet's conv stack (proven to work well on this
exact 160x320 simulator image) but strips the final regression head,
using it purely as a per-frame feature extractor.
"""

import tensorflow as tf
from tensorflow.keras import layers, Model

SEQUENCE_LENGTH = 5
IMG_SHAPE = (66, 200, 3)  # NVIDIA paper's crop/resize target


def build_cnn_feature_extractor(input_shape=IMG_SHAPE):
    """The PilotNet conv stack, used per-frame inside the TimeDistributed wrapper."""
    inputs = layers.Input(shape=input_shape)
    x = layers.Rescaling(1.0 / 127.5, offset=-1.0)(inputs)  # match NVIDIA's [-1, 1] normalization

    x = layers.Conv2D(24, 5, strides=2, activation="elu")(x)
    x = layers.Conv2D(36, 5, strides=2, activation="elu")(x)
    x = layers.Conv2D(48, 5, strides=2, activation="elu")(x)
    x = layers.Conv2D(64, 3, activation="elu")(x)
    x = layers.Conv2D(64, 3, activation="elu")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(100, activation="elu")(x)
    x = layers.Dropout(0.3)(x)

    return Model(inputs, x, name="pilotnet_feature_extractor")


def build_cnn_lstm(sequence_length=SEQUENCE_LENGTH, img_shape=IMG_SHAPE):
    feature_extractor = build_cnn_feature_extractor(img_shape)

    sequence_input = layers.Input(shape=(sequence_length, *img_shape), name="frame_sequence")
    features = layers.TimeDistributed(feature_extractor, name="td_cnn")(sequence_input)

    x = layers.LSTM(64, return_sequences=True)(features)
    x = layers.LSTM(32, return_sequences=False)(x)
    x = layers.Dense(50, activation="elu")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(10, activation="elu")(x)
    steering_output = layers.Dense(1, activation="linear", name="steering")(x)

    return Model(sequence_input, steering_output, name="cnn_lstm_steering")


if __name__ == "__main__":
    model = build_cnn_lstm()
    model.summary()
