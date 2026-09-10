"""
U-Net for binary lane segmentation.

Replaces the classical Canny + ROI-mask + Hough-transform pipeline
(1.Finding_Lanes/ in the original repo) with a learned encoder-decoder
segmentation network. This generalizes far better than hand-tuned
edge thresholds and ROI polygons: it handles curves, shadows, faded
paint, worn asphalt, and lighting changes that break Hough-line fits.

Input:  RGB image, default 160x320x3 (matches the simulator's camera feed)
Output: single-channel probability mask, same H/W, values in [0, 1]
"""

import tensorflow as tf
from tensorflow.keras import layers, Model


def conv_block(x, filters, name):
    """Two 3x3 conv layers with BN + ReLU, the standard U-Net unit block."""
    x = layers.Conv2D(filters, 3, padding="same", name=f"{name}_conv1")(x)
    x = layers.BatchNormalization(name=f"{name}_bn1")(x)
    x = layers.Activation("relu", name=f"{name}_relu1")(x)
    x = layers.Conv2D(filters, 3, padding="same", name=f"{name}_conv2")(x)
    x = layers.BatchNormalization(name=f"{name}_bn2")(x)
    x = layers.Activation("relu", name=f"{name}_relu2")(x)
    return x


def build_unet(input_shape=(160, 320, 3), base_filters=32):
    """
    Builds a 4-level U-Net.

    base_filters controls model capacity. 32 is a good default for
    lane masks (a much simpler target than general semantic segmentation,
    so we don't need the full 64-filter U-Net from the original paper).
    """
    inputs = layers.Input(shape=input_shape, name="image_input")

    # Normalize inside the graph so drive.py / inference never has to
    # remember to do it separately.
    x = layers.Rescaling(1.0 / 255.0, name="rescale")(inputs)

    # ---- Encoder ----
    c1 = conv_block(x, base_filters, "enc1")
    p1 = layers.MaxPooling2D(2, name="pool1")(c1)

    c2 = conv_block(p1, base_filters * 2, "enc2")
    p2 = layers.MaxPooling2D(2, name="pool2")(c2)

    c3 = conv_block(p2, base_filters * 4, "enc3")
    p3 = layers.MaxPooling2D(2, name="pool3")(c3)

    c4 = conv_block(p3, base_filters * 8, "enc4")
    p4 = layers.MaxPooling2D(2, name="pool4")(c4)

    # ---- Bottleneck ----
    bn = conv_block(p4, base_filters * 16, "bottleneck")

    # ---- Decoder (with skip connections) ----
    u4 = layers.Conv2DTranspose(base_filters * 8, 2, strides=2, padding="same", name="up4")(bn)
    u4 = layers.Concatenate(name="concat4")([u4, c4])
    d4 = conv_block(u4, base_filters * 8, "dec4")

    u3 = layers.Conv2DTranspose(base_filters * 4, 2, strides=2, padding="same", name="up3")(d4)
    u3 = layers.Concatenate(name="concat3")([u3, c3])
    d3 = conv_block(u3, base_filters * 4, "dec3")

    u2 = layers.Conv2DTranspose(base_filters * 2, 2, strides=2, padding="same", name="up2")(d3)
    u2 = layers.Concatenate(name="concat2")([u2, c2])
    d2 = conv_block(u2, base_filters * 2, "dec2")

    u1 = layers.Conv2DTranspose(base_filters, 2, strides=2, padding="same", name="up1")(d2)
    u1 = layers.Concatenate(name="concat1")([u1, c1])
    d1 = conv_block(u1, base_filters, "dec1")

    outputs = layers.Conv2D(1, 1, activation="sigmoid", name="mask_output")(d1)

    return Model(inputs, outputs, name="lane_unet")


def dice_coef(y_true, y_pred, smooth=1.0):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (
        tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth
    )


def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)


def bce_dice_loss(y_true, y_pred):
    """
    Combined loss: binary cross-entropy handles per-pixel calibration,
    Dice handles the severe class imbalance (lane pixels are a small
    fraction of the frame). Using BCE alone tends to converge to an
    all-background solution on thin lane markings.
    """
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    return tf.reduce_mean(bce) + dice_loss(y_true, y_pred)


if __name__ == "__main__":
    model = build_unet()
    model.summary()
