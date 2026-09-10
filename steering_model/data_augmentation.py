"""
Augmentation matching the techniques NVIDIA describes in "End to End
Learning for Self-Driving Cars" (Bojarski et al., 2016) and the common
Udacity behavioral-cloning extensions on top of it. The original repo's
notebook does no augmentation at all beyond maybe a flip, which is the
single biggest reason PilotNet clones overfit to the training track and
fail on unseen curves/lighting.

All functions operate on a single (image, steering_angle) pair so they
compose cleanly inside a sequence-building generator.
"""

import cv2
import numpy as np

IMG_SHAPE = (66, 200, 3)  # (H, W, C) after crop+resize, matches NVIDIA's input


def crop_and_resize(image):
    """
    NVIDIA crops the sky/hood out of the frame before feeding the CNN —
    those regions carry no lane information and just add noise.
    Assumes the raw simulator frame is 160x320.
    """
    cropped = image[60:135, :, :]  # drop sky (top) and hood (bottom)
    return cv2.resize(cropped, (IMG_SHAPE[1], IMG_SHAPE[0]), interpolation=cv2.INTER_AREA)


def random_brightness(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
    ratio = 0.4 + np.random.uniform()
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * ratio, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def random_shadow(image):
    """Simulates tree-shade / overpass shadows by darkening a random polygon strip."""
    h, w = image.shape[:2]
    top_x, bottom_x = np.random.randint(0, w, 2)
    overlay = image.copy()
    poly = np.array([[top_x, 0], [top_x + w // 2, 0], [bottom_x + w // 2, h], [bottom_x, h]])
    cv2.fillPoly(overlay, [poly], (0, 0, 0))
    alpha = np.random.uniform(0.5, 0.8)
    return cv2.addWeighted(overlay, 1 - alpha, image, alpha, 0)


def random_translate(image, steering, max_shift_x=30, max_shift_y=10):
    """
    Shifts the image horizontally/vertically and adjusts steering to match —
    teaches the model recovery behavior (what to do when off-center) without
    needing real recovery-driving data.
    """
    shift_x = np.random.uniform(-max_shift_x, max_shift_x)
    shift_y = np.random.uniform(-max_shift_y, max_shift_y)
    steering_adjusted = steering + shift_x * 0.004  # empirical correction factor
    h, w = image.shape[:2]
    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    shifted = cv2.warpAffine(image, M, (w, h))
    return shifted, steering_adjusted


def random_flip(image, steering):
    if np.random.rand() < 0.5:
        return np.fliplr(image), -steering
    return image, steering


def augment(image, steering):
    """Full augmentation pipeline applied during training only (not at inference)."""
    image, steering = random_translate(image, steering)
    image, steering = random_flip(image, steering)
    if np.random.rand() < 0.5:
        image = random_shadow(image)
    if np.random.rand() < 0.5:
        image = random_brightness(image)
    return image, steering


def choose_camera(row, correction=0.2):
    """
    Multi-camera trick from the NVIDIA paper: the simulator logs left,
    center, and right camera images per timestep. Using all three
    (with a steering correction for the off-center views) triples the
    effective training data and, more importantly, teaches the model
    what off-center driving looks like and how to correct for it —
    exactly what you'd otherwise need real "recovery" driving data for.

    `row` is expected to have keys: 'center', 'left', 'right', 'steering'.
    """
    choice = np.random.choice(["center", "left", "right"])
    if choice == "left":
        return row["left"], row["steering"] + correction
    elif choice == "right":
        return row["right"], row["steering"] - correction
    return row["center"], row["steering"]
