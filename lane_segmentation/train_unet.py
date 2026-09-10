"""
Trains the U-Net lane segmentation model on (image, mask) pairs produced
by generate_pseudo_masks.py (or a real hand-labeled dataset in the same
folder layout: identical filenames, masks suffixed "_mask.png").

Usage:
    python train_unet.py --images_dir Data/IMG --masks_dir Data/masks \
        --epochs 40 --batch_size 16
"""

import argparse
import os

import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from unet_model import build_unet, bce_dice_loss, dice_coef

IMG_SIZE = (320, 160)  # (width, height) for cv2.resize


class SegmentationSequence(tf.keras.utils.Sequence):
    def __init__(self, image_paths, mask_paths, batch_size=16, augment=False):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.batch_size = batch_size
        self.augment = augment

    def __len__(self):
        return int(np.ceil(len(self.image_paths) / self.batch_size))

    def __getitem__(self, idx):
        batch_img_paths = self.image_paths[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_mask_paths = self.mask_paths[idx * self.batch_size:(idx + 1) * self.batch_size]

        images, masks = [], []
        for img_path, mask_path in zip(batch_img_paths, batch_mask_paths):
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, IMG_SIZE)

            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, IMG_SIZE)
            mask = (mask > 127).astype(np.float32)

            if self.augment and np.random.rand() < 0.5:
                img = np.fliplr(img)
                mask = np.fliplr(mask)

            images.append(img)
            masks.append(mask[..., np.newaxis])

        return np.array(images, dtype=np.float32), np.array(masks, dtype=np.float32)


def build_file_lists(images_dir, masks_dir):
    image_paths, mask_paths = [], []
    for fname in sorted(os.listdir(masks_dir)):
        if not fname.endswith("_mask.png"):
            continue
        base = fname[: -len("_mask.png")]
        for ext in (".jpg", ".png"):
            candidate = os.path.join(images_dir, base + ext)
            if os.path.exists(candidate):
                image_paths.append(candidate)
                mask_paths.append(os.path.join(masks_dir, fname))
                break
    return image_paths, mask_paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", required=True)
    parser.add_argument("--masks_dir", required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--out_model", default="lane_unet.h5")
    args = parser.parse_args()

    image_paths, mask_paths = build_file_lists(args.images_dir, args.masks_dir)
    print(f"Found {len(image_paths)} image/mask pairs")

    train_img, val_img, train_mask, val_mask = train_test_split(
        image_paths, mask_paths, test_size=0.15, random_state=42
    )

    train_gen = SegmentationSequence(train_img, train_mask, args.batch_size, augment=True)
    val_gen = SegmentationSequence(val_img, val_mask, args.batch_size, augment=False)

    model = build_unet(input_shape=(160, 320, 3))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=bce_dice_loss,
        metrics=[dice_coef],
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            args.out_model, save_best_only=True, monitor="val_dice_coef", mode="max"
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_dice_coef", mode="max", patience=10, restore_best_weights=True
        ),
    ]

    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    model.save(args.out_model)
    print(f"Saved model to {args.out_model}")


if __name__ == "__main__":
    main()
