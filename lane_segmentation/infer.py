"""
Run the trained U-Net on a single image or a video, and overlay the
predicted lane mask in green — a direct visual replacement for the
Hough-line overlay in the original 1.Finding_Lanes notebook.

Usage:
    python infer.py --model lane_unet.h5 --image test.jpg
    python infer.py --model lane_unet.h5 --video drive.mp4 --out annotated.mp4
"""

import argparse

import cv2
import numpy as np
import tensorflow as tf

from unet_model import dice_coef, bce_dice_loss

IMG_SIZE = (320, 160)


def load_model(path):
    return tf.keras.models.load_model(
        path, custom_objects={"dice_coef": dice_coef, "bce_dice_loss": bce_dice_loss}
    )


def predict_mask(model, frame_bgr, threshold=0.5):
    orig_h, orig_w = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, IMG_SIZE)
    batch = np.expand_dims(resized.astype(np.float32), axis=0)

    pred = model.predict(batch, verbose=0)[0, ..., 0]
    pred = cv2.resize(pred, (orig_w, orig_h))
    return (pred > threshold).astype(np.uint8) * 255


def overlay_mask(frame_bgr, mask, color=(0, 255, 0), alpha=0.5):
    overlay = frame_bgr.copy()
    overlay[mask > 0] = color
    return cv2.addWeighted(overlay, alpha, frame_bgr, 1 - alpha, 0)


def run_on_image(model, image_path):
    frame = cv2.imread(image_path)
    mask = predict_mask(model, frame)
    result = overlay_mask(frame, mask)
    out_path = "annotated_" + image_path.split("/")[-1]
    cv2.imwrite(out_path, result)
    print(f"Wrote {out_path}")


def run_on_video(model, video_path, out_path):
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = cap.get(cv2.CAP_PROP_FPS) or 20
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        mask = predict_mask(model, frame)
        writer.write(overlay_mask(frame, mask))

    cap.release()
    writer.release()
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--image")
    parser.add_argument("--video")
    parser.add_argument("--out", default="annotated_output.mp4")
    args = parser.parse_args()

    m = load_model(args.model)
    if args.image:
        run_on_image(m, args.image)
    elif args.video:
        run_on_video(m, args.video, args.out)
    else:
        print("Provide --image or --video")
