"""
Bootstraps binary lane masks from raw driving frames using the ORIGINAL
repo's classical pipeline (Canny edges -> ROI mask -> Hough lines), so the
U-Net has something to learn from without needing a hand-labeled dataset
like TuSimple or CULane on day one.

This is a standard trick for "classical -> deep learning" upgrades: use the
weaker method as a noisy teacher, train a model on its outputs, and the
learned model ends up MORE robust than the teacher because the network
generalizes past the teacher's specific failure modes (it's trained on
thousands of frames, not tuned thresholds for one lighting condition).

Recommended path: run this on the frames from `Data/` (or a fresh capture
in the simulator's Training Mode) to get a first-pass dataset, train the
U-Net on it, then — if you want to go further — fine-tune that same model
on a small batch of ~200-500 HAND-LABELED masks (e.g. via CVAT or Labelbox).
That fine-tune is what actually surpasses the classical method; the
pseudo-label pass alone mostly matches it while being learnable and fast.

Usage:
    python generate_pseudo_masks.py --images_dir Data/IMG --out_dir Data/masks
"""

import argparse
import os

import cv2
import numpy as np
from tqdm import tqdm


def region_of_interest(edges):
    h, w = edges.shape
    mask = np.zeros_like(edges)
    polygon = np.array([[
        (0, h),
        (w, h),
        (int(w * 0.55), int(h * 0.6)),
        (int(w * 0.45), int(h * 0.6)),
    ]], dtype=np.int32)
    cv2.fillPoly(mask, polygon, 255)
    return cv2.bitwise_and(edges, mask)


def make_mask(image_bgr):
    """Canny + ROI + Hough, same spirit as the original 1.Finding_Lanes notebook."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    roi_edges = region_of_interest(edges)

    lines = cv2.HoughLinesP(
        roi_edges, rho=2, theta=np.pi / 180, threshold=50,
        minLineLength=40, maxLineGap=100,
    )

    mask = np.zeros_like(gray)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(mask, (x1, y1), (x2, y2), 255, thickness=8)

    return mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    filenames = [f for f in os.listdir(args.images_dir) if f.lower().endswith((".jpg", ".png"))]

    kept = 0
    for fname in tqdm(filenames, desc="Generating pseudo-masks"):
        img_path = os.path.join(args.images_dir, fname)
        image = cv2.imread(img_path)
        if image is None:
            continue

        mask = make_mask(image)

        # Skip frames where Hough found nothing usable — training on empty
        # masks in bulk would just teach the network to predict blank.
        if mask.sum() < 500:
            continue

        out_path = os.path.join(args.out_dir, fname.rsplit(".", 1)[0] + "_mask.png")
        cv2.imwrite(out_path, mask)
        kept += 1

    print(f"Wrote {kept}/{len(filenames)} masks to {args.out_dir}")


if __name__ == "__main__":
    main()
