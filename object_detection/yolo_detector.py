"""
YOLO object detection for the simulator camera feed — detects other
vehicles and pedestrians so the planning layer has something to react to
(the original repo, and even the CNN-LSTM upgrade, are steering-only and
have zero awareness of other agents on the road).

Uses Ultralytics YOLOv8 with COCO pretrained weights (classes we care
about: person, bicycle, car, motorcycle, bus, truck). No custom training
needed for the simulator's simple vehicle models — COCO's "car" class
generalizes fine to it.

Also does a rough monocular distance estimate from bounding-box geometry,
which is what feeds the MPC safety constraint in planning/mpc_controller.py.
"""

from dataclasses import dataclass
from typing import List

import numpy as np
from ultralytics import YOLO

# COCO class ids we care about for driving
RELEVANT_CLASSES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

# Rough real-world widths (meters) used for the pinhole distance estimate.
# Not calibrated to the simulator's exact camera intrinsics — good enough
# for relative "how urgently should I brake" decisions, not for anything
# safety-certified.
ASSUMED_WIDTH_M = {"person": 0.5, "bicycle": 0.6, "car": 1.8, "motorcycle": 0.8, "bus": 2.5, "truck": 2.5}

FOCAL_LENGTH_PX = 700  # approximate for the Udacity simulator's default camera FOV


@dataclass
class Detection:
    label: str
    confidence: float
    box_xyxy: tuple  # (x1, y1, x2, y2) in pixels
    distance_m: float
    lateral_offset_m: float  # +right / -left, relative to image center


class YoloDetector:
    def __init__(self, weights="yolov8n.pt", conf_threshold=0.4):
        self.model = YOLO(weights)
        self.conf_threshold = conf_threshold

    def _estimate_distance(self, box_xyxy, label, frame_width):
        x1, y1, x2, y2 = box_xyxy
        pixel_width = max(x2 - x1, 1e-3)
        real_width = ASSUMED_WIDTH_M.get(label, 1.8)
        distance_m = (real_width * FOCAL_LENGTH_PX) / pixel_width

        box_center_x = (x1 + x2) / 2
        image_center_x = frame_width / 2
        # crude lateral offset: pixels-off-center scaled by estimated distance / focal length
        lateral_offset_m = ((box_center_x - image_center_x) / FOCAL_LENGTH_PX) * distance_m

        return distance_m, lateral_offset_m

    def detect(self, frame_rgb: np.ndarray) -> List[Detection]:
        results = self.model.predict(frame_rgb, verbose=False, conf=self.conf_threshold)[0]
        detections = []

        for box in results.boxes:
            cls_id = int(box.cls[0])
            if cls_id not in RELEVANT_CLASSES:
                continue

            label = RELEVANT_CLASSES[cls_id]
            conf = float(box.conf[0])
            xyxy = tuple(box.xyxy[0].tolist())
            distance_m, lateral_m = self._estimate_distance(xyxy, label, frame_rgb.shape[1])

            detections.append(Detection(
                label=label, confidence=conf, box_xyxy=xyxy,
                distance_m=distance_m, lateral_offset_m=lateral_m,
            ))

        return detections

    @staticmethod
    def closest_in_path(detections: List[Detection], lane_half_width_m=1.5):
        """
        Returns the nearest detection that's actually in our lane (within
        lane_half_width_m of dead ahead) — this is what the MPC controller
        and behavior layer care about for braking decisions, not every
        object in frame (e.g. oncoming traffic in the other lane).
        """
        in_path = [d for d in detections if abs(d.lateral_offset_m) <= lane_half_width_m]
        if not in_path:
            return None
        return min(in_path, key=lambda d: d.distance_m)


def draw_detections(frame_bgr, detections: List[Detection]):
    import cv2
    for d in detections:
        x1, y1, x2, y2 = [int(v) for v in d.box_xyxy]
        color = (0, 0, 255) if d.distance_m < 8 else (0, 255, 0)
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
        label = f"{d.label} {d.distance_m:.1f}m"
        cv2.putText(frame_bgr, label, (x1, max(y1 - 8, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return frame_bgr
