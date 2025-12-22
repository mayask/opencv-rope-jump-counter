import logging
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from ultralytics import YOLO

logger = logging.getLogger(__name__)


@dataclass
class BodyPosition:
    """Represents tracked body position for jump detection."""

    x: float  # Normalized X position (0=left, 1=right)
    y: float  # Normalized Y position (0=top, 1=bottom)
    confidence: float
    landmark_name: str


class PoseDetector:
    """YOLOv8-pose wrapper for jump detection.

    Uses YOLO's human detection + pose estimation which is much better
    at distinguishing real humans from objects like treadmills.
    """

    # YOLO pose keypoint indices
    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16

    def __init__(
        self,
        model_complexity: int = 1,  # Ignored, kept for API compatibility
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,  # Ignored, kept for API compatibility
        track_point: str = "nose",
    ):
        # Load YOLOv8 pose model (downloads automatically on first run)
        # Using yolov8n-pose for speed (15+ FPS), yolov8m-pose for better accuracy
        model_name = "yolov8n-pose.pt"
        logger.info(f"Loading YOLO pose model: {model_name}")
        self.model = YOLO(model_name)

        self.min_confidence = min_detection_confidence
        self.track_point = track_point

        # For debug visualization
        self.last_results = None
        self.last_rejection_reason = None
        self.last_keypoints = None
        self.last_bbox = None

        # Motion tracking - detect static objects
        self._position_history: list[tuple[float, float]] = []
        self._has_seen_movement = False
        self._log_counter = 0

        logger.info(
            f"Initialized PoseDetector: model={model_name}, min_conf={min_detection_confidence}, tracking={track_point}"
        )

    def process_frame(self, frame: np.ndarray) -> Optional[BodyPosition]:
        """Process a frame and extract body position."""
        h, w = frame.shape[:2]

        # Run YOLO inference
        results = self.model(frame, verbose=False, conf=self.min_confidence)

        # Store for debug
        self.last_results = results
        self.last_rejection_reason = None
        self.last_keypoints = None
        self.last_bbox = None
        self._log_counter += 1

        # Check if any person detected
        if len(results) == 0 or results[0].keypoints is None:
            self.last_rejection_reason = "No person detected"
            return None

        keypoints = results[0].keypoints
        boxes = results[0].boxes

        if keypoints.xy is None or len(keypoints.xy) == 0:
            self.last_rejection_reason = "No keypoints"
            return None

        # Get the most confident person detection
        if boxes.conf is not None and len(boxes.conf) > 0:
            best_idx = boxes.conf.argmax().item()
            person_conf = boxes.conf[best_idx].item()
        else:
            best_idx = 0
            person_conf = 0.5

        # Get keypoints for best detection
        kpts = keypoints.xy[best_idx].cpu().numpy()  # Shape: (17, 2)
        kpts_conf = keypoints.conf[best_idx].cpu().numpy() if keypoints.conf is not None else None

        # Store for debug visualization
        self.last_keypoints = kpts
        if boxes.xyxy is not None and len(boxes.xyxy) > best_idx:
            self.last_bbox = boxes.xyxy[best_idx].cpu().numpy()

        # Get nose position
        nose_x, nose_y = kpts[self.NOSE]
        nose_conf = kpts_conf[self.NOSE] if kpts_conf is not None else person_conf

        # Check if nose was detected (coordinates are 0,0 if not visible)
        if nose_x == 0 and nose_y == 0:
            self.last_rejection_reason = "Nose not visible"
            return None

        # Normalize coordinates
        norm_x = nose_x / w
        norm_y = nose_y / h

        # Motion filter - reject static objects
        self._position_history.append((norm_x, norm_y))
        if len(self._position_history) > 30:
            self._position_history.pop(0)

        # Need at least 15 frames to check movement
        if len(self._position_history) < 15:
            self.last_rejection_reason = f"Warming up ({len(self._position_history)}/15)"
            return None

        y_vals = [p[1] for p in self._position_history]
        y_range = max(y_vals) - min(y_vals)

        # Must see significant movement (0.02 = 2%) to be considered a real person
        if y_range >= 0.02:
            self._has_seen_movement = True

        # Reject if we've never seen movement - static object
        if not self._has_seen_movement:
            self.last_rejection_reason = f"No movement yet (y_range={y_range:.3f})"
            if self._log_counter % 30 == 0:
                print(f"[POSE] Rejected - no movement: y_range={y_range:.3f}", flush=True)
            return None

        # Also reject if currently static (person stopped)
        if y_range < 0.01:
            self.last_rejection_reason = f"Static (y_range={y_range:.3f})"
            return None

        if self._log_counter % 30 == 0:
            print(f"[POSE] Detected: nose=({norm_x:.2f}, {norm_y:.2f}), conf={nose_conf:.2f}, person_conf={person_conf:.2f}", flush=True)

        return BodyPosition(
            x=norm_x,
            y=norm_y,
            confidence=nose_conf,
            landmark_name="nose",
        )

    def close(self) -> None:
        """Release resources."""
        pass  # YOLO doesn't need explicit cleanup
