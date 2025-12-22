import logging
from dataclasses import dataclass
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BodyPosition:
    """Represents tracked body position for jump detection."""

    x: float  # Normalized X position (0=left, 1=right)
    y: float  # Normalized Y position (0=top, 1=bottom)
    confidence: float
    landmark_name: str


class PoseDetector:
    """MediaPipe pose estimation wrapper for jump detection."""

    NOSE = 0

    def __init__(
        self,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        track_point: str = "nose",
    ):
        self.mp_pose = mp.solutions.pose
        # Enable segmentation to verify actual human pixels exist
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            enable_segmentation=True,  # Get segmentation mask
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.track_point = track_point

        # For debug visualization
        self.last_results = None
        self.last_rejection_reason = None
        self.last_segmentation_ratio = 0.0

        # Motion tracking - detect static objects
        self.position_history: list[tuple[float, float]] = []
        self.motion_window = 30  # ~1 second of frames
        self.min_motion = 0.02  # Require 2% movement to be "alive"
        self.is_moving = False

        logger.info(
            f"Initialized PoseDetector: model={model_complexity}, tracking={track_point}, segmentation=enabled"
        )

    def process_frame(self, frame: np.ndarray) -> Optional[BodyPosition]:
        """Process a frame and extract body position."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)

        # Store for debug
        self.last_results = results
        self.last_rejection_reason = None
        self.last_segmentation_ratio = 0.0

        if not results.pose_landmarks:
            self.last_rejection_reason = "No pose detected"
            return None

        nose = results.pose_landmarks.landmark[self.NOSE]

        # Log every 30 frames
        if not hasattr(self, '_log_counter'):
            self._log_counter = 0
        self._log_counter += 1

        # Motion filter - track position and reject static objects (like treadmill)
        if not hasattr(self, '_position_history'):
            self._position_history = []

        self._position_history.append((nose.x, nose.y))
        if len(self._position_history) > 30:  # ~2 sec window
            self._position_history.pop(0)

        if len(self._position_history) >= 30:
            y_vals = [p[1] for p in self._position_history]
            y_range = max(y_vals) - min(y_vals)

            # Reject if no significant vertical movement (static object)
            if y_range < 0.01:  # Less than 1% movement = static
                self.last_rejection_reason = f"Static (y_range={y_range:.3f})"
                if self._log_counter % 30 == 0:
                    print(f"[POSE] Rejected static: y_range={y_range:.3f}", flush=True)
                return None

        if self._log_counter % 30 == 0:
            print(f"[POSE] Detected: nose=({nose.x:.2f}, {nose.y:.2f}), vis={nose.visibility:.2f}", flush=True)

        return BodyPosition(
            x=nose.x,
            y=nose.y,
            confidence=nose.visibility,
            landmark_name="nose",
        )

    def _get_average_visibility(self, landmarks) -> float:
        """Calculate average visibility of key body landmarks."""
        key_indices = [
            self.NOSE,
            11, 12,  # shoulders
            23, 24,  # hips
            25, 26,  # knees
        ]
        visibilities = [landmarks.landmark[i].visibility for i in key_indices]
        return sum(visibilities) / len(visibilities)

    def close(self) -> None:
        """Release resources."""
        self.pose.close()
