import logging
from dataclasses import dataclass
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class HipPosition:
    """Represents the hip center position."""

    y: float  # Normalized Y position (0=top, 1=bottom)
    left_y: float
    right_y: float
    confidence: float


class PoseDetector:
    """MediaPipe pose estimation wrapper for jump detection."""

    # Hip landmark indices in MediaPipe pose
    LEFT_HIP = 23
    RIGHT_HIP = 24

    def __init__(
        self,
        model_complexity: int = 0,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        logger.info(
            f"Initialized PoseDetector with model_complexity={model_complexity}"
        )

    def process_frame(self, frame: np.ndarray) -> Optional[HipPosition]:
        """
        Process a frame and extract hip position.

        Args:
            frame: BGR image from OpenCV

        Returns:
            HipPosition if detected, None otherwise
        """
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)

        if not results.pose_landmarks:
            return None

        landmarks = results.pose_landmarks.landmark

        left_hip = landmarks[self.LEFT_HIP]
        right_hip = landmarks[self.RIGHT_HIP]

        # Calculate confidence as average visibility
        confidence = (left_hip.visibility + right_hip.visibility) / 2

        if confidence < 0.3:
            return None

        # Calculate hip center Y position
        hip_y = (left_hip.y + right_hip.y) / 2

        return HipPosition(
            y=hip_y,
            left_y=left_hip.y,
            right_y=right_hip.y,
            confidence=confidence,
        )

    def close(self) -> None:
        """Release resources."""
        self.pose.close()
