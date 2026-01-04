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
    bbox_height: float = 0.0  # Bounding box height as fraction of frame (for amplitude normalization)
    bbox_width: float = 0.0  # Bounding box width as fraction of frame (for drift normalization)


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
        self.min_head_confidence = 0.3  # Minimum confidence for head keypoints
        self.track_point = track_point

        # For debug visualization
        self.last_results = None
        self.last_rejection_reason = None
        self.last_keypoints = None
        self.last_bbox = None
        self.all_keypoints = []  # All detected people's keypoints
        self.all_bboxes = []  # All detected people's bboxes
        self.last_best_idx = 0  # Which detection was selected

        # Person tracking - prefer staying with the same person
        self._tracked_position: Optional[tuple[float, float]] = None  # Last known (x, y) of tracked person
        self._tracking_active = False  # Whether we're actively tracking someone

        # Motion tracking - detect static objects
        self._position_history: list[tuple[float, float]] = []
        self._has_seen_movement = False
        self._log_counter = 0

        logger.info(
            f"Initialized PoseDetector: model={model_name}, min_conf={min_detection_confidence}, tracking={track_point}"
        )

    def process_frame(self, frame: np.ndarray) -> Optional[BodyPosition]:
        """Process a frame and extract body position."""
        orig_h, orig_w = frame.shape[:2]
        scale = 1.0

        # Resize for faster inference if needed (skip if already small)
        if orig_w > 640:
            scale = 640 / orig_w
            frame = cv2.resize(frame, (640, int(orig_h * scale)))

        h, w = frame.shape[:2]

        # Run YOLO inference
        results = self.model(frame, verbose=False, conf=self.min_confidence)

        # Store for debug
        self.last_results = results
        self.last_rejection_reason = None
        self.last_keypoints = None
        self.last_bbox = None
        self.all_keypoints = []  # All detected people's keypoints
        self.all_bboxes = []  # All detected people's bboxes
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

        # Store ALL detected people for debug overlay and find best match
        num_detections = len(keypoints.xy)
        head_positions = []  # (idx, head_x, head_y, confidence) for each detection
        
        for i in range(num_detections):
            kpts_i = keypoints.xy[i].cpu().numpy()
            if scale != 1.0:
                kpts_i = kpts_i / scale
            self.all_keypoints.append(kpts_i)
            
            if boxes.xyxy is not None and len(boxes.xyxy) > i:
                bbox_i = boxes.xyxy[i].cpu().numpy()
                if scale != 1.0:
                    bbox_i = bbox_i / scale
                self.all_bboxes.append(bbox_i)
            else:
                self.all_bboxes.append(None)
            
            # Calculate head position for this detection
            head_x, head_y, head_conf = self._get_head_position(kpts_i, orig_w, orig_h)
            if head_x is not None:
                person_conf_i = boxes.conf[i].item() if boxes.conf is not None else 0.5
                head_positions.append((i, head_x, head_y, head_conf, person_conf_i))

        if not head_positions:
            self.last_rejection_reason = "No valid head positions"
            return None

        # Choose which person to track
        if self._tracking_active and self._tracked_position is not None:
            # Find the detection closest to our tracked position
            tracked_x, tracked_y = self._tracked_position
            best_idx = None
            best_dist = float('inf')
            
            for idx, hx, hy, hconf, pconf in head_positions:
                # Distance in normalized coordinates
                dist = ((hx - tracked_x) ** 2 + (hy - tracked_y) ** 2) ** 0.5
                if dist < best_dist:
                    best_dist = dist
                    best_idx = idx
                    person_conf = pconf
            
            # If closest person is too far (>20% of frame), fall back to most confident
            if best_dist > 0.2:
                best_idx = max(head_positions, key=lambda x: x[4])[0]  # Most confident
                person_conf = max(head_positions, key=lambda x: x[4])[4]
        else:
            # Not tracking yet - use most confident detection
            best_idx = max(head_positions, key=lambda x: x[4])[0]
            person_conf = max(head_positions, key=lambda x: x[4])[4]

        # Get keypoints for best detection
        kpts = keypoints.xy[best_idx].cpu().numpy()  # Shape: (17, 2)
        kpts_conf = keypoints.conf[best_idx].cpu().numpy() if keypoints.conf is not None else None

        # Scale keypoints back to original frame dimensions for overlay
        if scale != 1.0:
            kpts = kpts / scale

        # Store for debug visualization and calculate bbox dimensions
        self.last_keypoints = kpts
        self.last_best_idx = best_idx  # Track which detection is being used
        bbox_height_norm = 0.0
        bbox_width_norm = 0.0
        if boxes.xyxy is not None and len(boxes.xyxy) > best_idx:
            bbox = boxes.xyxy[best_idx].cpu().numpy()
            if scale != 1.0:
                bbox = bbox / scale
            self.last_bbox = bbox
            # Calculate bbox dimensions as fraction of frame
            bbox_height_norm = (bbox[3] - bbox[1]) / orig_h
            bbox_width_norm = (bbox[2] - bbox[0]) / orig_w

        # Get head center (average of visible head keypoints for robustness)
        head_keypoint_indices = [self.NOSE, self.LEFT_EYE, self.RIGHT_EYE, self.LEFT_EAR, self.RIGHT_EAR]
        visible_points = []
        total_conf = 0.0

        for idx in head_keypoint_indices:
            x, y = kpts[idx]
            if x > 0 and y > 0:  # Visible if not (0,0)
                conf = kpts_conf[idx] if kpts_conf is not None else person_conf
                visible_points.append((x, y, conf))
                total_conf += conf

        if not visible_points:
            self.last_rejection_reason = "No head keypoints visible"
            return None

        # Weighted average by confidence
        head_x = sum(p[0] * p[2] for p in visible_points) / total_conf
        head_y = sum(p[1] * p[2] for p in visible_points) / total_conf
        avg_conf = total_conf / len(visible_points)

        # Reject low-confidence head detections (prevents false positives from noise)
        if avg_conf < self.min_head_confidence:
            self.last_rejection_reason = f"Low head confidence ({avg_conf:.2f} < {self.min_head_confidence})"
            if self._log_counter % 30 == 0:
                logger.debug(f"Rejected - low head confidence: {avg_conf:.2f}")
            return None

        # Normalize coordinates (use original dimensions since kpts are scaled back)
        norm_x = head_x / orig_w
        norm_y = head_y / orig_h

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
                logger.debug(f"Rejected - no movement: y_range={y_range:.3f}")
            return None

        # Also reject if currently static (person stopped)
        if y_range < 0.01:
            self.last_rejection_reason = f"Static (y_range={y_range:.3f})"
            return None

        if self._log_counter % 30 == 0:
            logger.debug(f"Detected: head=({norm_x:.2f}, {norm_y:.2f}), conf={avg_conf:.2f}, person_conf={person_conf:.2f}")

        # Update tracked position for next frame
        self._tracked_position = (norm_x, norm_y)
        self._tracking_active = True

        return BodyPosition(
            x=norm_x,
            y=norm_y,
            confidence=avg_conf,
            landmark_name="head_center",
            bbox_height=bbox_height_norm,
            bbox_width=bbox_width_norm,
        )

    def _get_head_position(self, kpts: np.ndarray, orig_w: int, orig_h: int) -> tuple[Optional[float], Optional[float], float]:
        """Get normalized head position from keypoints. Returns (x, y, confidence) or (None, None, 0)."""
        head_keypoint_indices = [self.NOSE, self.LEFT_EYE, self.RIGHT_EYE, self.LEFT_EAR, self.RIGHT_EAR]
        visible_points = []
        total_conf = 0.0

        for idx in head_keypoint_indices:
            x, y = kpts[idx]
            if x > 0 and y > 0:
                # Assume reasonable confidence if not available
                visible_points.append((x, y, 0.5))
                total_conf += 0.5

        if not visible_points:
            return None, None, 0.0

        head_x = sum(p[0] * p[2] for p in visible_points) / total_conf
        head_y = sum(p[1] * p[2] for p in visible_points) / total_conf
        avg_conf = total_conf / len(visible_points)

        return head_x / orig_w, head_y / orig_h, avg_conf

    def close(self) -> None:
        """Release resources."""
        pass  # YOLO doesn't need explicit cleanup
