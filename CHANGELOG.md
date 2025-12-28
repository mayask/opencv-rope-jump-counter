# Algorithm Changelog

Document all algorithm changes, bugs, and findings with timestamps.

---

## 2025-12-28

### Fixed: False positives from walking toward camera
- **Issue**: `0-false-positive-02.mp4` detected 6 jumps when person was just walking
- **Root Cause**: Y drift reached 24%, threshold was 35% - walking created rhythmic oscillation
- **Fix**: Tightened `max_y_drift` from 0.35 to 0.20 in `jump.py`

### Added: Pose caching for fast tests
- **Change**: Tests now cache pose detections to JSON files
- **Benefit**: Test runs reduced from minutes to seconds
- **Commands**: `make test` (uses cache), `make test-cache` (regenerates)

---

## 2025-12-27

### Fixed: 12 false positive jumps when no one present
- **Issue**: Camera running all day, detected 12 jumps with no one jumping
- **Incidents**:
  - 11:36:27-11:36:38: 8 jumps (head moving across frame)
  - 11:44:06-11:44:12: 4 jumps (extremely low confidence 0.04-0.05)
- **Root Cause**: Head keypoint confidence not being filtered
- **Fix**: Added `min_head_confidence = 0.3` in `pose.py:167-171`

### Tuned: Parameters for 34-35 jumps test video
- **Issue**: Only detecting 16-28 jumps instead of 35
- **Changes**:
  - `x_history`: 30 → 15 frames (1s window was too long, natural drift rejected)
  - `max_x_drift`: 8% → 25% (allow natural horizontal movement)
  - `max_y_drift`: 20% → 35% (allow variable speed jumping) *Note: reverted to 20% on 2025-12-28*
  - `rhythm_tolerance`: 50% → 60% (allow more interval variation)
  - Rhythm interval max: 0.7s → 0.9s (allow slower jumping)

---

## 2025-12-26

### Added: Motion filter for static object rejection
- **Issue**: Static objects (furniture, posters) detected as humans
- **Fix**: Track position history, require 2% Y movement before accepting
- **Location**: `pose.py:169-191`

### Added: Distance-independent detection
- **Change**: Normalize all measurements by person's bounding box
- **Implementation**:
  - Amplitude: normalized by `bbox_height`
  - X drift: normalized by `bbox_width`
  - Y drift: normalized by `bbox_height`
- **Benefit**: Works whether person is close or far from camera

---

## Initial Implementation

### Chose YOLOv8-pose over MediaPipe
- **Reason**: Much better at distinguishing real humans from objects
- **Model**: `yolov8n-pose.pt` (nano, optimized for speed)

### Head center tracking
- **Decision**: Track weighted average of head keypoints (nose, eyes, ears)
- **Reason**: More robust than single point when keypoints are occluded
- **Location**: `pose.py:161-163`

### Rhythm validation
- **Decision**: Require 4 consistent oscillations before counting
- **Parameters**:
  - `confirmation_jumps = 4`
  - `rhythm_tolerance = 0.6` (60% interval deviation allowed)
  - Interval range: 0.15s - 0.9s
- **Reason**: Filter random movements, only count intentional jumping

---

## Current Thresholds Reference

| Parameter | Value | File | Purpose |
|-----------|-------|------|---------|
| `min_head_confidence` | 0.3 | pose.py | Reject noisy detections |
| `min_amplitude` | 0.06 | jump.py | Min 6% of bbox height |
| `max_amplitude` | 0.25 | jump.py | Max 25% of bbox height |
| `max_x_drift` | 0.25 | jump.py | Max 25% of bbox width |
| `max_y_drift` | 0.20 | jump.py | Max 20% of bbox height |
| `confirmation_jumps` | 4 | jump.py | Oscillations before counting |
| `rhythm_tolerance` | 0.6 | jump.py | 60% interval deviation |
| `min_jump_gap` | 0.15 | jump.py | Min 0.15s between jumps |
| `max_jump_interval` | 1.5 | jump.py | Max 1.5s between jumps |
