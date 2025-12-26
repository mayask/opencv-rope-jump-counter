"""
Acceptance tests for jump detection.

Test files should be placed in tests/fixtures/ with naming convention:
- N-description.mp4: expects exactly N jumps
- N-M-description.mp4: expects between N and M jumps (inclusive)

Examples:
- 35-jumping-in-garage.mp4: expects exactly 35 jumps
- 30-40-jumping-variable-speed.mp4: expects 30-40 jumps
- 0-walking-around.mp4: expects 0 jumps (no false positives)
"""

import re
import time
from pathlib import Path
from unittest.mock import patch

import cv2

try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False

from src.detection.jump import JumpDetector
from src.detection.pose import PoseDetector


class SimulatedClock:
    """Simulates time progression based on video FPS."""

    def __init__(self, fps: float, start_time: float = 0.0):
        self.fps = fps
        self.frame_duration = 1.0 / fps
        self.current_time = start_time
        self.frame_count = 0

    def advance_frame(self):
        """Advance time by one frame duration."""
        self.frame_count += 1
        self.current_time += self.frame_duration

    def time(self) -> float:
        """Return current simulated time."""
        return self.current_time

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def parse_expected_count(filename: str) -> tuple[int, int]:
    """
    Parse expected jump count from filename.

    Returns (min_expected, max_expected) tuple.
    For exact match (N-desc.mp4), min == max.
    For range (N-M-desc.mp4), returns the range.
    """
    # Match N-M-description or N-description pattern
    match = re.match(r"^(\d+)-(\d+)-", filename)
    if match:
        return int(match.group(1)), int(match.group(2))

    match = re.match(r"^(\d+)-", filename)
    if match:
        count = int(match.group(1))
        return count, count

    raise ValueError(f"Invalid test filename format: {filename}")


def process_video(video_path: Path) -> int:
    """
    Process a video file and return the detected jump count.

    Initializes detector fresh for each video and processes all frames.
    Uses simulated clock based on video FPS for accurate timing.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"\nProcessing {video_path.name}: {frame_count} frames @ {fps:.1f} FPS")

    # Create simulated clock for proper timing
    clock = SimulatedClock(fps)

    # Patch time.time in the jump module to use our simulated clock
    with patch("src.detection.jump.time.time", clock.time):
        pose_detector = PoseDetector()
        jump_detector = JumpDetector()

        try:
            frames_processed = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Advance simulated time for this frame
                clock.advance_frame()

                # Detect pose
                body_position = pose_detector.process_frame(frame)

                # Process jump detection
                if body_position:
                    jump_detector.process(body_position)

                frames_processed += 1

                # Progress indicator
                if frames_processed % 100 == 0:
                    print(f"  Processed {frames_processed}/{frame_count} frames, jumps: {jump_detector.session_jumps}")

            print(f"  Final: {frames_processed} frames processed, {jump_detector.session_jumps} jumps detected")
            return jump_detector.session_jumps

        finally:
            cap.release()
            pose_detector.close()


def get_test_videos() -> list[Path]:
    """Get all mp4 test videos from fixtures directory."""
    if not FIXTURES_DIR.exists():
        return []
    return sorted(FIXTURES_DIR.glob("*.mp4"))


# Generate test cases from fixture files
test_videos = get_test_videos()

if HAS_PYTEST:
    @pytest.mark.parametrize("video_path", test_videos, ids=[v.name for v in test_videos])
    def test_jump_detection(video_path: Path):
        """Test jump detection accuracy against expected count from filename."""
        min_expected, max_expected = parse_expected_count(video_path.name)

        detected = process_video(video_path)

        if min_expected == max_expected:
            # Exact match expected
            assert detected == min_expected, (
                f"Expected exactly {min_expected} jumps, detected {detected}"
            )
        else:
            # Range expected
            assert min_expected <= detected <= max_expected, (
                f"Expected {min_expected}-{max_expected} jumps, detected {detected}"
            )


def test_fixtures_exist():
    """Ensure at least one test fixture exists."""
    videos = get_test_videos()
    if not videos:
        pytest.skip("No test videos in tests/fixtures/. Add mp4 files named like: 35-description.mp4")


# Allow running directly
if __name__ == "__main__":
    videos = get_test_videos()
    if not videos:
        print(f"No test videos found in {FIXTURES_DIR}")
        print("Add mp4 files named like: 35-jumping.mp4 or 30-40-variable.mp4")
        exit(1)

    print(f"Found {len(videos)} test videos")

    results = []
    for video_path in videos:
        min_expected, max_expected = parse_expected_count(video_path.name)
        detected = process_video(video_path)

        if min_expected == max_expected:
            passed = detected == min_expected
            expected_str = str(min_expected)
        else:
            passed = min_expected <= detected <= max_expected
            expected_str = f"{min_expected}-{max_expected}"

        status = "✓ PASS" if passed else "✗ FAIL"
        results.append((video_path.name, expected_str, detected, passed))
        print(f"\n{status}: {video_path.name}")
        print(f"  Expected: {expected_str}, Detected: {detected}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    passed = sum(1 for r in results if r[3])
    failed = len(results) - passed

    for name, expected, detected, ok in results:
        status = "✓" if ok else "✗"
        print(f"  {status} {name}: expected {expected}, got {detected}")

    print(f"\nTotal: {passed}/{len(results)} passed")

    exit(0 if failed == 0 else 1)
