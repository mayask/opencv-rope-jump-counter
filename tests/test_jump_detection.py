"""
Acceptance tests for jump detection.

Test files should be placed in tests/fixtures/ with naming convention:
- N-description.mp4: expects exactly N jumps
- N-M-description.mp4: expects between N and M jumps (inclusive)

Examples:
- 35-jumping-in-garage.mp4: expects exactly 35 jumps
- 30-40-jumping-variable-speed.mp4: expects 30-40 jumps
- 0-walking-around.mp4: expects 0 jumps (no false positives)

Pose detections are cached to .json files for fast test runs.
Run `make test-cache` to regenerate cache after algorithm changes.
"""

import json
import re
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False

from src.detection.jump import JumpDetector
from src.detection.pose import BodyPosition

FIXTURES_DIR = Path(__file__).parent / "fixtures"
CACHE_DIR = Path(__file__).parent / "fixtures" / ".cache"


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


def get_cache_path(video_path: Path) -> Path:
    """Get cache file path for a video."""
    return CACHE_DIR / f"{video_path.stem}.json"


def load_cached_poses(video_path: Path) -> list[dict] | None:
    """Load cached pose detections if available."""
    cache_path = get_cache_path(video_path)
    if not cache_path.exists():
        return None

    with open(cache_path) as f:
        return json.load(f)


def generate_pose_cache(video_path: Path) -> list[dict]:
    """Extract poses from video and cache them."""
    import cv2
    from src.detection.pose import PoseDetector

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"\nCaching poses for {video_path.name}: {frame_count} frames @ {fps:.1f} FPS")

    pose_detector = PoseDetector()
    poses = []

    try:
        frames_processed = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            body_position = pose_detector.process_frame(frame)

            if body_position:
                # Convert to regular Python types for JSON serialization
                pose_dict = asdict(body_position)
                pose_dict = {k: float(v) if hasattr(v, 'item') else v for k, v in pose_dict.items()}
                poses.append(pose_dict)
            else:
                poses.append(None)

            frames_processed += 1
            if frames_processed % 100 == 0:
                print(f"  Cached {frames_processed}/{frame_count} frames")

        # Save cache
        CACHE_DIR.mkdir(exist_ok=True)
        cache_path = get_cache_path(video_path)
        with open(cache_path, 'w') as f:
            json.dump({"fps": fps, "poses": poses}, f)

        print(f"  Saved cache: {cache_path}")
        return {"fps": fps, "poses": poses}

    finally:
        cap.release()
        pose_detector.close()


def process_video(video_path: Path, use_cache: bool = True) -> int:
    """
    Process a video file and return the detected jump count.

    Uses cached pose detections for speed if available.
    """
    cache_data = load_cached_poses(video_path) if use_cache else None

    if cache_data is None:
        # No cache - need to generate
        cache_data = generate_pose_cache(video_path)

    fps = cache_data["fps"]
    poses = cache_data["poses"]

    print(f"\nProcessing {video_path.name}: {len(poses)} frames @ {fps:.1f} FPS (cached)")

    clock = SimulatedClock(fps)

    with patch("src.detection.jump.time.time", clock.time):
        jump_detector = JumpDetector()

        for i, pose_data in enumerate(poses):
            clock.advance_frame()

            if pose_data:
                body_position = BodyPosition(**pose_data)
                jump_detector.process(body_position)

            if (i + 1) % 500 == 0:
                print(f"  Processed {i + 1}/{len(poses)} frames, jumps: {jump_detector.session_jumps}")

        print(f"  Final: {len(poses)} frames, {jump_detector.session_jumps} jumps detected")
        return jump_detector.session_jumps


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
            assert detected == min_expected, (
                f"Expected exactly {min_expected} jumps, detected {detected}"
            )
        else:
            assert min_expected <= detected <= max_expected, (
                f"Expected {min_expected}-{max_expected} jumps, detected {detected}"
            )


def test_fixtures_exist():
    """Ensure at least one test fixture exists."""
    videos = get_test_videos()
    if not videos:
        pytest.skip("No test videos in tests/fixtures/. Add mp4 files named like: 35-description.mp4")


def regenerate_all_caches():
    """Regenerate pose cache for all test videos."""
    videos = get_test_videos()
    if not videos:
        print(f"No test videos found in {FIXTURES_DIR}")
        return

    print(f"Regenerating cache for {len(videos)} videos...")
    for video_path in videos:
        generate_pose_cache(video_path)
    print("\nCache regeneration complete!")


# Allow running directly
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--cache":
        regenerate_all_caches()
        exit(0)

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
