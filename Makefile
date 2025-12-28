.PHONY: test test-cache test-quick setup

PYTHON = CUDA_VISIBLE_DEVICES="" PYTHONPATH=$(PWD) venv/bin/python

# Run all acceptance tests (uses cached poses - fast)
test:
	$(PYTHON) tests/test_jump_detection.py

# Regenerate pose cache for all videos (slow - runs YOLO inference)
# Run this after changing pose detection algorithm
test-cache:
	$(PYTHON) tests/test_jump_detection.py --cache

# Run quick test on a single video (specify VIDEO=filename.mp4)
test-quick:
	$(PYTHON) -c "from tests.test_jump_detection import process_video; from pathlib import Path; print(process_video(Path('tests/fixtures/$(VIDEO)')))"

# Setup venv and install dependencies
setup:
	uv venv venv
	uv pip install opencv-python-headless ultralytics --python venv/bin/python
