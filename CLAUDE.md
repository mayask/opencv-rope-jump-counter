# Development Guidelines

## Testing

Always use Makefile commands for running tests. Never use Docker for tests.

```bash
# Run all acceptance tests (fast - uses cached poses)
make test

# Regenerate pose cache (slow - run after changing pose detection)
make test-cache

# Run test on a single video
make test-quick VIDEO=0-false-positive-01.mp4

# Setup venv (first time only)
make setup
```

### When to regenerate cache (`make test-cache`)

**Regenerate cache when changing:**
- `src/detection/pose.py` - pose detection algorithm, confidence thresholds, keypoint processing

**Do NOT regenerate cache when changing:**
- `src/detection/jump.py` - jump detection logic, amplitude/rhythm/drift thresholds
- Any other files - API, config, webhook, etc.

The cache stores raw pose detections (x, y, confidence, bbox). Jump detection logic runs on these cached values, so changes to jump.py are tested immediately with `make test`.

## Environment

- Always use venv, never Docker for local development/testing
- Tests use CPU-only inference (CUDA_VISIBLE_DEVICES="")
- Test videos go in tests/fixtures/ with naming: N-description.mp4 or N-M-description.mp4
- Pose detections are cached in tests/fixtures/.cache/ for fast test runs
