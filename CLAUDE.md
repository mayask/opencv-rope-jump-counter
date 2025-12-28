# Development Guidelines

## Algorithm Changes

**IMPORTANT**: Before making any algorithm changes, read `CHANGELOG.md` to understand past decisions and why they were made.

When making changes to detection logic:
1. Read `CHANGELOG.md` first - understand existing thresholds and why they exist
2. Run `make test` before and after changes
3. Add a timestamped entry to `CHANGELOG.md` with:
   - Date header: `## YYYY-MM-DD`
   - Entry type: `### Fixed:`, `### Added:`, `### Changed:`, `### Tuned:`
   - What changed, why, what bug it fixed, any trade-offs
4. Update the "Current Thresholds Reference" table if thresholds changed

Files that affect detection:
- `src/detection/pose.py` - pose extraction, confidence filtering, motion filter
- `src/detection/jump.py` - jump detection, amplitude/drift/rhythm validation

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
