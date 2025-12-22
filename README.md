# Rope Skipping Counter

Counts rope skipping jumps from an RTSP camera feed using YOLOv8 pose detection. Sends webhook notifications to Home Assistant on milestones.

![Demo](demo.gif)

## Features

- Real-time jump detection using head position tracking
- Rhythm validation to filter out walking/standing movements
- Distance-independent detection (works close or far from camera)
- Webhook notifications for milestone achievements
- Debug video stream with pose overlay

## Setup

### 1. Configure environment

Copy `.env.example` to `.env` and set:

```bash
# RTSP stream URL from your camera
RTSP_URL=rtsp://user:pass@camera-ip:554/stream

# Home Assistant webhook URL for notifications
WEBHOOK_URL=http://homeassistant.local:8123/api/webhook/your-webhook-id

# Announce every N jumps (default: 100)
MILESTONE_INTERVAL=100

# Timezone for daily reset
TZ=Europe/Amsterdam
```

### 2. Run with Docker Compose

```bash
# Production (real camera)
docker compose up -d rope-counter

# Test mode (loops a video file)
# Place your test video as camera-stream.mp4
docker compose --profile test up -d
```

Pre-built image available on Docker Hub: [raregoat8804/opencv-rope-jump-counter](https://hub.docker.com/r/raregoat8804/opencv-rope-jump-counter)

### 3. View debug stream

Open `http://localhost:8080/debug/stream` in a browser to see the live pose detection overlay.

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Health check |
| `GET /status` | Full system status |
| `GET /count` | Current jump counts |
| `GET /reset` | Reset all counters |
| `GET /snapshot` | Capture current frame |
| `GET /debug/stream` | Live MJPEG stream with pose overlay |
| `POST /session/start` | Manually start session |
| `POST /session/stop` | Manually stop session |

## Webhook Payload

```json
{
  "event": "milestone",
  "count": 100,
  "milestone": 100,
  "session_jumps": 100,
  "daily_total": 250
}
```

## Detection Algorithm

1. YOLOv8-pose detects human and extracts keypoints
2. Head position tracked using weighted average of visible keypoints (nose, eyes, ears)
3. Jump amplitude normalized to person's bounding box height (distance-independent)
4. Rhythm validation ensures consistent oscillation pattern before counting
5. Horizontal stability check filters out walking/running
