# SGA Flopping

A webcam app that watches for a basketball shooting motion and plays a video. Built for a bit about Shai Gilgeous-Alexander's foul-baiting → free-throw routine.

While idle, it shows a default image. Throw a shooting motion → it plays a random "foul bait" clip. Wait two seconds, motion again → it plays a "free throw" clip. One more shot → another free throw. Then it loops back to foul bait.

## How it works

**Pose detection.** MediaPipe's `PoseLandmarker` (Tasks API) tracks 33 body landmarks per webcam frame. The model file (~6 MB) is downloaded automatically on first run.

**Shot detection.** A shot is a two-phase sequence detected within 1.5 seconds:

1. **Load** — wrist below shoulder + elbow bent (< 110°)
2. **Release** — wrist above nose + elbow extended (> 150°)

The detector picks whichever arm has higher landmark visibility on a given frame, so it works whether you're facing the camera straight on or rotated up to ~90°.

**State machine.**

```
IDLE_A     --shot-->  PLAYING_A   --video ends-->  2s cooldown
IDLE_B1    --shot-->  PLAYING_B1  --video ends-->  2s cooldown
IDLE_B2    --shot-->  PLAYING_B2  --video ends-->  2s cooldown  -->  back to IDLE_A
```

Pool A is the foul-bait clips; Pool B is the free-throw clips. The same Pool B video won't play twice in a row when there are multiple choices.

## Setup

Requires Python 3.9–3.13.

```
pip install -r requirements.txt
```

Drop your `.mp4` / `.mov` / `.mkv` / `.avi` / `.webm` / `.m4v` files into:

- `assets/pool_a/` — foul-bait clips
- `assets/pool_b/` — free-throw clips

Then run:

```
python main.py
```

A single 1600×900 window opens (titled **SGA Flopping**) that shows the default image, or the current video. A 480×360 webcam preview with the skeleton overlay and current state label is composited into the bottom-right corner so you can see yourself frame up while filming the rest of the screen.

## Controls

| Key | Action |
| --- | ------ |
| `Q` | Quit |
| `R` | Reset to `IDLE_A` |
| `H` | Toggle the webcam overlay |

## Tuning

The shot-detection thresholds live at the top of `pose_detector.py` in the `ShotDetector` class:

- `BENT_ANGLE_MAX` (110°) — max elbow angle that counts as "loaded"
- `EXTENDED_ANGLE_MIN` (150°) — min elbow angle that counts as "released"
- `LOAD_TIMEOUT` (1.5 s) — max time allowed between load and release
- `MIN_VIS` (0.4) — minimum landmark visibility before an arm is trusted

The cooldown is `COOLDOWN` at the top of `main.py`.

## Files

- `main.py` — camera loop, state machine, window + overlay, controls
- `pose_detector.py` — MediaPipe wrapper + shot-motion state machine
- `media_player.py` — `ffpyplayer` wrapper for video + audio playback
- `assets/default.png` — shown while idle
- `assets/pool_a/`, `assets/pool_b/` — drop your videos here
