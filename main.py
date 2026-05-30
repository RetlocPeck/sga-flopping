import random
import time
from pathlib import Path

import cv2
import numpy as np

from pose_detector import PoseDetector, ShotDetector, POSE_CONNECTIONS
from media_player import VideoPlayer

ROOT = Path(__file__).parent
ASSETS = ROOT / "assets"
DEFAULT_IMG = ASSETS / "default.png"
POOL_A_DIR = ASSETS / "pool_a"
POOL_B_DIR = ASSETS / "pool_b"
VIDEO_EXTS = {'.mp4', '.mov', '.mkv', '.avi', '.webm', '.m4v'}

DISPLAY_W, DISPLAY_H = 1600, 900
WEBCAM_W, WEBCAM_H = 480, 360
COOLDOWN = 10.0

MAIN_WINDOW = "SGA Flopping"
WEBCAM_WINDOW = "Webcam preview"


def draw_skeleton(img, landmarks):
    h, w = img.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in POSE_CONNECTIONS:
        if landmarks[a].visibility > 0.3 and landmarks[b].visibility > 0.3:
            cv2.line(img, pts[a], pts[b], (200, 255, 200), 2)
    for i, (x, y) in enumerate(pts):
        if landmarks[i].visibility > 0.3:
            cv2.circle(img, (x, y), 3, (0, 200, 255), -1)


def ensure_assets():
    ASSETS.mkdir(exist_ok=True)
    POOL_A_DIR.mkdir(exist_ok=True)
    POOL_B_DIR.mkdir(exist_ok=True)
    if not DEFAULT_IMG.exists():
        img = np.full((DISPLAY_H, DISPLAY_W, 3), (30, 30, 50), dtype=np.uint8)
        cv2.putText(img, "SGA FLOPPING", (380, 420),
                    cv2.FONT_HERSHEY_SIMPLEX, 3.5, (255, 255, 255), 10)
        cv2.putText(img, "Waiting for shooting motion...",
                    (380, 540), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (200, 200, 200), 3)
        cv2.imwrite(str(DEFAULT_IMG), img)


def list_videos(folder):
    if not folder.exists():
        return []
    return sorted([p for p in folder.iterdir() if p.suffix.lower() in VIDEO_EXTS])


def fit_image(img, target_w, target_h):
    h, w = img.shape[:2]
    scale = min(target_w / w, target_h / h)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(img, (nw, nh))
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    x = (target_w - nw) // 2
    y = (target_h - nh) // 2
    canvas[y:y + nh, x:x + nw] = resized
    return canvas


def pick_video(pool, last=None):
    if not pool:
        return None
    if len(pool) == 1 or last is None:
        return random.choice(pool)
    choices = [p for p in pool if p != last] or pool
    return random.choice(choices)


def main():
    ensure_assets()

    default_img = cv2.imread(str(DEFAULT_IMG))
    default_canvas = fit_image(default_img, DISPLAY_W, DISPLAY_H)

    pool_a = list_videos(POOL_A_DIR)
    pool_b = list_videos(POOL_B_DIR)
    print(f"Pool A (foul bait):  {len(pool_a)} videos in {POOL_A_DIR}")
    print(f"Pool B (free throw): {len(pool_b)} videos in {POOL_B_DIR}")
    print("Drop .mp4/.mov/.mkv files into those folders, then re-run if needed.")
    print("Keys:  Q = quit  |  R = reset state  |  H = toggle webcam window")

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam (index 0)")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    pose = PoseDetector()
    shot = ShotDetector()

    cv2.namedWindow(MAIN_WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(MAIN_WINDOW, DISPLAY_W, DISPLAY_H)
    cv2.namedWindow(WEBCAM_WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WEBCAM_WINDOW, WEBCAM_W, WEBCAM_H)
    webcam_visible = True

    # States: IDLE_A -> PLAYING_A -> COOLDOWN_TO_B1 -> IDLE_B1 -> PLAYING_B1
    #      -> COOLDOWN_TO_B2 -> IDLE_B2 -> PLAYING_B2 -> COOLDOWN_TO_A -> IDLE_A
    state = "IDLE_A"
    state_entered = time.monotonic()
    player = None
    last_played = {"A": None, "B": None}

    def start_video(pool, label, pool_key):
        nonlocal player, state, state_entered
        path = pick_video(pool, last_played[pool_key])
        if path is None:
            print(f"[!] Pool {pool_key} is empty — skipping to cooldown")
            transition_after_video(label)
            return
        last_played[pool_key] = path
        print(f"[{label}] Playing {path.name}")
        player = VideoPlayer(path)
        state = f"PLAYING_{label}"
        state_entered = time.monotonic()

    def transition_after_video(label):
        nonlocal state, state_entered
        next_state = {
            "A": "COOLDOWN_TO_B1",
            "B1": "COOLDOWN_TO_B2",
            "B2": "COOLDOWN_TO_A",
        }[label]
        state = next_state
        state_entered = time.monotonic()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)

            playing = state.startswith("PLAYING_")
            landmarks = None
            if not playing:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                landmarks = pose.detect(rgb)

            if webcam_visible:
                preview = frame.copy()
                if landmarks is not None:
                    draw_skeleton(preview, landmarks)
                preview = cv2.resize(preview, (WEBCAM_W, WEBCAM_H))
                color = (0, 255, 0) if state.startswith("IDLE_") else (0, 200, 255)
                cv2.putText(preview, state, (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                cv2.imshow(WEBCAM_WINDOW, preview)

            now = time.monotonic()

            if state == "IDLE_A":
                cv2.imshow(MAIN_WINDOW, default_canvas)
                if shot.update(landmarks):
                    start_video(pool_a, "A", "A")

            elif state == "IDLE_B1":
                cv2.imshow(MAIN_WINDOW, default_canvas)
                if shot.update(landmarks):
                    start_video(pool_b, "B1", "B")

            elif state == "IDLE_B2":
                cv2.imshow(MAIN_WINDOW, default_canvas)
                if shot.update(landmarks):
                    start_video(pool_b, "B2", "B")

            elif playing:
                vframe = player.get_frame()
                if player.done:
                    player.close()
                    player = None
                    label = state.split("_", 1)[1]
                    transition_after_video(label)
                    cv2.imshow(MAIN_WINDOW, default_canvas)
                elif vframe is not None:
                    cv2.imshow(MAIN_WINDOW, fit_image(vframe, DISPLAY_W, DISPLAY_H))
                else:
                    cv2.imshow(MAIN_WINDOW, default_canvas)

            elif state.startswith("COOLDOWN_TO_"):
                cv2.imshow(MAIN_WINDOW, default_canvas)
                if now - state_entered >= COOLDOWN:
                    state = "IDLE_" + state[len("COOLDOWN_TO_"):]
                    state_entered = now
                    shot.reset()

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                if player:
                    player.close()
                    player = None
                state = "IDLE_A"
                state_entered = time.monotonic()
                shot.reset()
                print("[reset] back to IDLE_A")
            elif key == ord('h'):
                webcam_visible = not webcam_visible
                if not webcam_visible:
                    cv2.destroyWindow(WEBCAM_WINDOW)
                else:
                    cv2.namedWindow(WEBCAM_WINDOW, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(WEBCAM_WINDOW, WEBCAM_W, WEBCAM_H)
    finally:
        if player:
            player.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
