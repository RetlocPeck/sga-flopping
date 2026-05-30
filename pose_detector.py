import time
import urllib.request
from pathlib import Path

import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)
MODEL_PATH = Path(__file__).parent / "pose_landmarker_lite.task"

NOSE = 0
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16

POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (24, 26), (26, 28),
    (15, 17), (15, 19), (15, 21), (17, 19),
    (16, 18), (16, 20), (16, 22), (18, 20),
]


def _ensure_model():
    if MODEL_PATH.exists():
        return
    print(f"Downloading pose model to {MODEL_PATH.name} ...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Model downloaded.")


def _angle(a, b, c):
    ba = np.array([a[0] - b[0], a[1] - b[1]])
    bc = np.array([c[0] - b[0], c[1] - b[1]])
    denom = np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9
    cos = np.dot(ba, bc) / denom
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


class PoseDetector:
    def __init__(self):
        _ensure_model()
        options = mp_vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(MODEL_PATH)),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.detector = mp_vision.PoseLandmarker.create_from_options(options)
        self._t0 = time.monotonic()

    def detect(self, frame_rgb):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        ts_ms = int((time.monotonic() - self._t0) * 1000)
        result = self.detector.detect_for_video(mp_image, ts_ms)
        if not result.pose_landmarks:
            return None
        return result.pose_landmarks[0]


class ShotDetector:
    """Load (wrist below shoulder, elbow bent) -> Release (wrist above nose, elbow extended)."""

    LOAD_TIMEOUT = 1.5
    BENT_ANGLE_MAX = 110
    EXTENDED_ANGLE_MIN = 150
    MIN_VIS = 0.4

    def __init__(self):
        self.state = "IDLE"
        self.load_time = 0.0

    def reset(self):
        self.state = "IDLE"

    def update(self, landmarks):
        if landmarks is None:
            return False

        nose_y = landmarks[NOSE].y

        l_vis = (landmarks[L_SHOULDER].visibility +
                 landmarks[L_ELBOW].visibility +
                 landmarks[L_WRIST].visibility) / 3
        r_vis = (landmarks[R_SHOULDER].visibility +
                 landmarks[R_ELBOW].visibility +
                 landmarks[R_WRIST].visibility) / 3
        if max(l_vis, r_vis) < self.MIN_VIS:
            return False

        if l_vis >= r_vis:
            s, e, w = landmarks[L_SHOULDER], landmarks[L_ELBOW], landmarks[L_WRIST]
        else:
            s, e, w = landmarks[R_SHOULDER], landmarks[R_ELBOW], landmarks[R_WRIST]

        shoulder = (s.x, s.y)
        elbow = (e.x, e.y)
        wrist = (w.x, w.y)
        elbow_angle = _angle(shoulder, elbow, wrist)

        now = time.monotonic()
        if self.state == "IDLE":
            if wrist[1] > shoulder[1] and elbow_angle < self.BENT_ANGLE_MAX:
                self.state = "LOADED"
                self.load_time = now
            return False

        if now - self.load_time > self.LOAD_TIMEOUT:
            self.state = "IDLE"
            return False

        if wrist[1] < nose_y and elbow_angle > self.EXTENDED_ANGLE_MIN:
            self.state = "IDLE"
            return True

        return False
