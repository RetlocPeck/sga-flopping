import cv2
import numpy as np
from ffpyplayer.player import MediaPlayer


class VideoPlayer:
    """Plays a video file with audio via ffpyplayer.

    get_frame() returns the most recent BGR frame (or the last one if no
    new frame is ready yet) so the caller can poll on its own loop cadence.
    """

    def __init__(self, path):
        self.player = MediaPlayer(str(path), ff_opts={'out_fmt': 'rgb24'})
        self.last_frame = None
        self.done = False

    def get_frame(self):
        if self.done:
            return None

        frame, val = self.player.get_frame()
        if val == 'eof':
            self.done = True
            self._silence()
            return self.last_frame

        if frame is not None:
            img, _t = frame
            w, h = img.get_size()
            raw = bytes(img.to_bytearray()[0])
            arr = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3)
            self.last_frame = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

        return self.last_frame

    def _silence(self):
        for fn, arg in (("set_pause", True), ("set_mute", True), ("set_volume", 0.0)):
            try:
                getattr(self.player, fn)(arg)
            except Exception:
                pass

    def close(self):
        self.done = True
        self._silence()
        try:
            self.player.close_player()
        except Exception:
            pass
