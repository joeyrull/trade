"""Decode iPhone clips into frames + audio + fps."""
from dataclasses import dataclass
import subprocess
import numpy as np
import cv2


@dataclass
class Clip:
    frames: list      # list of HxWx3 BGR arrays
    audio: np.ndarray # mono float waveform (may be empty)
    fps: float
    sr: int           # audio sample rate (0 if no audio)
    image_size: tuple # (width, height)


def load_clip(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 240.0
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    h, w = (frames[0].shape[:2] if frames else (0, 0))
    audio, sr = _load_audio(path)
    return Clip(frames=frames, audio=audio, fps=fps, sr=sr, image_size=(w, h))


_AUDIO_SR = 16000  # mono sample rate used for cross-correlation sync


def _load_audio(path):
    """Extract mono audio for sync; return (waveform, sr) or (empty, 0).

    Uses ffmpeg (via imageio-ffmpeg) because .mov/.mp4 video containers can't be read by
    soundfile/libsndfile. Falls back to soundfile (for bare WAV/FLAC) then to empty.
    """
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        cmd = [exe, "-i", path, "-vn", "-ac", "1", "-ar", str(_AUDIO_SR),
               "-f", "f32le", "-loglevel", "quiet", "pipe:1"]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        data = np.frombuffer(proc.stdout, dtype=np.float32).astype(np.float64)
        if data.size:
            return data, _AUDIO_SR
    except Exception:
        pass
    try:
        import soundfile as sf
        data, sr = sf.read(path)
        if data.ndim > 1:
            data = data.mean(axis=1)
        return np.asarray(data, dtype=np.float64), sr
    except Exception:
        return np.zeros(0), 0
