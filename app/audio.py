from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from app.config import settings

SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg"}
SAMPLE_RATE = 16_000


def _run(command: list[str], timeout: int) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(command, check=True, capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError as exc:
        raise RuntimeError("FFmpeg and ffprobe are required to decode uploaded audio") from exc
    except subprocess.TimeoutExpired as exc:
        raise ValueError("Audio decoding exceeded its time limit") from exc
    except subprocess.CalledProcessError as exc:
        message = (exc.stderr or "").strip()[-300:]
        raise ValueError(f"Audio could not be decoded: {message or 'invalid audio file'}") from exc


def normalize_audio(source: Path, original_name: str, destination: Path) -> tuple[float, dict]:
    extension = Path(original_name).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError("Supported formats: WAV, MP3, M4A, AAC, FLAC, and OGG")
    probe = _run([
        "ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
        "stream=sample_rate,channels:format=duration", "-of", "json", str(source),
    ], timeout=30)
    try:
        metadata = json.loads(probe.stdout)
        duration = float(metadata["format"]["duration"])
        streams = metadata.get("streams", [])
        source_rate = int(streams[0]["sample_rate"]) if streams else 0
        source_channels = int(streams[0]["channels"]) if streams else 0
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise ValueError("Audio metadata could not be read") from exc
    if duration < 1.0:
        raise ValueError("Audio must be at least one second long")
    if duration > settings.max_audio_seconds:
        raise ValueError(f"Audio exceeds the {settings.max_audio_seconds}-second limit")
    if source_rate < 8_000 or source_rate > 192_000 or source_channels < 1 or source_channels > 8:
        raise ValueError("Unsupported source sample rate or channel count")

    _run([
        "ffmpeg", "-v", "error", "-nostdin", "-y", "-i", str(source), "-map", "0:a:0",
        "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le", str(destination),
    ], timeout=min(600, max(60, int(duration * 4))))
    with sf.SoundFile(destination) as wav:
        if wav.samplerate != SAMPLE_RATE or wav.channels != 1 or wav.frames == 0:
            raise ValueError("Audio conversion did not produce valid 16 kHz mono PCM")
        duration = wav.frames / wav.samplerate
        frame_rms: list[float] = []
        clipped, total = 0, 0
        for block in wav.blocks(blocksize=SAMPLE_RATE // 4, dtype="float32"):
            if not np.isfinite(block).all():
                raise ValueError("Audio contains invalid samples")
            if block.size:
                frame_rms.append(float(np.sqrt(np.mean(np.square(block)))))
                clipped += int(np.count_nonzero(np.abs(block) >= 0.999))
                total += int(block.size)
    if not frame_rms or max(frame_rms) < 0.003:
        raise ValueError("Audio is silent or too quiet for voice matching")
    activity_floor = max(0.003, float(np.percentile(frame_rms, 20)) * 0.35)
    active_fraction = sum(rms >= activity_floor for rms in frame_rms) / len(frame_rms)
    usable_seconds = duration * active_fraction
    if usable_seconds < settings.min_usable_speech_seconds:
        raise ValueError(
            f"INSUFFICIENT_AUDIO: approximately {usable_seconds:.1f}s of active audio; "
            f"at least {settings.min_usable_speech_seconds:g}s is required"
        )
    quality = {
        "duration_seconds": round(duration, 3),
        "usable_speech_seconds_estimate": round(usable_seconds, 3),
        "active_frame_fraction": round(active_fraction, 3),
        "clipping_fraction": round(clipped / max(total, 1), 6),
        "source_sample_rate": source_rate,
        "source_channels": source_channels,
        "normalized_sample_rate": SAMPLE_RATE,
    }
    if usable_seconds < settings.recommended_enrollment_seconds:
        quality["warning"] = "Shorter than the configured recommended enrollment duration; matches may be less reliable."
    if quality["clipping_fraction"] >= 0.005:
        quality["warning"] = "Clipping detected; review this sample before relying on its match."
    return duration, quality


class TemporaryAudio:
    """Disk-backed upload and normalized WAV paths with cleanup on exit."""

    def __init__(self, suffix: str):
        self._directory = tempfile.TemporaryDirectory(prefix="speaker-api-")
        root = Path(self._directory.name)
        self.source = root / f"upload{suffix}"
        self.normalized = root / "normalized.wav"

    def cleanup(self) -> None:
        self._directory.cleanup()
