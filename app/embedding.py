from __future__ import annotations

import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import numpy as np
import soundfile as sf

from app.config import settings


class InferenceBusy(RuntimeError):
    pass


class SpeechBrainEncoder:
    """One lazily initialized ECAPA model per process and bounded parallel inference."""

    def __init__(self) -> None:
        self._model = None
        self._load_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._active = 0
        self._waiting = 0
        self._slots = threading.BoundedSemaphore(settings.max_concurrent_inferences)
        self._queue_slots = threading.BoundedSemaphore(
            settings.max_concurrent_inferences + settings.max_queued_inferences
        )

    @property
    def model_loaded(self) -> bool:
        return self._model is not None

    def workload(self) -> dict[str, int]:
        with self._state_lock:
            return {"active_inferences": self._active, "queued_inferences": self._waiting}

    @contextmanager
    def inference_slot(self) -> Iterator[None]:
        """Reserve one of the bounded CPU/GPU work slots before decoding audio."""
        if not self._queue_slots.acquire(blocking=False):
            raise InferenceBusy("Inference queue is full; retry shortly")
        acquired = False
        waiting = True
        with self._state_lock:
            self._waiting += 1
        try:
            acquired = self._slots.acquire(timeout=settings.queue_wait_seconds)
            if not acquired:
                raise InferenceBusy("Timed out waiting for a free inference slot")
            with self._state_lock:
                self._waiting -= 1
                waiting = False
                self._active += 1
            yield
        finally:
            with self._state_lock:
                if acquired:
                    self._active -= 1
                elif waiting:
                    self._waiting -= 1
            if acquired:
                self._slots.release()
            self._queue_slots.release()

    def _load_model(self):
        if self._model is not None:
            return self._model
        with self._load_lock:
            if self._model is not None:
                return self._model
            try:
                import torch
                from speechbrain.inference.speaker import EncoderClassifier

                torch.set_num_threads(settings.torch_num_threads)
                device = settings.device
                if device == "cuda" and not torch.cuda.is_available():
                    device = "cpu"
                self._model = EncoderClassifier.from_hparams(
                    source=settings.model_id,
                    savedir=str(Path(settings.model_cache_dir).resolve()),
                    run_opts={"device": device},
                    revision=settings.model_revision,
                )
            except Exception as exc:
                raise RuntimeError("SpeechBrain ECAPA model failed to load") from exc
        return self._model

    def extract(self, wav_path: Path) -> np.ndarray:
        model = self._load_model()
        import torch

        with sf.SoundFile(wav_path) as wav:
            if wav.samplerate != 16_000 or wav.channels != 1:
                raise ValueError("Model input must be normalized to 16 kHz mono")
            audio = wav.read(dtype="float32")
        if len(audio) < 16_000 or not np.isfinite(audio).all():
            raise ValueError("Audio is too short or invalid for speaker embedding")
        waveform = torch.from_numpy(np.ascontiguousarray(audio)).unsqueeze(0)
        device = next(model.mods.parameters()).device
        waveform = waveform.to(device)
        with torch.inference_mode():
            result = model.encode_batch(waveform)
        vector = result.detach().float().cpu().numpy().reshape(-1).astype(np.float32)
        norm = float(np.linalg.norm(vector))
        if not vector.size or not np.isfinite(vector).all() or norm <= 1e-12:
            raise RuntimeError("Model returned an invalid speaker embedding")
        return vector / norm


encoder = SpeechBrainEncoder()
