from __future__ import annotations

import json
import logging
import os
import tempfile
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from app.audio import SUPPORTED_EXTENSIONS, TemporaryAudio, normalize_audio
from app.config import settings
from app.database import Base, SessionLocal, engine
from app.embedding import InferenceBusy, encoder
from app.models import SpeakerProfile
from app.security import configured_clients, decrypt_embedding, encrypt_embedding, fernet, get_tenant

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("speechbrain-voice-api")
MODEL_NAME = "speechbrain-ecapa-voxceleb"
MODEL_VERSION = f"{settings.model_id}@{settings.model_revision}"


@asynccontextmanager
async def lifespan(_: FastAPI):
    if not configured_clients():
        raise RuntimeError("Configure at least one private API token in API_KEYS_JSON")
    fernet()  # Validate encryption configuration before accepting requests.
    Base.metadata.create_all(bind=engine)
    yield


app = FastAPI(
    title="SpeechBrain Voice Matching API",
    version="1.0.0",
    description=(
        "Tenant-isolated speaker enrollment and open-set voice matching using local "
        "SpeechBrain ECAPA embeddings. Scores are raw cosine similarities, not probabilities."
    ),
    lifespan=lifespan,
)


async def save_upload(upload: UploadFile, target: Path) -> str:
    filename = upload.filename or "audio.wav"
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(415, "Unsupported audio format")
    total = 0
    try:
        with target.open("wb") as output:
            while chunk := await upload.read(1024 * 1024):
                total += len(chunk)
                if total > settings.max_upload_mb * 1024 * 1024:
                    raise HTTPException(413, f"Audio exceeds the {settings.max_upload_mb} MB upload limit")
                output.write(chunk)
    finally:
        await upload.close()
    if total == 0:
        raise HTTPException(400, "Uploaded audio is empty")
    return filename


def _prepare_and_embed(source: Path, filename: str, normalized: Path) -> tuple[np.ndarray, float, dict]:
    with encoder.inference_slot():
        duration, quality = normalize_audio(source, filename, normalized)
        vector = encoder.extract(normalized)
        return vector, duration, quality


async def embed_upload(upload: UploadFile) -> tuple[np.ndarray, float, dict, TemporaryAudio]:
    filename = upload.filename or "audio.wav"
    suffix = Path(filename).suffix.lower()
    temp = TemporaryAudio(suffix)
    try:
        filename = await save_upload(upload, temp.source)
        # FFmpeg and torch are blocking; keep them off the async event loop.
        from starlette.concurrency import run_in_threadpool

        vector, duration, quality = await run_in_threadpool(
            _prepare_and_embed, temp.source, filename, temp.normalized
        )
        return vector, duration, quality, temp
    except InferenceBusy as exc:
        temp.cleanup()
        raise HTTPException(429, str(exc), headers={"Retry-After": "10"}) from exc
    except ValueError as exc:
        temp.cleanup()
        raise HTTPException(422, str(exc)) from exc
    except HTTPException:
        temp.cleanup()
        raise
    except Exception as exc:
        temp.cleanup()
        logger.exception("Audio embedding failed")
        raise HTTPException(503, "Voice embedding service is temporarily unavailable") from exc


def embedding_for(profile: SpeakerProfile) -> np.ndarray:
    raw = decrypt_embedding(profile.encrypted_embedding)
    vector = np.frombuffer(raw, dtype=np.float32).copy()
    if vector.size != profile.embedding_dimension or not np.isfinite(vector).all():
        raise RuntimeError("Stored voice reference is corrupt or incompatible")
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise RuntimeError("Stored voice reference is invalid")
    return vector / norm


@app.get("/healthz", tags=["service"])
def healthz():
    """Liveness check; does not load the model."""
    return {"status": "ok", "service": "speechbrain-voice-api"}


@app.get("/readyz", tags=["service"])
def readyz():
    workload = encoder.workload()
    return {
        "status": "ready" if encoder.model_loaded else "cold",
        "model_loaded": encoder.model_loaded,
        "model": MODEL_VERSION,
        "device_configured": settings.device,
        "max_concurrent_inferences": settings.max_concurrent_inferences,
        "queue_capacity": settings.max_queued_inferences,
        **workload,
    }


@app.get("/v1/speakers", tags=["speaker profiles"])
def list_speakers(tenant_id: str = Depends(get_tenant)):
    with SessionLocal() as db:
        profiles = db.scalars(
            select(SpeakerProfile).where(SpeakerProfile.tenant_id == tenant_id).order_by(SpeakerProfile.created_at)
        ).all()
        return {"speakers": [
            {"speaker_id": p.speaker_id, "display_name": p.display_name, "sample_duration_seconds": p.sample_duration_seconds,
             "usable_speech_seconds_estimate": p.usable_speech_seconds, "model": p.model_id,
             "model_revision": p.model_revision, "created_at": p.created_at}
            for p in profiles
        ]}


@app.post("/v1/speakers/enroll", status_code=201, tags=["speaker profiles"])
async def enroll_speaker(
    audio: UploadFile = File(...),
    display_name: str = Form(..., min_length=1, max_length=160),
    consent_confirmed: bool = Form(...),
    speaker_id: str | None = Form(default=None, max_length=36),
    tenant_id: str = Depends(get_tenant),
):
    if not consent_confirmed:
        raise HTTPException(400, "Explicit voice-enrollment consent is required")
    display_name = display_name.strip()
    if not display_name:
        raise HTTPException(422, "Display name cannot be blank")
    try:
        resolved_id = str(uuid.UUID(speaker_id)) if speaker_id else str(uuid.uuid4())
    except ValueError as exc:
        raise HTTPException(422, "speaker_id must be a UUID") from exc

    vector, duration, quality, temp = await embed_upload(audio)
    try:
        with SessionLocal.begin() as db:
            profile = db.scalar(select(SpeakerProfile).where(
                SpeakerProfile.tenant_id == tenant_id, SpeakerProfile.speaker_id == resolved_id
            ))
            if profile is None:
                profile = SpeakerProfile(
                    tenant_id=tenant_id, speaker_id=resolved_id, display_name=display_name,
                    encrypted_embedding=encrypt_embedding(vector.astype(np.float32).tobytes()),
                    embedding_dimension=int(vector.size), sample_duration_seconds=duration,
                    usable_speech_seconds=quality["usable_speech_seconds_estimate"], model_id=settings.model_id,
                    model_revision=settings.model_revision, consent_confirmed=True,
                )
                db.add(profile)
                action = "created"
            else:
                profile.display_name = display_name
                profile.encrypted_embedding = encrypt_embedding(vector.astype(np.float32).tobytes())
                profile.embedding_dimension = int(vector.size)
                profile.sample_duration_seconds = duration
                profile.usable_speech_seconds = quality["usable_speech_seconds_estimate"]
                profile.model_id = settings.model_id
                profile.model_revision = settings.model_revision
                profile.consent_confirmed = True
                action = "replaced"
            db.flush()
    except IntegrityError as exc:
        raise HTTPException(409, "Speaker profile conflicts with an existing profile") from exc
    finally:
        temp.cleanup()
    return {"status": action, "speaker_id": resolved_id, "display_name": display_name,
            "sample_quality": quality, "embedding": {"stored_encrypted": True, "dimension": int(vector.size),
            "model": settings.model_id, "model_revision": settings.model_revision}}


@app.delete("/v1/speakers/{speaker_id}", tags=["speaker profiles"])
def delete_speaker(speaker_id: str, tenant_id: str = Depends(get_tenant)):
    with SessionLocal.begin() as db:
        profile = db.scalar(select(SpeakerProfile).where(
            SpeakerProfile.tenant_id == tenant_id, SpeakerProfile.speaker_id == speaker_id
        ))
        if profile is None:
            raise HTTPException(404, "Speaker profile not found")
        db.delete(profile)
    return {"status": "deleted", "speaker_id": speaker_id}


@app.post("/v1/identify", tags=["identification"])
async def identify_speaker(
    audio: UploadFile = File(...),
    speaker_id: str | None = Form(default=None, max_length=36),
    tenant_id: str = Depends(get_tenant),
):
    vector, duration, quality, temp = await embed_upload(audio)
    try:
        with SessionLocal() as db:
            query = select(SpeakerProfile).where(
                SpeakerProfile.tenant_id == tenant_id,
                SpeakerProfile.consent_confirmed.is_(True),
                SpeakerProfile.model_id == settings.model_id,
                SpeakerProfile.model_revision == settings.model_revision,
            )
            if speaker_id:
                query = query.where(SpeakerProfile.speaker_id == speaker_id)
            profiles = db.scalars(query).all()
            candidates = []
            for profile in profiles:
                reference = embedding_for(profile)
                if reference.shape != vector.shape:
                    continue
                score = float(np.clip(np.dot(vector, reference), -1.0, 1.0))
                candidates.append({"speaker_id": profile.speaker_id, "display_name": profile.display_name,
                                   "cosine_similarity": round(score, 6)})
        candidates.sort(key=lambda item: item["cosine_similarity"], reverse=True)
        top = candidates[:settings.top_candidates]
        if not candidates:
            match_status = "NOT_ENROLLED"
            matched = None
        elif candidates[0]["cosine_similarity"] < settings.match_threshold:
            match_status = "UNKNOWN"
            matched = None
        elif len(candidates) > 1 and candidates[0]["cosine_similarity"] - candidates[1]["cosine_similarity"] < settings.match_margin:
            match_status = "AMBIGUOUS"
            matched = None
        else:
            match_status = "MATCHED"
            matched = candidates[0]
        return {
            "match_status": match_status,
            "matched_speaker": matched,
            "best_candidate": candidates[0] if candidates else None,
            "candidates": top,
            "score_type": "raw cosine similarity; not a calibrated probability",
            "decision_policy": {"threshold": settings.match_threshold, "minimum_margin": settings.match_margin},
            "sample_quality": quality,
            "model": {"name": MODEL_NAME, "id": settings.model_id, "revision": settings.model_revision},
            "audio_duration_seconds": round(duration, 3),
        }
    finally:
        temp.cleanup()


@app.exception_handler(Exception)
async def unhandled_exception_handler(_, exc: Exception):
    logger.error("Unhandled API error: %s", type(exc).__name__, exc_info=(type(exc), exc, exc.__traceback__))
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})
