from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import DateTime, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class SpeakerProfile(Base):
    __tablename__ = "speaker_profiles"
    __table_args__ = (UniqueConstraint("tenant_id", "speaker_id", name="uq_tenant_speaker"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(128), index=True, nullable=False)
    speaker_id: Mapped[str] = mapped_column(String(36), nullable=False)
    display_name: Mapped[str] = mapped_column(String(160), nullable=False)
    encrypted_embedding: Mapped[str] = mapped_column(Text, nullable=False)
    embedding_dimension: Mapped[int] = mapped_column(Integer, nullable=False)
    sample_duration_seconds: Mapped[float] = mapped_column(nullable=False)
    usable_speech_seconds: Mapped[float] = mapped_column(nullable=False)
    model_id: Mapped[str] = mapped_column(String(160), nullable=False)
    model_revision: Mapped[str] = mapped_column(String(80), nullable=False)
    consent_confirmed: Mapped[bool] = mapped_column(nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utcnow, onupdate=utcnow)
