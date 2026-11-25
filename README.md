# SpeechBrain Voice Matching API

Small, standalone FastAPI service for consent-based speaker enrollment and cross-recording identification. Other applications call this service over HTTPS; the SpeechBrain model runs in this project, not in the client projects or Vercel.

## What this service does

- SpeechBrain ECAPA-TDNN embeddings (`speechbrain/spkrec-ecapa-voxceleb`).
- Resamples uploaded WAV/MP3/M4A/AAC/FLAC/OGG audio to actual 16 kHz mono PCM using FFmpeg.
- Enrolls one named, consented voice reference per profile; submitting an existing `speaker_id` replaces its active reference.
- Encrypts embeddings with Fernet before saving them to the database. Embedding vectors are never returned by API responses.
- Compares query audio only against consented profiles belonging to the API token's fixed tenant.
- Returns `MATCHED`, `UNKNOWN`, `AMBIGUOUS`, or `NOT_ENROLLED`, with raw cosine scores (never probabilities).
- Keeps the model loaded once per service process and allows a bounded number of simultaneous inference calls. Calls beyond the active limit wait in a bounded queue; overload returns HTTP 429.

This is **speaker matching**, not diarization or transcription. The calling project must provide a clip containing one diarized speaker. A mixed-speaker conversation should not be submitted as one voice sample.

## Concurrency behavior

`MAX_CONCURRENT_INFERENCES` configures simultaneous model calls per process (default `2`); `MAX_QUEUED_INFERENCES` bounds waiters (default `8`). Audio decoding and embedding work run off the async HTTP event loop. Keep one Uvicorn worker per container so the process does not load duplicate copies of the ~1 GB runtime/model environment. Scale with additional containers only after benchmarking RAM, CPU, and latency. Every replica has an independent in-memory model; the database must be shared and durable when scaling past one instance.

The default of two is a configurable starting point, not a performance guarantee. If running on 1–2 CPU cores, consider setting active inference to `1` or `2`; if GPU-backed or on a larger CPU instance, benchmark before raising it. The global model is inference-only; requests can execute concurrently up to the configured semaphore capacity. Additional requests queue only to the configured bound.

## Local run with Docker

1. Copy `.env.example` to `.env`.
2. Generate a Fernet key:

   ```bash
   python -c 'from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())'
   ```

3. Replace `EMBEDDING_ENCRYPTION_KEY` and the example client tokens in `API_KEYS_JSON`. Each token maps to a tenant ID; give each consuming application a distinct token and tenant. Never commit `.env` or put a server API token in browser JavaScript.
4. Start the container:

   ```bash
   docker compose up --build
   ```

5. Liveness: `http://localhost:7860/healthz`; model/capacity status: `http://localhost:7860/readyz`; OpenAPI: `http://localhost:7860/docs`.

First inference downloads the model from Hugging Face into the mounted `model-cache` volume. Allow outbound access during initial startup/use. The model is lazily loaded on the first valid inference call.

## API

All `/v1` endpoints require `Authorization: Bearer <project-private-token>`. `API_KEYS_JSON` is written as `{"tenant-id":"private-token"}`; the token selects a fixed tenant and callers cannot override tenant scope in a request.

### Enroll or replace reference

```bash
curl -X POST "$VOICE_API_URL/v1/speakers/enroll" \
  -H "Authorization: Bearer $VOICE_API_TOKEN" \
  -F 'display_name=Arun' \
  -F 'consent_confirmed=true' \
  -F 'audio=@arun-clean-speech.wav'
```

To replace an existing active reference, also pass `-F 'speaker_id=<returned UUID>'`. Aim for about 20 seconds of clean speech across one consenting speaker. The API accepts at least three estimated active speech seconds by default but flags shorter samples as lower-quality; this is not an accuracy guarantee.

### Identify a speaker

```bash
curl -X POST "$VOICE_API_URL/v1/identify" \
  -H "Authorization: Bearer $VOICE_API_TOKEN" \
  -F 'audio=@new-speaker-turn.wav'
```

To compare only to a specific enrolled profile, also send `-F 'speaker_id=<UUID>'`.

### Call it from a Vercel server-side route

After the API container is deployed to a host that gives it an HTTPS URL, open the **calling Vercel project's Settings → Environment Variables** and add:

- `SPEAKER_MATCH_API_URL`: the deployed API's base URL, without a trailing slash.
- `SPEAKER_MATCH_API_TOKEN`: the private token configured as the value for that project's tenant in the API service's `API_KEYS_JSON`.

Redeploy the Vercel project after adding the variables. Keep the token in a Vercel server-side route; never put it in browser JavaScript. The SpeechBrain/PyTorch model still runs in this container service, not in Vercel.

```ts
const form = new FormData();
form.append("audio", new Blob([speakerAudioBytes]), "speaker-turn.wav");

const result = await fetch(`${process.env.SPEAKER_MATCH_API_URL}/v1/identify`, {
  method: "POST",
  headers: { Authorization: `Bearer ${process.env.SPEAKER_MATCH_API_TOKEN}` },
  body: form,
  signal: AbortSignal.timeout(120_000),
});

if (!result.ok) throw new Error(`Voice matching returned HTTP ${result.status}`);
const match = await result.json();
```

### List and delete profiles

```bash
curl "$VOICE_API_URL/v1/speakers" -H "Authorization: Bearer $VOICE_API_TOKEN"
curl -X DELETE "$VOICE_API_URL/v1/speakers/<UUID>" -H "Authorization: Bearer $VOICE_API_TOKEN"
```

### Response meaning

- `MATCHED`: best cosine score cleared the configured threshold and its lead over the next candidate cleared the margin.
- `UNKNOWN`: no enrolled voice cleared the threshold.
- `AMBIGUOUS`: top candidates were too close; review manually.
- `NOT_ENROLLED`: this tenant has no compatible enrolled references.

Raw cosine values are not calibrated probabilities. Defaults `MATCH_THRESHOLD=0.72` and `MATCH_MARGIN=0.08` are initial policy values only and must be calibrated on independent, consented recordings for the target conditions.

## Production notes

- Deploy as a Docker/container service with persistent model cache and durable database storage. Do not deploy the model as a standard Vercel Function; the Python/PyTorch/model environment exceeds its normal bundle limit.
- Keep TLS in front of the API. Use a secrets manager for API tokens and the Fernet key. Losing or rotating the Fernet key without re-encrypting references makes stored embeddings unreadable.
- The sample SQLite volume is appropriate for one instance. For multiple replicas, use a shared managed PostgreSQL database (for example, `postgresql+psycopg://USER:PASSWORD@HOST:5432/DB`) and keep the Fernet key identical across all replicas. Each replica can keep its own model cache.
- This API stores encrypted voice embeddings and metadata, not source audio. Uploaded files are temporary and deleted after processing.
- Add request rate limits at the ingress/gateway and rotate per-project API credentials. Do not log tokens, uploaded audio, or embedding values.
- Since embeddings are biometric data, require explicit consent and follow the organization's retention, deletion, and data-processing rules.
- CPU/GPU support is chosen by deployment hardware and `DEVICE`; GPU requires a CUDA-compatible PyTorch image and a host that actually provides CUDA. The default Docker image uses CPU-only PyTorch.

## Configuration

See `.env.example`. The service intentionally fixes each API key to one tenant in `API_KEYS_JSON`; this prevents caller-supplied tenant IDs from enabling cross-tenant profile access.
