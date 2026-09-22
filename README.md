# Toy Agent 🧸⚡

The software brain and conversational orchestrator for an AI-powered smart toy for children.

---

## 1. Architectural Boundary & Repository Responsibilities

This system maintains a strict separation of concerns across two independent repositories:

```
┌────────────────────────────────────────────────────────────────────────┐
│                              slm-router                                │
│  - Independent service exposing a FastAPI endpoint                     │
│  - Hosts the real Qwen2.5-1.5B-Instruct Small Language Model           │
│  - Classifies natural language requests on-device (LOCAL/COMMAND/CLOUD)│
│  - Exposes POST /route with execution latency telemetry                │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ HTTP (POST /route)
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                              toy-agent                                 │
│  - THIS REPOSITORY                                                     │
│  - Conversational Agent Brain & Orchestrator                           │
│  - Consumes the real slm-router API via RouterClient                   │
│  - Dispatches to specialized handlers:                                 │
│      • LocalHandler   (answers on-device queries)                      │
│      • CommandHandler (safe device/action confirmations)               │
│      • MemoryHandler  (structured key-value storage)                   │
│      • CloudHandler   (client stub / bridge for complex queries)       │
│  - Voice interfaces (STTProvider, TTSProvider) for speech I/O          │
│  - Child-safe error handling and failure resilience                    │
│  - Foundation for future ESP32-S3 hardware & voice streaming           │
└────────────────────────────────────────────────────────────────────────┘
```

> **Strict Architectural Rule:**
> `toy-agent` communicates with `slm-router` solely across its HTTP API boundary. No models, classification heuristics, or internal modules from `slm-router` are copied, vendored, or imported into `toy-agent`.

---

## 2. Phase 2 Architecture: Text → SLM-Router → RoutingDecision

Phase 2 establishes the strict contract boundary between `toy-agent` and the external `slm-router` service:

```
                    ┌─────────────────────────┐
                    │     User Text Query     │
                    └────────────┬────────────┘
                                 │
                                 ▼
                     ┌───────────────────────┐
                     │   AgentOrchestrator   │
                     └───────────┬───────────┘
                                 │
                                 ▼
                       ┌───────────────────┐
                       │   RouterClient    │
                       └─────────┬─────────┘
                                 │  HTTP POST /route {"query": "..."}
                                 ▼
                     ┌───────────────────────┐
                     │    slm-router API     │  (External service on :8008)
                     └───────────┬───────────┘
                                 │  Qwen2.5-1.5B-Instruct classification
                                 ▼
                     ┌───────────────────────┐
                     │    RoutingDecision    │
                     └───────────────────────┘
```

### Authoritative SLM-Router Contract

The external `slm-router` returns strictly routing decisions (it does **not** generate final answers, retrieve memory, execute commands, or call cloud LLMs):

```json
{
  "processing": "LOCAL" | "CLOUD",
  "memory_required": true | false,
  "memory_request": {
    "keys": ["favorite_animal", "child_name"]
  } | null
}
```

- **`processing`**: Must be either `"LOCAL"` or `"CLOUD"`.
- **`memory_required`**: Strict boolean (`true` or `false`).
- **`memory_request`**: Non-null if and only if `memory_required == true`, containing one or more semantic key identifiers.

---

## 3. Real Integration & Live Router Verification

### Step 1: Start the External `slm-router` Service
In the `slm-router` repository:

```bash
uv run uvicorn --app-dir src slm_router.api:app --host 127.0.0.1 --port 8008
```

### Step 2: Diagnostic Health Check via `toy-agent`
In the `toy-agent` directory:

```bash
.venv/bin/python main.py --health
# ✅ SLM Router is healthy at http://localhost:8008: {'status': 'healthy', 'model': 'Qwen/Qwen2.5-1.5B-Instruct'}
```

### Step 3: Run Interactive CLI (Phase 2 Routing Mode)
```bash
.venv/bin/python main.py --verbose
```

### Verified Live Queries (External Qwen2.5-1.5B on Port 8008)

| Query | Processing | Memory Required | Memory Keys | HTTP Latency |
|---|---|---|---|---|
| `"Tell me a joke"` | `LOCAL` | `false` | `null` | 2.80s |
| `"What is my favorite animal?"` | `LOCAL` | `true` | `["favorite_animal"]` | 2.60s |
| `"What is my name and favorite animal?"` | `LOCAL` | `true` | `["child_name", "favorite_animal"]` | 2.77s |
| `"What is the weather today?"` | `CLOUD` | `false` | `null` | 2.38s |
| `"Turn on the lights"` | `LOCAL` | `false` | `null` | 2.30s |

---

## 4. Voice Input & Speech Processing Layer (`app.audio`)

The audio layer implements a decoupled architecture:
```
Microphone / Hardware
         ↓
    AudioInput          (app.audio.input, app.audio.microphone)
         ↓ bytes (16kHz mono WAV)
    STTProvider         (app.audio.stt, app.audio.whisper_stt)
         ↓ str
  Recognized Text
```

### Core Abstractions
- **`AudioInput` (`app.audio.input`)**: Abstract base class defining `async def read() -> bytes`. Decouples where audio originates (embedded I2S buffer, companion app stream, file, or physical microphone) from how it is processed.
  - `MicrophoneAudioInput` (`app.audio.microphone`): Real development microphone input capturing physical audio via `sounddevice` and packaging standard 16kHz mono 16-bit WAV bytes.
  - `BufferAudioInput`: In-memory byte buffer for tests and network streaming.
  - `DevelopmentAudioInput`: Deterministic payload wrapper for test suites.
- **`STTProvider` (`app.audio.stt`)**: Abstract base class defining `async def transcribe(audio_data: bytes, **kwargs) -> str`.
  - `WhisperSTTProvider` (`app.audio.whisper_stt`): Real local speech recognition using quantized `faster-whisper` (`tiny.en` on CPU/int8). Zero cloud API keys or external services required.
  - `DevelopmentSTTProvider`: Deterministic test double for reproducible offline testing.
- **Audio Error Handling (`app.audio.exceptions`)**: Structured exceptions (`AudioInputError`, `TranscriptionError`) handle unavailable microphones, empty captures, truncated WAVs, and missing dependencies cleanly.
- **`TTSProvider` (`app.audio.tts`)**: Abstract base class defining `async def synthesize(text: str) -> bytes`.
- **Orchestrator STT Method**: `AgentOrchestrator.transcribe_audio(audio_data: bytes) -> str` isolates the audio-to-text step before routing.

---

## 5. Real Voice Test (Microphone → Whisper STT)

### Step 1: Install Voice Dependencies
```bash
pip install -e ".[voice]"
# or: pip install sounddevice numpy faster-whisper
```

### Step 2: Run Real Microphone & STT Test
```bash
python main.py --voice
```

Optional arguments:
- `--voice-duration 5.0`: Set custom recording duration in seconds (default is `4.0s`).
- `--voice-loop`: Run continuously in an interactive voice loop.
- `-v`, `--verbose`: Enable detailed model loading and audio debug logging.

Expected output:
```
============================================================
🎤 TOY AGENT — REAL VOICE PIPELINE (PHASE 1)
Audio Input:  MicrophoneAudioInput (16kHz mono)
STT Engine:   WhisperSTTProvider (faster-whisper 'tiny.en')
Duration:     4.0s
============================================================

🎤 Speak now...
[capture audio]
[run real STT]

Recognized text:
"What is my favorite animal?"
```

### Step 3: Run Real Microphone → STT → SLM-Router Pipeline
```bash
python main.py --voice-router
```

Expected output:
```
============================================================
TOY AGENT — REAL VOICE → STT → ROUTER
============================================================

🎤 Speak now...

[capture audio]

Recognized text:
"What is my favorite animal?"

Sending to SLM Router...

Routing decision:
{
  "processing": "LOCAL",
  "memory_required": true,
  "memory_request": {
    "keys": [
      "favorite_animal"
    ]
  }
}

Pipeline:
✓ physical microphone
✓ real speech-to-text
✓ router classification

STOP HERE.

Do not retrieve the value of memory keys.
Do not generate a response.
Do not execute anything.
```

## 6. Configuration & Environment Variables

Configuration is loaded from environment variables or `.env`:

| Variable | Default | Description |
|---|---|---|
| `ROUTER_BASE_URL` | `http://localhost:8008` | URL of the running `slm-router` API |
| `ROUTER_ROUTE_ENDPOINT` | `/route` | Router classification endpoint path |
| `ROUTER_HEALTH_ENDPOINT` | `/health` | Router health check endpoint path |
| `ROUTER_TIMEOUT_SECONDS` | `15.0` | Timeout allowing full SLM token generation |
| `MEMORY_STORAGE_PATH` | `data/memory.json` | File path for local key-value persistence |
| `ENVIRONMENT` | `development` | Deployment environment |
| `LOG_LEVEL` | `INFO` | Console log verbosity (`DEBUG`, `INFO`) |
| `CLOUD_PROVIDER_STUB_MODE` | `true` | When true, returns safe stub for CLOUD route |

---

## 7. Measured Performance & Latency Telemetry

When connected to the real `Qwen2.5-1.5B-Instruct` model on Apple Silicon (MPS), real queries exhibit the following baseline latencies:

| Query | Classified Route | SLM Cls Latency | SLM Handler Latency | HTTP Latency | End-to-End Total |
|---|---|---|---|---|---|
| `"Hello"` | `LOCAL` | 1.59s | 0.47s | 2.10s | **2.10s** |
| `"Turn on the lights"` | `COMMAND` | 1.03s | 0.79s | 1.83s | **1.83s** |
| `"Tell me a joke"` | `LOCAL` | 1.04s | 0.78s | 1.83s | **1.83s** |
| `"What is the capital of France?"` | `LOCAL` | 1.10s | 0.42s | 1.54s | **1.54s** |

---

## 8. Testing

### Run Phase 1 Audio Tests
```bash
.venv/bin/pytest tests/test_audio.py -v
```

### Run Full Test Suite
```bash
.venv/bin/pytest tests/ -v
```

All 80 unit tests pass cleanly:
- **AudioInput tests**: verifies `BufferAudioInput` and `DevelopmentAudioInput`, ensuring empty/None inputs raise `AudioInputError`.
- **STTProvider tests**: verifies deterministic transcription, registered mappings, fallback behavior, corrupt payload rejection (`TranscriptionError`), and strictly checks for zero Mac-native imports across `app/audio`.
- **RouterClient tests**: health check success/failure, timeout handling, connection refused, malformed response, HTTP errors.
- **Orchestrator tests**: routing decision dispatch, failure recovery, telemetry capture.
- **Memory & Handler tests**: structured KV storage, local handler, command handler, cloud stub.

---

## 9. Troubleshooting

- **Router connection refused (`http://localhost:8008/route`)**:
  - Ensure the `slm-router` uvicorn server is running: `curl http://127.0.0.1:8008/health`.
  - Check that port 8008 is not blocked by a firewall.
- **Port 8000 conflict**:
  - Port 8000 is frequently reserved on development machines. `slm-router` and `toy-agent` default to port `8008` to prevent port collisions.
- **Timeout on long queries**:
  - On CPU/MPS devices, generating multi-paragraph responses can take ~10 seconds. Keep `ROUTER_TIMEOUT_SECONDS=15.0`.

---

## 10. Architectural Roadmap

```
[Phase 1]
Text Input ──► toy-agent (Mock Router) ──► Handlers ──► Text Output

[Phase 1.5]
Text Input ──► toy-agent ──► RouterClient ──► real slm-router API (Qwen 1.5B) ──► Handlers ──► Text Output

[Phase 2 - Current]
Audio In ──► STTProvider ──► AgentOrchestrator ──► RouterClient (POST /route) ──► Handlers ──► TTSProvider ──► Audio Out
(Portable abstract interfaces, zero macOS lock-in, health diagnostics, full telemetry)

[Phase 3 - Future Target]
Hardware Integration: ESP32-S3 I2S Mic (INMP441) ──► Agent ──► I2S Amp (MAX98357A) Speaker
Natural Female Voice TTS Benchmark & Selection
```
