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
│  - Child-safe error handling and failure resilience                    │
│  - Foundation for future ESP32-S3 hardware & voice streaming           │
└────────────────────────────────────────────────────────────────────────┘
```

> **Strict Architectural Rule:**
> `toy-agent` communicates with `slm-router` solely across its HTTP API boundary. No models, classification heuristics, or internal modules from `slm-router` are copied, vendored, or imported into `toy-agent`.

---

## 2. Real End-to-End Integration Workflow (Phase 1.5)

```
                    ┌─────────────────────────┐
                    │     User Text Input     │
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
                     │    slm-router API     │  (FastAPI on :8008)
                     └───────────┬───────────┘
                                 │  Calls Router.route()
                                 ▼
                     ┌───────────────────────┐
                     │  Qwen2.5-1.5B-Instruct│  (On-device SLM)
                     └───────────┬───────────┘
                                 │  Returns real routing decision & text
                                 ▼
                     ┌───────────────────────┐
                     │    RoutingDecision    │  (Mapped in toy-agent)
                     └───────────┬───────────┘
                                 │
         ┌───────────────────────┼───────────────────────┐
         ▼                       ▼                       ▼
   [ Route: LOCAL ]        [ Route: COMMAND ]      [ Route: CLOUD ]
         │                       │                       │
         ▼                       ▼                       ▼
   LocalHandler            CommandHandler          CloudHandler
   (Direct SLM Answer)     (Dynamic SLM Action)    (Cloud Bridge / Stub)
         │                       │                       │
         └───────────────────────┴───────────────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │ Child-Safe Text Response│
                    └─────────────────────────┘
```

---

## 3. Real Integration Startup Order

To run the real, unmocked conversational toy pipeline:

### Step 1: Start the Real `slm-router` Service
In the `slm-router` project repository:

```bash
# In slm-router directory:
uv run uvicorn api:app --host 127.0.0.1 --port 8008
```

You will see:
```text
Loading SLM on device: mps...
SLM loaded on mps.
INFO: Uvicorn running on http://127.0.0.1:8008
```

### Step 2: Start `toy-agent`
In the `toy-agent` directory:

```bash
# In toy-agent directory:
.venv/bin/python main.py --verbose --base-url http://localhost:8008
```

### Step 3: Converse
Interact with the real model in real-time:
```text
You: Hello
Toy: Hello! How can I assist you today?

You: Turn on the lights.
Toy: Sure thing! The lights have been turned on for you. Enjoy your evening!

You: What is the capital of France?
Toy: The capital of France is Paris.
```

---

## 4. Configuration & Environment Variables

Create `.env` from `.env.example`:

| Variable | Default | Description |
|---|---|---|
| `ROUTER_BASE_URL` | `http://localhost:8008` | URL of the running `slm-router` API |
| `ROUTER_ROUTE_ENDPOINT` | `/route` | Router classification endpoint path |
| `ROUTER_TIMEOUT_SECONDS` | `15.0` | Timeout allowing full SLM token generation |
| `MEMORY_STORAGE_PATH` | `data/memory.json` | File path for local key-value persistence |
| `ENVIRONMENT` | `development` | Deployment environment |
| `LOG_LEVEL` | `INFO` | Console log verbosity (`DEBUG`, `INFO`) |
| `CLOUD_PROVIDER_STUB_MODE` | `true` | When true, returns safe stub for CLOUD route |

---

## 5. Measured Performance & Latency Telemetry

When connected to the real `Qwen2.5-1.5B-Instruct` model on Apple Silicon (MPS), real queries exhibit the following baseline latencies:

| Query | Classified Route | SLM Cls Latency | SLM Handler Latency | HTTP Latency | End-to-End Total |
|---|---|---|---|---|---|
| `"Hello"` | `LOCAL` | 1.168s | 0.515s | 1.702s | **1.734s** |
| `"Turn on the lights."` | `COMMAND` | 1.043s | 1.012s | 2.040s | **2.040s** |
| `"Tell me a joke."` | `LOCAL` | 1.058s | 0.723s | 1.784s | **1.784s** |
| `"What is the capital of France?"` | `LOCAL` | 1.164s | 0.498s | 1.666s | **1.666s** |
| `"Can you turn down the volume?"` | `COMMAND` | 1.146s | 1.662s | 2.812s | **2.812s** |
| `"Why is the sky blue?"` | `LOCAL` (Long answer) | 1.039s | 11.117s | 12.162s | **12.163s** |

---

## 6. Testing

### Run All Tests
```bash
.venv/bin/pytest tests/ -v
```

All 37 tests will run and pass:
- 36 offline unit tests covering request serialization, memory store, handlers, and error resilience.
- 1 live integration test (`test_live_slm_router_integration`) verifying direct HTTP communication with the active `slm-router`.

---

## 7. Troubleshooting

- **Router connection refused (`http://localhost:8008/route`)**:
  - Ensure the `slm-router` uvicorn server is running: `curl http://127.0.0.1:8008/health`.
  - Check that port 8008 is not blocked by a firewall.
- **Port 8000 conflict**:
  - Port 8000 is frequently reserved on development machines. `slm-router` and `toy-agent` default to port `8008` to prevent port collisions.
- **Timeout on long queries**:
  - On CPU/MPS devices, generating multi-paragraph responses can take ~10 seconds. Keep `ROUTER_TIMEOUT_SECONDS=15.0`.

---

## 8. Architectural Roadmap

```
[Current Phase 1.5]
Text Input ──► toy-agent ──► RouterClient ──► real slm-router API (Qwen 1.5B) ──► Handler ──► Text Output

[Upcoming Phase 2]
Microphone / Audio ──► STT ──► toy-agent ──► real slm-router API ──► Handler ──► TTS (Natural Female Voice) ──► Speaker

[Final ESP32-S3 Architecture]
INMP441 Mic ──► ESP32-S3 I2S ──► Companion Bridge ──► Agent Orchestrator ──► MAX98357A Amp ──► Speaker
```
