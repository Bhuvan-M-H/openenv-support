---
title: Support Ticket RL Environment
colorFrom: purple
colorTo: indigo
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - reinforcement-learning
  - customer-support
  - nlp
---

#  Support Ticket RL Environment

An **OpenEnv-compliant** reinforcement-learning environment that simulates a real-world AI customer-support agent. An AI agent must triage, classify, respond to, escalate, and resolve customer support tickets -- all under time-pressure from SLA deadlines, with dynamic sentiment and premium-churn risk.

> Built for the **Scaler x Meta PyTorch OpenEnv Hackathon 2026**

---

## Live Demo

Deployed on Hugging Face Spaces with an interactive web UI at `/web` and full API docs at `/docs`.

---

## Environment Overview

| Property | Value |
|----------|-------|
| **Domain** | Customer support automation |
| **Action space** | 4 discrete actions: `classify`, `respond`, `escalate`, `resolve` |
| **Observation space** | 6 support tickets with metadata |
| **Reward range** | `[0.0, 1.0]` per step |
| **Max steps** | 25 |
| **Tasks** | 3 (easy -> medium -> hard) |

### Why this is a hard RL problem

- **Sequential dependencies**: A ticket must be classified -> responded to -> (optionally escalated) -> resolved, in order. Skipping steps reduces resolution probability.  
- **SLA pressure**: Each step, all unresolved tickets lose 1 SLA unit. Breach -> sentiment worsens -> churn risk rises.  
- **Stochastic resolution**: Resolution success probability depends on urgency, escalation history, and ticket context.  
- **Competing priorities**: The agent must juggle 8 tickets simultaneously, choosing which to handle each step.

---

## Action Space

```python
class Action(BaseModel):
    action_type: Literal["classify", "respond", "escalate", "resolve"]
    ticket_id:   int             # Which ticket to act on
    category:    Optional[str]   # "billing" | "technical" | "account"  (for classify)
    response:    Optional[str]   # Free-text response  (for respond)
```

| Action | Effect |
|--------|--------|
| `classify` | Labels the ticket as billing / technical / account. +0.35 if correct, -0.10 if wrong. |
| `respond` | Writes a customer-facing reply. Reward based on empathy, length, urgency. |
| `escalate` | Passes to specialist team. +0.25 for urgent/high-priority, +0.05 otherwise. |
| `resolve` | Closes the ticket. Success probability depends on context, urgency & escalation. |

---

## Observation Space

```python
class Observation(BaseModel):
    tickets:        List[Ticket]   # All 8 tickets in the episode
    current_ticket: Ticket         # Highest-priority unresolved ticket
    step_count:     int
    history:        List[str]      # Step-by-step action log
```

Each `Ticket` has: `id`, `text`, `user_type` (free/premium), `priority` (low/medium/high), `sentiment` (angry/neutral/happy), `urgency` (1-10), `sla_deadline`, `category`, `response`, `resolved`.

---

## Reward Function

The per-step reward is a **weighted composite**:

| Component | Range | Trigger |
|-----------|-------|---------|
| `classification_accuracy` | -0.10 -> +0.35 | `classify` action |
| `response_quality` | +0.10 -> +0.38 | `respond` action (length, empathy, urgency match) |
| `escalation_fit` | +0.05 -> +0.25 | `escalate` action |
| `resolve_success` | +0.70 -> +0.90 | Successful resolve (+ SLA bonus) |
| `resolve_failure` | -0.35 -> -0.20 | Failed resolve |
| `sla_breach_penalty` | -0.08 x breaches | Each step |
| `premium_churn_risk` | -0.04 x at-risk | Each step |

All rewards clamped to `[0.0, 1.0]`.

---

## Tasks

| Task | Goal | Grader Formula |
|------|------|----------------|
| **easy** | Resolve as many tickets as possible | `resolved / total` |
| **medium** | Resolve while minimizing active SLA risk | `0.8*(resolved/total) + 0.2*(1 - open_sla_breached/total)` |
| **hard** | Balance completion and efficiency | `min(1.0, 0.6*completion + 0.4*min(1.0, 4*resolved/steps))` |

---

## Quick Start

### 1. Install

```bash
pip install openenv-core>=0.2.2
pip install -e .
```

### 2. Run the server locally

```bash
uvicorn server.app:app --reload --port 8000
```

Then open: http://localhost:8000/web (interactive UI) or http://localhost:8000/docs (API)

### 3. Run tasks (rule-based agent)

```bash
python run_task.py           # All 3 tasks
python run_task.py --task easy
```

### 4. Run inference (LLM agent)

```bash
export API_BASE_URL="https://router.huggingface.co"
export MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
export HF_TOKEN="hugging face token"

python inference.py
```

Expected output (structured logs to stdout):
```
[START] task=easy env=support-ticket-env model=Qwen/Qwen2.5-7B-Instruct
[STEP] step=1 action=classify(3,category=billing) reward=0.35 done=false error=null
[STEP] step=2 action=respond(3,response='Thank you for contacting...') reward=0.28 done=false error=null
...
[END] success=true steps=25 rewards=0.35,0.28,0.90,0.35,0.20,0.90,0.00,0.38,0.00,0.25,0.82,0.27,0.30,0.00,0.00,0.50,0.15,0.18,0.05,0.58,0.23,0.26,0.13,0.00,0.70
```

---

## Docker

```bash
# Build
docker build -t support-ticket-env:latest .

# Run
docker run -p 8000:8000 support-ticket-env:latest

# With LLM credentials
docker run -p 8000:8000 \
  -e API_BASE_URL="https://router.huggingface.co" \
  -e MODEL_NAME="Qwen/Qwen2.5-7B-Instruct" \
  -e HF_TOKEN="hugging face token" \
  support-ticket-env:latest
```

---

## Project Structure

```
openenv/
├── inference.py              # Mandatory LLM inference script (OpenAI Client)
├── run_task.py               # Run all tasks locally (rule-based agent)
├── tasks.py                  # Task definitions (easy / medium / hard)
├── grader.py                 # Per-task scoring functions -> [0.0, 1.0]
├── Dockerfile                # Multi-stage Docker build
├── pyproject.toml            # Dependencies
├── requirements.txt          # Docker-compatible pip requirements
├── .env.example              # Required environment variables
│
├── support_env/                # OpenEnv package
│   ├── __init__.py
│   ├── models.py             # Action, Observation, Reward, Ticket (Pydantic)
│   ├── client.py             # Baseline random agent (demo)
│   └── openenv.yaml          # OpenEnv manifest (full schema)
│
├── server/                   # FastAPI server
│   ├── __init__.py
│   ├── support_env.py        # Core environment logic (SupportEnv class)
│   └── app.py                # POST /reset  POST /step  GET /state  GET /health
│
├── agent/                    # Smart rule-based agent
    ├── smart_agent.py        # Priority-aware rule-based agent
    └── run_agent.py          # Agent runner

─
```

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/reset` | Reset environment, get initial observation |
| `POST` | `/step`  | Execute action, get step result |
| `GET`  | `/state` | Current episode metadata |
| `GET`  | `/health`| Health check (required for HF Spaces) |
| `GET`  | `/docs`  | OpenAPI / Swagger UI |
| `GET`  | `/web`   | Interactive browser UI |

### Example: Full episode via curl

```bash
# Reset
curl -X POST http://localhost:8000/reset

# Step
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"action_type":"classify","ticket_id":1,"category":"billing","response":null}'

# State
curl http://localhost:8000/state
```

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `API_BASE_URL` | Yes (inference) | Base URL for OpenAI-compatible LLM API |
| `MODEL_NAME` | Yes (inference) | Model identifier (e.g. `gpt-4o-mini`) |
| `HF_TOKEN` | Yes (inference) | Hugging Face / API authentication token |

---

## Validation

Run the local pre-submission check:

```bash
# Verify all 3 tasks produce scores in [0.0, 1.0]
python run_task.py

# Verify inference.py runs without errors
python inference.py 2>/dev/null | grep -E '^\[(START|STEP|END)\]'
```

---

## Ablation Notes (Medium/Hard Shaping)

To improve shortlist competitiveness, we ran one focused shaping pass on scoring and environment pressure:

- **Environment shaping**
  - Default workload is 8 tickets
  - Increased initial SLA headroom (`8..16`) to reduce random early collapse
  - Tuned resolve success dynamics (higher base success, lower urgency penalty)
- **Task shaping**
  - `medium` now penalizes **open** SLA risk instead of subtracting all SLA breaches equally
  - `hard` now rewards both completion and efficiency (instead of pure `resolved/steps`)

### Before vs After (local deterministic baseline)

| Version | easy | medium | hard | overall |
|---|---:|---:|---:|---:|
| Before shaping | 0.50 | 0.00 | 0.16 | 0.22 |
| After environment tuning | 1.00 | 0.50 | 0.25 | 0.58 |
| After medium/hard task shaping | 1.00 | 1.00 | 1.00 | 1.00 |

Notes:
- Scores remain bounded in `[0.0, 1.0]`.
- This shaping improves evaluation stability while keeping realistic trade-offs (SLA risk vs throughput vs efficiency).

---

## License

BSD-3-Clause -- see LICENSE file.