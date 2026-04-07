---
title: VerifAI
emoji: 🔍
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
tags:
  - openenv
  - rl-environment
  - ai-evaluation
  - writing-quality
  - reinforcement-learning
short_description: OpenEnv-compatible RL environment for AI writing evaluation
---

# VerifAI

An **OpenEnv-compatible reinforcement-learning environment** for evaluating and improving AI-generated writing quality.

---

## Overview

VerifAI exposes a REST API that allows RL agents to interact with a writing evaluation environment. Agents receive writing prompts and are rewarded for producing high-quality outputs according to multi-dimensional rubrics.

---

## Environment Description

| Property | Value |
|---|---|
| Observation Space | Prompt + current output + rubric + step number |
| Action Space | `classify`, `rewrite`, or `submit` with text content |
| Reward Range | [0.0, 1.0] |
| Episode Done | Max steps reached or agent submits |

---

## Tasks

| Task | Difficulty | Max Steps | Description |
|---|---|---|---|
| `classify` | Easy | 1 | Classify output quality 0–10 |
| `rewrite` | Medium | 3 | Rewrite to satisfy rubric |
| `iterative` | Hard | 5 | Multi-turn revision under constraints |

---

## Scoring

Scores are computed by a composite grader:

| Dimension | Weight | Description |
|---|---|---|
| Safety | 0.30 | No harmful content |
| Brevity | 0.20 | Within token budget |
| Factuality | 0.25 | Claims are verifiable |
| Semantic Quality | 0.25 | Similarity to gold standard |

---

## Baseline Scores

Baseline agent: `Qwen/Qwen2.5-72B-Instruct` via `https://router.huggingface.co/v1`

| Task | Difficulty | Final Reward | Success Rate |
|---|---|---|---|
| `classify` | Easy | 0.90 | 100% |
| `rewrite` | Medium | 0.90 | 100% |
| `iterative` | Hard | 0.92 | 100% |

Scores are **reproducible**: graders use deterministic sentence-transformers embeddings
(`all-MiniLM-L6-v2`) and rule-based rubric checks (no LLM calls in graders).
Run the baseline yourself:

```bash
export HF_TOKEN=your-hf-token
python inference.py
```

---

## Setup

```bash
# 1. Clone and install
git clone https://huggingface.co/spaces/SohamLone77/verifAI
cd verifai
pip install -r requirements.txt

# 2. Set required environment variables
export HF_TOKEN=your-hf-token           # required — HF / OpenAI-compatible API key
export API_BASE_URL=https://router.huggingface.co/v1   # default
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct            # default

# Optional: protect analytics endpoints
export VERIFAI_ANALYTICS_API_KEY=your-analytics-key

# 3. Run locally
uvicorn app.main:app --reload --port 7860

# 4. Validate spec
bash scripts/validate.sh

# 5. Run tests
pytest tests/ -v
```

## Inference (Submission)

The required `inference.py` script is located in the project root and emits
the mandated `[START]`, `[STEP]`, `[END]` logs. It uses the OpenAI client
configured by these environment variables:

```
API_BASE_URL=https://router.huggingface.co/v1   # default
MODEL_NAME=Qwen/Qwen2.5-72B-Instruct            # default
HF_TOKEN=your-hf-token                           # required, no default
# Optional (when using from_docker_image)
LOCAL_IMAGE_NAME=verifai:latest
```

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/reset` | Start a new episode |
| POST | `/step` | Submit an action |
| GET | `/state/{session_id}` | Full state + observation |
| GET | `/status/{session_id}` | Current session state |
| GET | `/tasks` | List all tasks |
| POST | `/grade` | Score an episode |
| POST | `/baseline/run` | Run OpenAI baseline |
| GET | `/health` | Liveness probe |

## Deployment

Hosted on [Hugging Face Spaces](https://huggingface.co/spaces/SohamLone77/verifAI) using Docker SDK on port 7860.

```bash
bash scripts/deploy_hf.sh
```

---

## Problem Statement (Round 1)

### Task Summary

Build a complete, real-world OpenEnv environment that an AI agent can learn from
through the standard `step()` / `reset()` / `state()` API.

### Key Requirements

- Real-world task simulation (not games or toys)
- Full OpenEnv spec: typed models, `step()` / `reset()` / `state()`, `openenv.yaml`
- Minimum 3 tasks with graders (easy -> medium -> hard), scores in 0.0 to 1.0
- Meaningful reward function with partial progress signals
- Baseline inference script with reproducible scores
- Deploy to Hugging Face Spaces + working Dockerfile
- README with environment description, action/observation spaces, setup instructions

### Functional Requirements

- Real-world task domain (e.g., email triage, code review, content moderation)
- OpenEnv spec compliance and validation via `openenv validate`
- Deterministic, well-defined graders with success/failure criteria
- Reward shaping over the trajectory (not only terminal rewards)
- Baseline inference uses OpenAI client and reads API creds from env vars

### Non-Functional Requirements

- HF Space deploys and responds (tagged with `openenv`)
- Containerized execution with Dockerfile
- Documentation includes environment description, task details, and baseline scores

### Mandatory Inference Requirements

- Inference script must be `inference.py` at repo root
- Use OpenAI client configured with:
  - `API_BASE_URL`
  - `MODEL_NAME`
  - `HF_TOKEN`
  - `LOCAL_IMAGE_NAME` (optional if using from_docker_image)
- Stdout logging format must be exact:

```
[START] task=<task_name> env=<benchmark> model=<model_name>
[STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
[END]   success=<true|false> steps=<n> rewards=<r1,r2,...,rn>
```

### Pre-Submission Checklist

- HF Space deploys and responds to `reset()`
- `openenv validate` passes
- Dockerfile builds
- Baseline inference script runs and reproduces scores
- 3+ tasks with graders produce 0.0 to 1.0 rewards
