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

| Task | Difficulty | Avg Reward | Success Rate |
|---|---|---|---|
| `classify` | Easy | 0.72 | 86% |
| `rewrite` | Medium | 0.61 | 64% |
| `iterative` | Hard | 0.48 | 41% |

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
git clone https://huggingface.co/spaces/SohamLone77/verifai
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

Hosted on [Hugging Face Spaces](https://huggingface.co/spaces/SohamLone77/verifai) using Docker SDK on port 7860.

```bash
bash scripts/deploy_hf.sh
```
