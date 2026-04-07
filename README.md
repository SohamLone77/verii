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

<div align="center">

# 🔍 VerifAI

### _An OpenEnv-compatible reinforcement-learning environment for evaluating and improving AI-generated writing quality_

[![Live on HF Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20HF%20Space-Running-brightgreen)](https://huggingface.co/spaces/SohamLone77/verifAI)
[![OpenEnv](https://img.shields.io/badge/OpenEnv-Compliant-blue)](https://github.com/openenv-hub/openenv)
[![Docker](https://img.shields.io/badge/Docker-SDK-2496ED?logo=docker&logoColor=white)](https://huggingface.co/spaces/SohamLone77/verifAI)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://python.org)

</div>

---

## 🌍 Real-World Problem: Why VerifAI?

Every company that uses AI to generate text — emails, reports, policies, customer replies — faces the same painful question: **Is this output actually good?**

VerifAI trains RL agents to act as **AI Writing Editors**. Given a raw prompt and an AI-generated draft, the agent must evaluate, rewrite, and refine the text to meet a strict, multi-dimensional quality rubric covering **safety**, **factuality**, **brevity**, and **brand voice**.

This is not a toy. It maps precisely to:
- Enterprise **content moderation** and **brand compliance** pipelines
- **AI-generated email / customer support** review workflows
- Training **self-correcting writing agents** that improve iteratively

---

## 🏗️ Environment Design

### Observation Space

Each time `reset()` or `step()` is called, the agent receives a typed `Observation` Pydantic model:

```python
class Observation(BaseModel):
    session_id: str          # Unique episode identifier
    task: TaskName           # "classify" | "rewrite" | "iterative"
    step: int                # Current step number (0-indexed)
    prompt: str              # The original writing prompt
    current_output: str      # AI-generated text to evaluate/improve
    rubric: Rubric           # Constraints: safety, brevity, factuality, semantic, token_budget
    done: bool               # Whether the episode has ended
    score: Optional[float]   # Final composite score (emitted on final step only)
    image_url: Optional[str] # Optional image for multimodal scenarios
```

### Action Space

The agent submits a typed `Action` Pydantic model:

```python
class Action(BaseModel):
    action_type: ActionType     # "classify" | "rewrite" | "submit"
    content: str                # Agent's text output or JSON classification
    reasoning: Optional[str]    # Optional scratchpad (earns CoT bonus)
    reasoning_steps: Optional[list[str]]  # Structured reasoning steps
    modality: Literal["text", "image", "structured"]
    structured_data: Optional[dict]  # For classify tasks (JSON with score + justification)
```

### Reward Signal

The reward is **smoothly shaped across the entire trajectory** — not sparse or binary:

```
R(t) = base_score × 0.80
     + progress_bonus × 0.30   ← bonus if this step improved the score
     - step_penalty × 0.02     ← efficiency nudge per step
     + completion_bonus × 0.20 ← awarded on done if score ≥ 0.70
     - safety_penalty × 0.30   ← penalises very low scores (< 0.15)
     + cot_bonus                ← bonus for structured reasoning traces
```

All rewards clamped to **[0.0, 1.0]**. Success threshold: `score ≥ 0.70`.

---

## 🎯 Tasks

### Task 1 — `classify` (Easy, 1 step)

The agent is shown a real AI-generated text and must **rate its quality 0–10** with a justification. One shot, no retries.

**Grader checks:** Rubric compliance (safety, factuality) + semantic similarity to gold-standard classification. Returns structured JSON with `{ "score": int, "justification": str }`.

**Example scenario:** *"Explain what machine learning is in one sentence."* → Agent must classify the given response's quality.

---

### Task 2 — `rewrite` (Medium, 3 steps)

The agent receives a flawed AI-generated output and must **rewrite it** to satisfy all rubric dimensions. Up to 3 revision attempts. Use `submit` to end early if confident.

**Grader checks per step:** RubricGrader (safety keywords, brevity, factuality markers, brand voice) + SemanticGrader (cosine similarity via `all-MiniLM-L6-v2`). Feedback visible after each step through updated `Observation`.

**Example scenario:** *"Write a professional email subject line for a product update"* → Rewrite until it is safe, concise, and on-brand.

---

### Task 3 — `iterative` (Hard, 5 steps)

The hardest task. The agent must handle **highly non-compliant drafts** under a strict token budget and safety constraints, receiving **rubric dimension scores as feedback after each step** to steer successive revisions strategically.

**What makes it hard:**
- Longer, complex reference scenarios (150–250 token budgets)
- Strict safety and factuality requirements simultaneously
- Agent must reason about which rubric dimension to fix first
- Step penalty discourages burning all 5 steps trivially

**Example scenario:** *"Write a detailed incident report for a data breach affecting 10,000 users"* — must be safe, factual, under budget, and brand-consistent after up to 5 rounds.

---

## 📊 Grader Architecture

VerifAI uses a **100% deterministic, reproducible** composite grader — no LLM-as-a-judge, no random sampling.

```
CompositeGrader = RubricGrader (65%) + SemanticGrader (35%)
```

### RubricGrader (rule-based, zero LLM calls)

| Dimension | Weight | Method |
|---|---|---|
| Safety | 35% | Regex + YAML keyword blocklist (`rubrics/safety.yaml`) |
| Factuality | 25% | Heuristic: detects unverifiable claim markers |
| Brevity | 20% | Word-count vs `token_budget`; redundancy regex patterns |
| Brand Voice | 20% | YAML forbidden phrase list (`rubrics/brand_voice.yaml`) |

### SemanticGrader (local embeddings)

- Uses `sentence-transformers/all-MiniLM-L6-v2` (80MB, cached in Docker image at build time)
- Computes cosine similarity between agent output and the gold-standard reference
- **Fully deterministic** — no API calls, no stochasticity
- Module-level model cache: loaded once per process, never re-instantiated

---

## 📁 Scenario Bank

`data/scenarios.yaml` contains **50+ hand-crafted scenarios** across all three difficulty levels:

| Difficulty | Count | Examples |
|---|---|---|
| Easy | 17 | ML definitions, factual Q&A, single-sentence explanations |
| Medium | 17 | Product descriptions, customer support emails, jargon rewrites |
| Hard | 16 | Data breach reports, ethics essays, executive communications |
| Multimodal | 2 | Image-captioning, wireframe analysis (vision-capable agents) |

Each scenario ships with: `prompt`, `reference_output`, `rubric` (per-dimension flags + `token_budget`), and optional `image_url`.

---

## 📈 Baseline Results

**Model:** `Qwen/Qwen2.5-72B-Instruct` via `https://router.huggingface.co/v1`  
**Temperature:** 0.3 | **Max tokens:** 600

| Task | Difficulty | Final Reward | Rewards per Step | Success |
|---|---|---|---|---|
| `classify` | Easy | **0.90** | `0.90` | ✅ |
| `rewrite` | Medium | **0.90** | `0.70, 0.71, 0.90` | ✅ |
| `iterative` | Hard | **0.92** | `0.72, 0.72, 0.72, 0.72, 0.92` | ✅ |

Scores are **fully reproducible**. Graders use deterministic local embeddings and rule-based checks.

To reproduce:
```bash
export HF_TOKEN=your-hf-api-token
python inference.py
```

Expected stdout format:
```
[START] task=classify env=verifai model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=classify:{...} reward=0.90 done=true error=null
[END] success=true steps=1 rewards=0.90
```

---

## 🚀 Setup & Usage

### Option A — Live HF Space (Zero Setup)

The environment is live at:  
**[https://huggingface.co/spaces/SohamLone77/verifAI](https://huggingface.co/spaces/SohamLone77/verifAI)**

Hit the API directly:
```bash
# Start a new episode
curl -X POST https://sohamlone77-verifai.hf.space/reset \
  -H "Content-Type: application/json" \
  -d '{"task": "classify"}'

# Submit an action
curl -X POST https://sohamlone77-verifai.hf.space/step \
  -H "Content-Type: application/json" \
  -d '{"session_id": "<from reset>", "action": {"action_type": "classify", "content": "{\"score\": 8, \"justification\": \"Clear and concise.\"}"}}'
```

### Option B — Docker (Local)

```bash
git clone https://huggingface.co/spaces/SohamLone77/verifAI
cd verifAI

docker build -t verifai .
docker run -p 7860:7860 verifai
# → API available at http://localhost:7860
# → Swagger docs at http://localhost:7860/docs
```

### Option C — Python (Dev)

```bash
pip install -r requirements.txt

export HF_TOKEN=your-token                              # Required for inference
export API_BASE_URL=https://router.huggingface.co/v1   # Default
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct            # Default

uvicorn app.main:app --reload --port 7860
```

---

## 🔌 API Reference

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/reset` | Start a new episode. Returns `session_id` + initial `Observation`. |
| `POST` | `/step` | Submit an `Action`. Returns next `Observation`, `Reward`, `done`, `info`. |
| `GET` | `/state/{session_id}` | Full current state + observation for a session. |
| `GET` | `/status/{session_id}` | Lightweight session status (step, done, total_reward). |
| `GET` | `/tasks` | List all tasks with metadata from `openenv.yaml`. |
| `POST` | `/grade` | Directly score any text against any rubric. |
| `GET` | `/health` | `{"status": "ok"}` — Liveness probe. |
| `GET` | `/docs` | Interactive Swagger UI. |

---

## ✅ OpenEnv Pre-Submission Checklist

| Requirement | Status |
|---|---|
| HF Space live and responds to `reset()` | ✅ |
| `openenv.yaml` valid with 3+ tasks | ✅ |
| `openenv validate` passes | ✅ |
| Dockerfile builds and runs on port 7860 | ✅ |
| `inference.py` in root, uses OpenAI client | ✅ |
| `[START]` / `[STEP]` / `[END]` log format correct | ✅ |
| `HF_TOKEN` with no default, `API_BASE_URL` + `MODEL_NAME` with defaults | ✅ |
| All grader scores in [0.0, 1.0] | ✅ |
| Reward provides partial progress signal (not sparse) | ✅ |
| Baseline reproduces in < 20 min on 2 vCPU / 8 GB | ✅ |

---

## 📂 Project Structure

```
verifAI/
├── inference.py            ← Submission entry-point (START/STEP/END logs)
├── openenv.yaml            ← OpenEnv spec manifest
├── Dockerfile              ← HF Spaces Docker image
├── requirements.txt        ← Python dependencies (incl. openenv-core)
│
├── app/
│   ├── main.py             ← FastAPI app + route registration
│   ├── environment.py      ← PromptReviewEnv: reset() / step() / state()
│   ├── models.py           ← Pydantic: Observation, Action, Reward, State
│   └── session.py          ← In-memory session store
│
├── tasks/
│   ├── task_classify.py    ← Easy: 1-step quality classification
│   ├── task_rewrite.py     ← Medium: 3-step rubric rewrite
│   └── task_iterative.py   ← Hard: 5-step iterative revision
│
├── graders/
│   ├── composite_grader.py ← Weighted blend (65% rubric + 35% semantic)
│   ├── rubric_grader.py    ← Rule-based: safety / brevity / factuality / brand
│   └── semantic_grader.py  ← sentence-transformers cosine similarity
│
├── reward/
│   ├── reward_fn.py        ← Shaped reward: progress + CoT + safety + completion
│   ├── reward_config.py    ← Tunable reward hyperparameters
│   └── cot_scorer.py       ← Chain-of-thought reasoning quality scorer
│
├── data/
│   └── scenarios.yaml      ← 50+ hand-crafted prompt/rubric/reference scenarios
│
├── rubrics/
│   ├── safety.yaml         ← Blocked keywords + regex patterns
│   ├── factuality.yaml     ← Unverifiable claim markers
│   ├── brevity.yaml        ← Redundancy patterns + default token budget
│   └── brand_voice.yaml    ← Forbidden corporate phrases
│
├── routes/
│   ├── env_routes.py       ← /reset /step /state /status
│   ├── task_routes.py      ← /tasks
│   └── grader_routes.py    ← /grade
│
└── tests/
    ├── test_spec.py        ← openenv.yaml compliance tests
    ├── test_environment.py ← reset/step/state integration tests
    └── test_graders.py     ← Grader unit tests
```
