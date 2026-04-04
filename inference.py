"""
Inference Script Example
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.
    LOCAL_IMAGE_NAME The name of the local image to use for the environment if you are using from_docker_image()
                     method
- Defaults are set only for API_BASE_URL and MODEL_NAME
    (and should reflect your active inference setup):
    API_BASE_URL = os.getenv("API_BASE_URL", "<your-active-endpoint>")
    MODEL_NAME = os.getenv("MODEL_NAME", "<your-active-model>")

- The inference script must be named `inference.py` and placed in the root directory of the project
- Participants must use OpenAI Client for all LLM calls using above variables
STDOUT FORMAT
- The script must emit exactly three line types to stdout, in this order:
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> rewards=<r1,r2,...,rn>
  Rules:
    - One [START] line at episode begin.
    - One [STEP] line per step, immediately after env.step() returns.
    - One [END] line after env.close(), always emitted (even on exception).
    - reward and rewards are formatted to 2 decimal places.
    - done and success are lowercase booleans: true or false.
    - error is the raw last_action_error string, or null if none.
    - All fields on a single line with no newlines within a line.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Optional

from openai import OpenAI

from app.environment import PromptReviewEnv
from app.models import Action, ActionType, TaskName
from tasks import load_task

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

BENCHMARK = os.getenv("VERIFAI_BENCHMARK", "verifai")
TASK_OVERRIDE = os.getenv("VERIFAI_TASK")


def _build_client() -> OpenAI:
    if not HF_TOKEN:
        print("HF_TOKEN is required but not set.", file=sys.stderr)
        raise SystemExit(1)
    return OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)


def _format_action_for_log(action: Action, raw_text: str) -> str:
    if action.modality == "structured" and action.structured_data is not None:
        payload = json.dumps(action.structured_data, separators=(",", ":"))
        action_text = payload
    else:
        action_text = raw_text
    action_text = " ".join(action_text.split())
    return f"{action.action_type.value}:{action_text}".strip()


def _build_messages(obs, task) -> list[dict[str, Any]]:
    system_prompt = getattr(task, "system_prompt", "")
    task_name = getattr(task, "name", "unknown")

    user_prompt = (
        f"Task: {task_name}\n"
        f"Step: {obs.step} of {task.max_steps}\n"
        f"Prompt: {obs.prompt}\n"
        f"Current Output: {obs.current_output}\n"
        f"Rubric: {obs.rubric.model_dump()}\n"
    )

    if task_name == "classify":
        user_prompt += (
            "Return only a JSON object with keys 'score' (int 0-10) and "
            "'justification' (string)."
        )
    else:
        user_prompt += "Return only the revised text."

    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    content: list[dict[str, Any]] = [{"type": "text", "text": user_prompt}]
    if obs.image_url:
        content.append({"type": "image_url", "image_url": {"url": obs.image_url}})
    elif obs.image_b64:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{obs.image_b64}"}})

    messages.append({"role": "user", "content": content})
    return messages


def _decide_action_type(task_name: str, step: int, max_steps: int) -> ActionType:
    if task_name == "classify":
        return ActionType.classify
    if step >= max_steps - 1:
        return ActionType.submit
    return ActionType.rewrite


def _run_task(client: OpenAI, task_name: str) -> None:  # noqa: C901
    env = PromptReviewEnv()
    task = load_task(TaskName(task_name))

    # Initialise before try so finally never references unbound names
    state = None
    step_response = None
    success_text = "false"
    rewards: list[float] = []
    last_error: Optional[str] = None

    print(f"[START] task={task_name} env={BENCHMARK} model={MODEL_NAME}")

    obs, state = env.reset(task_name=TaskName(task_name))

    try:
        while not state.done:
            action_type = _decide_action_type(task_name, state.step, state.max_steps)
            messages = _build_messages(obs, task)
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.3,
                max_tokens=600,
            )
            assistant_text = response.choices[0].message.content.strip()

            structured_payload = None
            if task_name == "classify":
                try:
                    structured_payload = json.loads(assistant_text)
                except json.JSONDecodeError:
                    structured_payload = None

            if structured_payload is not None:
                action = Action(
                    action_type=action_type,
                    content="",
                    modality="structured",
                    structured_data=structured_payload,
                )
            else:
                action = Action(
                    action_type=action_type,
                    content=assistant_text,
                    modality="text",
                )

            step_response = env.step(state=state, obs=obs, action=action)
            obs = step_response.observation

            rewards.append(step_response.reward.value)
            reward_text = f"{step_response.reward.value:.2f}"
            done_text = str(step_response.done).lower()
            action_log = _format_action_for_log(action, assistant_text)
            error_text = last_error or "null"
            print(
                f"[STEP] step={state.step} action={action_log} reward={reward_text} "
                f"done={done_text} error={error_text}"
            )

            if step_response.done:
                break

        final_score = step_response.info.get("score") if step_response is not None else None
        episode_info = env.get_episode_info(state, final_score)
        success_text = str(episode_info.success).lower()
    except Exception as exc:
        last_error = " ".join(str(exc).split())
        success_text = "false"
    finally:
        rewards_text = ",".join(f"{value:.2f}" for value in rewards)
        total_steps = state.step if state is not None else 0
        print(
            f"[END] success={success_text} steps={total_steps} rewards={rewards_text}"
        )


def main() -> None:
    client = _build_client()
    tasks = [TASK_OVERRIDE] if TASK_OVERRIDE else ["classify", "rewrite", "iterative"]
    for task_name in tasks:
        _run_task(client, task_name)


if __name__ == "__main__":
    main()
