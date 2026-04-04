# COST_TRACKING
from __future__ import annotations

from typing import Optional

import numpy as np

from app.models import Rubric
from graders import GraderResult, register_grader
from reward.cost_tracker import CostTracker


import os
from openai import OpenAI

# Module-level cached model — loaded once per process, not per call.
_LOCAL_MODEL = None


def _get_local_model():
    """Return the cached sentence-transformers model, loading it on first call."""
    global _LOCAL_MODEL
    if _LOCAL_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _LOCAL_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _LOCAL_MODEL


def _get_client() -> Optional[OpenAI]:
    """Lazily instantiate the OpenAI client. Returns None if key not set."""
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        return None
    return OpenAI(api_key=key)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two embedding vectors."""
    va = np.array(a, dtype=float)
    vb = np.array(b, dtype=float)
    norm_a = np.linalg.norm(va)
    norm_b = np.linalg.norm(vb)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(va, vb) / (norm_a * norm_b))


@register_grader("semantic")
class SemanticGrader:
    """
    Embedding-based grader.

    Primary path: local sentence-transformers (all-MiniLM-L6-v2) — always
    deterministic and requires no API key. This is the default used during
    inference and evaluation.

    Optional path: OpenAI text-embedding-3-small — used only when
    OPENAI_API_KEY is set AND a cost_tracker is supplied (i.e., live sessions
    with explicit cost tracking). This path is skipped during baseline scoring.

    When no gold standard is available, scores against the original prompt
    as a loose relevance signal.
    """

    def grade(
        self,
        prompt: str,
        output: str,
        rubric: Optional[Rubric] = None,
        reference: Optional[str] = None,
        cost_tracker: Optional[CostTracker] = None,
    ) -> GraderResult:
        anchor = reference if reference else prompt
        usage_metadata: dict = {}

        client = _get_client()
        if client and cost_tracker is not None:
            # OpenAI Embeddings path — only when explicitly tracking costs
            resp = client.embeddings.create(
                input=[anchor, output],
                model="text-embedding-3-small",
            )
            anchor_emb = resp.data[0].embedding
            output_emb = resp.data[1].embedding

            usage_metadata = {
                "model": "text-embedding-3-small",
                "prompt_tokens": resp.usage.prompt_tokens,
                "completion_tokens": resp.usage.total_tokens - resp.usage.prompt_tokens,
            }
            cost_tracker.track(
                model=usage_metadata["model"],
                prompt_tokens=usage_metadata["prompt_tokens"],
                completion_tokens=usage_metadata["completion_tokens"],
            )
            usage_metadata["tracked"] = True
        else:
            # Default: local embeddings — deterministic, no API key required
            model = _get_local_model()
            emb = model.encode([anchor, output])
            anchor_emb = emb[0].tolist()
            output_emb = emb[1].tolist()

        sim = _cosine_similarity(anchor_emb, output_emb)

        # Map cosine sim [-1, 1] -> [0, 1] (in practice most will be [0, 1])
        score = max(0.0, min(1.0, (sim + 1.0) / 2.0))

        return GraderResult(
            score=round(score, 4),
            breakdown={"semantic_similarity": round(sim, 4)},
            notes=[f"Cosine similarity to {'reference' if reference else 'prompt'}: {sim:.4f}"],
            metadata={"usage": usage_metadata} if usage_metadata else {}
        )
