"""Blocks adapter for the local TypeSafe-compatible System One API.

The request part is the same JSON body accepted by ``POST /v1/systemone``.
The one response artifact is the same JSON document that endpoint returns.
"""

from __future__ import annotations

import json
import os
import sys
import threading
from pathlib import Path
from typing import Any, Optional

from blocks_network import StartTaskMessage, TaskContext

# ``agent`` is a sibling of the replica package in this repository. Adding the
# project root uses uncommitted local changes during development; deployed
# agents resolve the same package from the declared project dependency.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from truetype.api import SystemOneRequest  # noqa: E402
from truetype.engine import (  # noqa: E402
    DEFAULT_TEMPERATURE,
    PREFIX_CACHE_HARD_CAP,
    EngineConfig,
    GemmaLetterEngine,
)
from truetype.service import TypeSafeReplica  # noqa: E402


_service: TypeSafeReplica | None = None
_service_lock = threading.Lock()


def _get_service() -> TypeSafeReplica:
    """Create the same configured singleton engine used by the HTTP API."""
    global _service
    if _service is not None:
        return _service
    with _service_lock:
        if _service is None:
            engine = GemmaLetterEngine(EngineConfig(
                model_id=os.environ.get("TYPESAFE_REPLICA_MODEL_ID", "google/gemma-4-12B"),
                device=os.environ.get("TYPESAFE_REPLICA_DEVICE", "auto"),
                dtype=os.environ.get("TYPESAFE_REPLICA_DTYPE", "auto"),
                top_k=int(os.environ.get("TYPESAFE_REPLICA_TOP_K", "5")),
                prefix_cache=os.environ.get("TYPESAFE_REPLICA_PREFIX_CACHE", "1")
                not in {"0", "false", "False"},
                max_cached_prefixes=int(os.environ.get("TYPESAFE_REPLICA_MAX_CACHED_PREFIXES", "16")),
                prefix_cache_hard_cap=int(os.environ.get(
                    "TYPESAFE_REPLICA_PREFIX_CACHE_HARD_CAP", str(PREFIX_CACHE_HARD_CAP)
                )),
            ))
            engine.load()
            _service = TypeSafeReplica(engine)
    return _service


def _request_body(task: StartTaskMessage) -> dict[str, Any]:
    """Read one JSON request part, rejecting absent or malformed task input."""
    for part in task.request_parts or []:
        if getattr(part, "text", None) is not None:
            try:
                value = json.loads(str(part.text))
            except json.JSONDecodeError as exc:
                raise ValueError("Input must be a JSON object matching POST /v1/systemone") from exc
            if not isinstance(value, dict):
                raise ValueError("Input must be a JSON object matching POST /v1/systemone")
            return value
    raise ValueError("Missing required JSON request input")


def handler(task: StartTaskMessage, ctx: Optional[TaskContext] = None) -> dict[str, Any]:
    """Evaluate a System One request and return its HTTP-equivalent response."""
    request = SystemOneRequest.model_validate(_request_body(task))
    if not request.questions:
        raise ValueError("questions must not be empty")

    if ctx is not None:
        ctx.report_status("Loading model and evaluating questions…")

    result = _get_service().system_one(
        state=request.state,
        questions={qid: spec.model_dump() for qid, spec in request.questions.items()},
        model=request.model,
        temperature=request.temperature,
    )
    response = {"model": result.model, "answers": result.answers, "usage": result.usage}
    return {
        "artifacts": [{
            "data": json.dumps(response),
            "mimeType": "application/json",
            "outputId": "response",
            "fileName": "systemone-response.json",
        }],
    }
