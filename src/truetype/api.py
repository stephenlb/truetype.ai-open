"""TypeSafe-compatible HTTP API backed by Gemma 4 letter logits.

Drop-in for ``POST https://api.typesafe.ai/v1/systemone`` for the question
primitives this replica supports. The response shape matches TypeSafe's docs:
``model``, ``answers`` (keyed by question id), and ``usage``.
"""

from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from .engine import (
    DEFAULT_TEMPERATURE,
    PREFIX_CACHE_HARD_CAP,
    EngineConfig,
    GemmaLetterEngine,
)
from .service import TypeSafeReplica

logger = logging.getLogger(__name__)

_api_key = os.environ.get("TYPESAFE_REPLICA_API_KEY")


class QuestionSpec(BaseModel):
    type: str
    instructions: str | dict | list
    criteria: dict | list | None = None


class SystemOneRequest(BaseModel):
    state: str | dict | list
    model: str = "gemma-4-12b"
    questions: dict[str, QuestionSpec]
    temperature: float = Field(default=DEFAULT_TEMPERATURE, gt=0)


class SystemOneResponse(BaseModel):
    model: str
    answers: dict
    usage: dict


_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    engine = GemmaLetterEngine(EngineConfig(
        model_id=os.environ.get("TYPESAFE_REPLICA_MODEL_ID", "google/gemma-4-12B"),
        top_k=int(os.environ.get("TYPESAFE_REPLICA_TOP_K", "5")),
        # Prefix KV caching makes repeat questions ~5x faster. The first request for
        # a given question shape pays a normal (uncached) forward pass to populate it.
        prefix_cache=os.environ.get("TYPESAFE_REPLICA_PREFIX_CACHE", "1") not in {"0", "false", "False"},
        max_cached_prefixes=int(os.environ.get("TYPESAFE_REPLICA_MAX_CACHED_PREFIXES", "16")),
        prefix_cache_hard_cap=int(
            os.environ.get("TYPESAFE_REPLICA_PREFIX_CACHE_HARD_CAP", str(PREFIX_CACHE_HARD_CAP))
        ),
    ))
    engine.load()
    service = TypeSafeReplica(engine)

    # Optional startup warming for known hot questions, as a JSON object of question
    # specs: {"refund": {"type": "noul", "instructions": "..."}, ...}. Off by default:
    # warming only pays off for prefixes that will be reused many times.
    warm_spec = os.environ.get("TYPESAFE_REPLICA_WARM_QUESTIONS")
    if warm_spec:
        try:
            questions = json.loads(warm_spec)
            ms = service.warm(questions)
            logger.info("warmed %d question prefix(es) in %.0fms", len(questions), ms)
        except Exception:
            logger.exception("failed to warm TYPESAFE_REPLICA_WARM_QUESTIONS; continuing unwarmed")

    _state["service"] = service
    yield
    _state.clear()


app = FastAPI(title="truetype.ai replica", version="0.1.0", lifespan=lifespan)


def _authorize(authorization: str | None) -> None:
    if _api_key is None:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    if authorization.removeprefix("Bearer ").strip() != _api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")


@app.get("/health")
def health() -> dict:
    service = _state.get("service")
    if service is None:
        return {"status": "starting", "model_loaded": False}
    engine = service.engine
    return {
        "status": "ok",
        "model_loaded": engine.is_loaded,
        "prefix_cache": {
            "enabled": engine.config.prefix_cache,
            "hits": engine.prefix_cache_hits,
            "misses": engine.prefix_cache_misses,
            "evictions": engine.prefix_cache_evictions,
            "entries": len(engine._prefix_caches),
            "capacity": engine.config.max_cached_prefixes,
            "hard_cap": engine.config.prefix_cache_hard_cap,
        },
    }


@app.post("/v1/systemone", response_model=SystemOneResponse)
def system_one(
    request: SystemOneRequest,
    authorization: str | None = Header(default=None),
) -> SystemOneResponse:
    _authorize(authorization)
    service: TypeSafeReplica | None = _state.get("service")
    if service is None:
        raise HTTPException(status_code=503, detail="Model is still loading")

    if isinstance(request.questions, dict) and not request.questions:
        raise HTTPException(status_code=422, detail="questions must not be empty")

    try:
        result = service.system_one(
            state=request.state,
            questions={qid: spec.model_dump() for qid, spec in request.questions.items()},
            model=request.model,
            temperature=request.temperature,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return SystemOneResponse(model=result.model, answers=result.answers, usage=result.usage)


def main() -> None:
    import uvicorn

    uvicorn.run(
        "truetype.api:app",
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "8000")),
    )


if __name__ == "__main__":
    main()