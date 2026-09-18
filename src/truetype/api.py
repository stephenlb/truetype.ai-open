"""TypeSafe-compatible HTTP API backed by Gemma 4 letter logits.

Drop-in for ``POST https://api.typesafe.ai/v1/systemone`` for the question
primitives this replica supports. The response shape matches TypeSafe's docs:
``model``, ``answers`` (keyed by question id), and ``usage``.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from .engine import EngineConfig, GemmaLetterEngine
from .service import TypeSafeReplica

_api_key = os.environ.get("TYPESAFE_REPLICA_API_KEY")


class QuestionSpec(BaseModel):
    type: str
    instructions: str | dict | list
    criteria: dict | list | None = None


class SystemOneRequest(BaseModel):
    state: str | dict | list
    model: str = "gemma-4-12b"
    questions: dict[str, QuestionSpec]
    temperature: float = Field(default=1.0, gt=0)


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
    ))
    engine.load()
    _state["service"] = TypeSafeReplica(engine)
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
    return {"status": "ok", "model_loaded": bool(service and service.engine.is_loaded)}


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