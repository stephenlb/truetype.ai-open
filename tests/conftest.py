"""Shared fixtures for the 50-test verification suite.

The Gemma 4 12B model loads once per test session (module-scoped engine) and is
shared across all functional tests. Each question class submits its cases as a
single batched ``system_one`` call, mirroring the TypeSafe batch API contract.
"""

from __future__ import annotations

import pytest

from src.truetype.engine import EngineConfig, GemmaLetterEngine
from src.truetype.service import TypeSafeReplica


@pytest.fixture(scope="session")
def engine() -> GemmaLetterEngine:
    eng = GemmaLetterEngine(EngineConfig(top_k=5))
    eng.load()
    return eng


@pytest.fixture(scope="session")
def service(engine: GemmaLetterEngine) -> TypeSafeReplica:
    return TypeSafeReplica(engine)
