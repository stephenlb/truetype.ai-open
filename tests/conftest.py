"""Shared fixtures for the 50-test verification suite.

The Gemma 4 12B model loads once per test session (module-scoped engine) and is
shared across all functional tests. Each question class submits its cases as a
single batched ``system_one`` call, mirroring the TypeSafe batch API contract.
"""

from __future__ import annotations

import os

import pytest

from src.truetype.engine import EngineConfig, GemmaLetterEngine
from src.truetype.service import TypeSafeReplica


@pytest.fixture(scope="session")
def engine() -> GemmaLetterEngine:
    device = os.environ.get("TRUETYPE_TEST_DEVICE", "auto")
    if device == "mps":
        import torch

        if not torch.backends.mps.is_available():
            pytest.fail("TRUETYPE_TEST_DEVICE=mps requires an available PyTorch MPS device")
    eng = GemmaLetterEngine(EngineConfig(top_k=5, device=device))
    eng.load()
    if device == "mps":
        assert eng._model.device.type == "mps"
    return eng


@pytest.fixture(scope="session")
def service(engine: GemmaLetterEngine) -> TypeSafeReplica:
    return TypeSafeReplica(engine)
