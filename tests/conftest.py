"""Keep scaffold evaluation unit tests from spending money on external judges."""

import pytest


@pytest.fixture(autouse=True)
def isolate_scaffold_evaluator(request, monkeypatch):
    if request.module.__name__.endswith("test_m5"):
        from src import m5_enrichment
        monkeypatch.setattr(m5_enrichment, "OPENAI_API_KEY", "")
        monkeypatch.setattr(m5_enrichment, "CACHE_DIR", request.getfixturevalue("tmp_path"))
    if request.module.__name__.endswith("test_m4"):
        from src import m4_eval

        def unavailable_judge(dataset):
            raise RuntimeError("External evaluator disabled in scaffold unit tests")

        monkeypatch.setattr(m4_eval, "_run_ragas", unavailable_judge, raising=False)
