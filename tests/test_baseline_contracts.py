"""Verify baseline generation consumes retrieval context, not ground truth."""

from types import SimpleNamespace

import pytest

import config
import naive_baseline


def test_baseline_uses_configured_model_and_retrieved_context(monkeypatch):
    monkeypatch.setattr(config, "GENERATION_MODEL", "project-model")

    def create(**kwargs):
        assert kwargs["model"] == "project-model"
        user_message = kwargs["messages"][-1]["content"]
        assert "Source says 15 days" in user_message
        assert "How many days?" in user_message
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="15 days."))])

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    assert naive_baseline._generate_answer(client, "How many days?", ["Source says 15 days"]) == "15 days."


def test_baseline_rejects_empty_model_answer():
    def create(**kwargs):
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=None))])

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    with pytest.raises(ValueError):
        naive_baseline._generate_answer(client, "query", ["context"])
