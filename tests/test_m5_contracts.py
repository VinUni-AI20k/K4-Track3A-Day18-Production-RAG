import json

from src import m5_enrichment as m5


def response():
    return {"summary": "Nghỉ phép năm.", "questions": ["Nghỉ bao nhiêu ngày?"],
            "context": "Chính sách nhân sự.",
            "metadata": {"topic": "leave", "source": "invented", "parent_id": "bad"}}


def test_combined_preserves_evidence_and_source(monkeypatch):
    monkeypatch.setattr(m5, "_enrich_single_call", lambda *args: response())
    original = {"text": "Nhân viên được nghỉ 15 ngày.",
                "metadata": {"source": "leave.md", "parent_id": "p1", "chunk_id": "c1"}}
    result = m5.enrich_chunks([original])[0]
    assert result.original_text == original["text"]
    assert result.auto_metadata["source"] == "leave.md"
    assert result.auto_metadata["parent_id"] == "p1"
    assert result.auto_metadata["chunk_id"] == "c1"
    assert response()["questions"][0] in result.enriched_text
    assert response()["summary"] in result.enriched_text


def test_success_cache_avoids_second_request(tmp_path, monkeypatch):
    monkeypatch.setattr(m5, "CACHE_DIR", tmp_path, raising=False)
    monkeypatch.setattr(m5, "OPENAI_API_KEY", "unit-key")
    calls = []
    def request(*args):
        calls.append(args)
        return json.dumps(response())
    monkeypatch.setattr(m5, "_request_enrichment", request, raising=False)
    first = m5._enrich_single_call("Nghỉ phép 15 ngày.", "leave.md")
    second = m5._enrich_single_call("Nghỉ phép 15 ngày.", "leave.md")
    assert len(calls) == 1
    assert first["status"] == "llm"
    assert second["status"] == "cache"


def test_malformed_response_falls_back_without_caching(tmp_path, monkeypatch):
    monkeypatch.setattr(m5, "CACHE_DIR", tmp_path, raising=False)
    monkeypatch.setattr(m5, "OPENAI_API_KEY", "unit-key")
    monkeypatch.setattr(m5, "_request_enrichment", lambda *a: '{"questions": "wrong"}', raising=False)
    result = m5._enrich_single_call("Chính sách nghỉ phép.", "leave.md")
    assert result["status"] == "fallback"
    assert result["summary"]
    assert isinstance(result["questions"], list)
    assert not list(tmp_path.glob("*.json"))


def test_failed_request_never_leaks_exception(tmp_path, monkeypatch):
    monkeypatch.setattr(m5, "CACHE_DIR", tmp_path, raising=False)
    monkeypatch.setattr(m5, "OPENAI_API_KEY", "unit-key")
    def failed(*args):
        raise RuntimeError("secret credential")
    monkeypatch.setattr(m5, "_request_enrichment", failed, raising=False)
    result = m5._enrich_single_call("Nội dung chính sách.", "policy.md")
    assert result["status"] == "fallback"
    assert "secret" not in json.dumps(result)


def test_empty_input_and_invalid_methods():
    assert m5.enrich_chunks([]) == []
    import pytest
    with pytest.raises(ValueError):
        m5.enrich_chunks([], methods=["unknown"])
