"""Reranker boundary cases; stub inference, exercise actual ranking logic."""

import numpy as np
import pytest

from src.m3_rerank import CrossEncoderReranker, benchmark_reranker


def test_rerank_preserves_identity_and_original_score():
    class Model:
        def predict(self, pairs, **kwargs):
            assert pairs == [("question", "VPN"), ("question", "Leave")]
            return np.array([-2., 3.])

    reranker = CrossEncoderReranker()
    reranker._model = Model()
    docs = [{"text": "VPN", "score": .9, "metadata": {"chunk_id": "vpn"}},
            {"text": "Leave", "score": .4, "metadata": {"chunk_id": "leave"}}]
    results = reranker.rerank("question", docs, top_k=1)
    assert len(results) == 1
    assert results[0].text == "Leave"
    assert results[0].rerank_score == 3.
    assert results[0].original_score == .4
    assert results[0].metadata == {"chunk_id": "leave"}
    assert docs[0]["text"] == "VPN"


@pytest.mark.parametrize("prediction", [2.5, np.float32(2.5), np.array([2.5]), np.array([[2.5]])])
def test_rerank_accepts_single_document_predictions(prediction):
    class Model:
        def predict(self, pairs, **kwargs):
            return prediction

    reranker = CrossEncoderReranker()
    reranker._model = Model()
    results = reranker.rerank("query", [{"text": "Only document"}])
    assert len(results) == 1
    assert results[0].rerank_score == 2.5
    assert results[0].original_score == 0.
    assert results[0].metadata == {}


@pytest.mark.parametrize("prediction", [[1.], [1., float("nan")]])
def test_rerank_rejects_invalid_predictions(prediction):
    class Model:
        def predict(self, pairs, **kwargs):
            return prediction

    reranker = CrossEncoderReranker()
    reranker._model = Model()
    with pytest.raises(ValueError):
        reranker.rerank("query", [{"text": "one"}, {"text": "two"}])


def test_rerank_empty_and_zero_top_k_do_not_load_model(monkeypatch):
    reranker = CrossEncoderReranker()

    def unexpected_load():
        raise AssertionError("Empty requests must not load the model")

    monkeypatch.setattr(reranker, "_load_model", unexpected_load)
    assert reranker.rerank("query", []) == []
    assert reranker.rerank("query", [{"text": "one"}], top_k=0) == []


def test_benchmark_rejects_zero_runs():
    with pytest.raises(ValueError):
        benchmark_reranker(CrossEncoderReranker(), "query", [], n_runs=0)
def test_optional_flashrank_fails_explicitly():
    from src.m3_rerank import FlashrankReranker
    import pytest
    with pytest.raises(NotImplementedError):
        FlashrankReranker().rerank('query', [{'text': 'raw'}])
