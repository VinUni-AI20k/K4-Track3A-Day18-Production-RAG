"""Behavior contracts missing from the original lab tests."""

import numpy as np
import pytest

from src import m1_chunking as m1
from src.m2_search import BM25Search, DenseSearch, SearchResult, reciprocal_rank_fusion


def test_hierarchy_limits_and_document_identity():
    text = "Đây là nội dung dài. " * 100
    pa, ca = m1.chunk_hierarchical(text, 200, 80, {"source": "a.md"})
    pb, cb = m1.chunk_hierarchical(text, 200, 80, {"source": "b.md"})
    assert pa and ca and pb and cb
    assert all(len(p.text) <= 200 for p in pa + pb)
    assert all(len(c.text) <= 80 for c in ca + cb)
    assert {p.metadata["parent_id"] for p in pa}.isdisjoint(
        {p.metadata["parent_id"] for p in pb}
    )
    by_id = {p.metadata["parent_id"]: p for p in pa + pb}
    assert all(c.parent_id in by_id for c in ca + cb)
    assert all(c.text in by_id[c.parent_id].text for c in ca + cb)


def test_hierarchy_preserves_long_paragraph_content():
    text = "abcdefghij" * 40
    parents, children = m1.chunk_hierarchical(text, 110, 30)
    assert parents and children
    assert "".join(p.text for p in parents) == text
    assert "".join(c.text for c in children) == text


@pytest.mark.parametrize("parent_size,child_size", [(0, 10), (10, 0), (10, 20)])
def test_hierarchy_rejects_invalid_sizes(parent_size, child_size):
    with pytest.raises(ValueError):
        m1.chunk_hierarchical("Nội dung.", parent_size, child_size)


def test_semantic_splits_topic_change_and_retains_source(monkeypatch):
    class Encoder:
        def encode(self, sentences, **kwargs):
            return np.array([[1., 0.], [1., 0.], [0., 1.]])

    monkeypatch.setattr(m1, "_get_semantic_model", lambda: Encoder(), raising=False)
    chunks = m1.chunk_semantic(
        "Nghỉ phép 15 ngày. Đăng ký trên HR Portal. VPN cần MFA.",
        threshold=0.8, metadata={"source": "policy.md"},
    )
    assert [c.text for c in chunks] == [
        "Nghỉ phép 15 ngày. Đăng ký trên HR Portal.", "VPN cần MFA.",
    ]
    assert all(c.metadata["source"] == "policy.md" for c in chunks)


def test_semantic_threshold_changes_grouping(monkeypatch):
    class Encoder:
        def encode(self, sentences, **kwargs):
            return np.array([[1., 0.], [.8, .6]])

    monkeypatch.setattr(m1, "_get_semantic_model", lambda: Encoder(), raising=False)
    assert len(m1.chunk_semantic("Câu một. Câu hai.", threshold=.9)) == 2
    assert len(m1.chunk_semantic("Câu một. Câu hai.", threshold=.7)) == 1


def test_structure_preserves_fenced_code_table_and_header_path():
    text = """# Chính sách
## Mua sắm
| Giá | Người duyệt |
| --- | --- |
| 55 triệu | CEO |
```markdown
# Heading inside code
```
### Hồ sơ
- Ba báo giá
"""
    chunks = m1.chunk_structure_aware(text, {"source": "mua_sam.md"})
    assert chunks
    table_chunk = next(c for c in chunks if "55 triệu" in c.text)
    assert "| Giá | Người duyệt |" in table_chunk.text
    assert "```markdown\n# Heading inside code\n```" in table_chunk.text
    assert not any(c.metadata["section"] == "# Heading inside code" for c in chunks)
    checklist = next(c for c in chunks if "Ba báo giá" in c.text)
    assert checklist.metadata["header_path"] == [
        "# Chính sách", "## Mua sắm", "### Hồ sơ",
    ]
    assert checklist.metadata["source"] == "mua_sam.md"


def test_structure_keeps_plain_text_without_heading():
    chunks = m1.chunk_structure_aware("Nội dung không có tiêu đề.")
    assert len(chunks) == 1
    assert chunks[0].text == "Nội dung không có tiêu đề."
    assert chunks[0].metadata["section"] == ""


def test_empty_inputs_do_not_create_chunks():
    assert m1.chunk_semantic("  \n ") == []
    assert m1.chunk_hierarchical("") == ([], [])
    assert m1.chunk_structure_aware(" \n ") == []


def test_rrf_adds_rank_contributions_and_preserves_sources():
    a = [SearchResult("x", 99., {}, "bm25"), SearchResult("y", 1., {}, "bm25")]
    b = [SearchResult("y", .8, {}, "dense")]
    fused = reciprocal_rank_fusion([a, b], k=60, top_k=2)
    assert fused and fused[0].text == "y"
    assert fused[0].score == pytest.approx(1 / 62 + 1 / 61)
    assert fused[0].method == "hybrid"
    different_sources = reciprocal_rank_fusion([
        [SearchResult("same text", 1., {"source": "old.md"}, "bm25")],
        [SearchResult("same text", 1., {"source": "new.md"}, "dense")],
    ])
    assert len(different_sources) == 2


def test_bm25_normalizes_case_and_handles_empty_input():
    search = BM25Search()
    search.index([])
    assert search.search("nghỉ phép") == []
    search.index([
        {"text": "Nhân viên được NGHỈ PHÉP.", "metadata": {"source": "leave"}},
        {"text": "VPN cần MFA.", "metadata": {}},
        {"text": "Mua laptop cần CEO duyệt.", "metadata": {}},
    ])
    results = search.search("nghỉ phép")
    assert results and results[0].metadata["source"] == "leave"
    assert search.search("", top_k=0) == []


def test_dense_indexes_payload_and_uses_requested_collection():
    from qdrant_client import QdrantClient

    class Encoder:
        def encode(self, texts, **kwargs):
            def vector(text):
                v = np.zeros(1024)
                v[0 if "leave" in text else 1] = 1.
                return v
            if isinstance(texts, str):
                return vector(texts)
            return np.array([vector(text) for text in texts])

    # Keep the real Qdrant client and our indexing/query logic, stub model inference only.
    search = DenseSearch.__new__(DenseSearch)
    search.client = QdrantClient(":memory:")
    search._encoder = Encoder()
    search.index([
        {"text": "leave policy", "metadata": {"source": "policy.md", "chunk_id": "p1"}},
        {"text": "vpn policy", "metadata": {"source": "it.md", "chunk_id": "p2"}},
    ], collection="contract_test")
    results = search.search("leave", collection="contract_test", top_k=1)
    assert len(results) == 1
    assert results[0].text == "leave policy"
    assert results[0].metadata["source"] == "policy.md"
    assert results[0].method == "dense"
    assert search.search("leave", collection="contract_test", top_k=0) == []
