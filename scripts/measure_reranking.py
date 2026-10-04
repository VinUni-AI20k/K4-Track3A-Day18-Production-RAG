"""Measure real reranking on 20 candidates; separate cold load from warm calls."""

import json
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.m1_chunking import chunk_hierarchical, load_documents
from src.m2_search import BM25Search
from src.m3_rerank import CrossEncoderReranker, benchmark_reranker


def main():
    query = "Nhân viên được nghỉ phép bao nhiêu ngày?"
    chunks = []
    for document in load_documents():
        _, children = chunk_hierarchical(document["text"], metadata=document["metadata"])
        chunks.extend({"text": child.text, "metadata": child.metadata} for child in children)
    search = BM25Search()
    search.index(chunks)
    candidates = search.search(query, top_k=20)
    # Complete to 20 with corpus chunks if fewer have a positive lexical score.
    docs = [{"text": r.text, "score": r.score, "metadata": r.metadata} for r in candidates]
    selected = {d["metadata"]["chunk_id"] for d in docs}
    for chunk in chunks:
        if len(docs) == 20:
            break
        if chunk["metadata"]["chunk_id"] not in selected:
            docs.append({**chunk, "score": 0.})
            selected.add(chunk["metadata"]["chunk_id"])
    reranker = CrossEncoderReranker()
    started = time.perf_counter()
    model = reranker._load_model()
    load_ms = (time.perf_counter() - started) * 1000
    started = time.perf_counter()
    results = reranker.rerank(query, docs, top_k=3)
    first_ms = (time.perf_counter() - started) * 1000
    warm = benchmark_reranker(reranker, query, docs, n_runs=5)
    assert len(docs) == 20 and len(results) == 3
    assert all(results[i].rerank_score >= results[i + 1].rerank_score for i in range(2))
    assert "nghi_phep" in results[0].metadata["source"]
    report = {"model": reranker.model_name, "device": str(model.device),
              "python": platform.python_version(), "query": query,
              "candidate_method": "BM25 with corpus fill if needed",
              "candidate_count": len(docs), "top_k": 3,
              "model_load_ms": round(load_ms, 2), "first_inference_ms": round(first_ms, 2),
              "warm_runs": 5, "warm_latency_ms": warm,
              "results": [asdict(result) for result in results],
              "note": "Retrieval-only smoke, not answer accuracy or RAGAS evaluation."}
    path = ROOT / "reports" / "m3_reranking_report.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
