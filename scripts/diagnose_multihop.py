"""Read-only retrieval/rerank diagnostic against the already-built Qdrant index."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config import COLLECTION_NAME
from src.m2_search import HybridSearch
from src.m3_rerank import CrossEncoderReranker
from src.m1_chunking import load_documents, chunk_hierarchical
from src.pipeline import _document_metadata, _expand_contexts, input_fingerprint


def main():
    saved = json.loads((ROOT / 'reports/production_answers.json').read_text(encoding='utf-8'))
    assert saved['input_fingerprint'] == input_fingerprint()
    query = saved['rows'][11]['question']
    search = HybridSearch()
    assert search.dense.backend == 'qdrant_server'
    points, next_offset = search.dense.client.scroll(COLLECTION_NAME, limit=1000, with_payload=True, with_vectors=False)
    assert next_offset is None, 'Diagnostic expects entire small lab index'
    points.sort(key=lambda p: p.id)
    search.bm25.index([{'text': point.payload['text'], 'metadata': point.payload} for point in points])
    documents = load_documents()
    search.parents = {}
    for doc, meta in zip(documents, _document_metadata(documents)):
        parents, _ = chunk_hierarchical(doc['text'], metadata=meta)
        for parent in parents:
            search.parents[parent.metadata['parent_id']] = {'text': parent.text, 'metadata': parent.metadata}
    candidates = [r for r in search.search(query) if not r.metadata.get('superseded')]
    docs = [{'text': f"{r.metadata.get('document_title', '')}\n{r.metadata['original_text']}",
             'score': r.score, 'metadata': r.metadata} for r in candidates]
    reranked = CrossEncoderReranker().rerank(query, docs, top_k=len(docs))
    contexts, sources = _expand_contexts(reranked, search)
    report = {'evidence_type': 'fresh_readonly_retrieval_diagnostic_no_answer_generation',
              'query': query, 'input_fingerprint': input_fingerprint(),
              'candidate_count': len(candidates),
              'fused_candidates': [{'source': r.metadata['source'], 'chunk_id': r.metadata['chunk_id'],
                                    'rrf_score': r.score} for r in candidates],
              'reranked': [{'source': r.metadata['source'], 'chunk_id': r.metadata['chunk_id'],
                            'rerank_score': r.rerank_score} for r in reranked],
              'selected_parent_sources': sources,
              'selected_contexts_match_original_run': contexts == saved['rows'][11]['contexts']}
    (ROOT / 'reports/multihop_diagnostic.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
