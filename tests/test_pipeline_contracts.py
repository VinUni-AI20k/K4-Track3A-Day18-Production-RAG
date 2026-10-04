from types import SimpleNamespace
from src import pipeline as p


def result(parent, raw, source='policy.md'):
    return SimpleNamespace(text='LLM invented text', score=0.5, metadata={
        'parent_id': parent, 'original_text': raw, 'source': source})


def test_parent_expansion_deduplicates_and_never_uses_generated_text():
    search = SimpleNamespace(parents={'p1': {'text': 'Full raw policy', 'metadata': {'source': 'policy.md'}}})
    contexts, sources = p._expand_contexts([result('p1', 'child1'), result('p1', 'child2'),
                                           result('missing', 'raw child fallback')], search)
    assert contexts == ['Full raw policy', 'raw child fallback']
    assert sources[0]['parent_id'] == 'p1'
    assert all('invented' not in text for text in contexts)


def test_document_policy_versions_derived_from_headers():
    old = '# Leave (Phiên bản 2023)\n> Phiên bản: 1.0 | Ngày hiệu lực: 01/01/2023\n12 ngày'
    new = '# Leave (Phiên bản 2024)\n> Phiên bản: 2.0 | Ngày hiệu lực: 01/01/2024\nThay thế hoàn toàn phiên bản 1.0. 15 ngày'
    metas = p._document_metadata([{'text': old, 'metadata': {'source': 'old.md'}},
                                  {'text': new, 'metadata': {'source': 'new.md'}}])
    assert metas[0]['superseded'] is True
    assert metas[1]['superseded'] is False
    assert metas[1]['effective_date'] == '2024-01-01'


def test_empty_retrieval_does_not_call_generator(monkeypatch):
    search = SimpleNamespace(search=lambda q: [], parents={})
    reranker = SimpleNamespace(rerank=lambda *a, **k: [])
    def forbidden(*args):
        raise AssertionError('No evidence: should not call API')
    monkeypatch.setattr(p, '_generate_answer', forbidden, raising=False)
    answer, contexts = p.run_query('Unknown?', search, reranker)
    assert contexts == []
    assert 'Không tìm thấy' in answer


def test_current_query_filters_superseded_but_historical_keeps_it(monkeypatch):
    old, new = result('old', 'old raw'), result('new', 'new raw')
    old.metadata['superseded'] = True
    new.metadata['superseded'] = False
    search = SimpleNamespace(search=lambda q: [old, new], parents={})
    class Reranker:
        def rerank(self, query, docs, top_k):
            return [SimpleNamespace(text=d['text'], metadata=d['metadata']) for d in docs]
    monkeypatch.setattr(p, '_generate_answer', lambda *args: 'answer', raising=False)
    assert p.run_query('Leave now?', search, Reranker())[1] == ['new raw']
    assert p.run_query('Leave in 2023?', search, Reranker())[1] == ['old raw', 'new raw']
    assert p.run_query('Theo phiên bản 1.0?', search, Reranker())[1] == ['old raw', 'new raw']
    assert p.run_query('Theo v1.0?', search, Reranker())[1] == ['old raw', 'new raw']
