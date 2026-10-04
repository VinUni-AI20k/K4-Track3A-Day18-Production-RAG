from types import SimpleNamespace
import json
from src import pipeline as p


def test_evidence_labels_preserve_raw_text_and_metadata():
    raw = ['12 ký tự.', 'Không tự xử lý.']
    sources = [{'source': 'password.md'}, {'source': 'security.md'}]
    assert p._evidence_contexts(raw, sources) == [
        '[Nguồn: password.md]\n12 ký tự.', '[Nguồn: security.md]\nKhông tự xử lý.']
    assert raw == ['12 ký tự.', 'Không tự xử lý.']


def test_plan_only_splits_multi_part_queries_and_caches(tmp_path, monkeypatch):
    monkeypatch.setattr(p, 'PLAN_CACHE', tmp_path, raising=False)
    calls = []
    def request(query):
        calls.append(query)
        return json.dumps(['Senior nghỉ phép theo thâm niên?', 'Lương Senior bao nhiêu?'])
    monkeypatch.setattr(p, '_request_query_plan', request, raising=False)
    assert p._plan_queries('Mật khẩu dài bao nhiêu?')[0] == ['Mật khẩu dài bao nhiêu?']
    query = 'Senior được nghỉ bao nhiêu ngày và lương bao nhiêu?'
    assert len(p._plan_queries(query)[0]) == 2
    assert p._plan_queries(query)[1] == 'cache'
    assert calls == [query]


def test_invalid_plan_falls_back_without_answer_leakage(tmp_path, monkeypatch):
    monkeypatch.setattr(p, 'PLAN_CACHE', tmp_path, raising=False)
    monkeypatch.setattr(p, '_request_query_plan', lambda q: '{"answer":"35 triệu"}', raising=False)
    query = 'Ngày phép và lương?'
    assert p._plan_queries(query) == ([query], 'fallback')


def test_facet_coverage_selects_both_required_parents():
    results = [SimpleNamespace(text='child',score=.8, metadata={'parent_id': name,'source':name})
               for name in ('leave','noise','salary')]
    search = SimpleNamespace(parents={name:{'text':name+' raw','metadata':{'parent_id':name,'source':name}}
                                      for name in ('leave','noise','salary')})
    class Reranker:
        def rerank(self, query, docs, top_k):
            first = 'salary' if query == 'salary facet' else 'leave'
            return sorted([SimpleNamespace(text=d['text'],metadata=d['metadata'],rerank_score=1 if d['metadata']['source']==first else .1)
                           for d in docs],key=lambda r:r.rerank_score,reverse=True)
    ranked = p._rerank_with_coverage('full', ['leave facet','salary facet'],results,search,Reranker())
    contexts, sources = p._expand_contexts(ranked,search)
    assert [s['source'] for s in sources][:2] == ['leave','salary']
    assert contexts[:2] == ['leave raw','salary raw']


def test_generator_and_judge_receive_same_source_labels_checkpoint_stays_raw(tmp_path, monkeypatch):
    import openai
    from src.m4_eval import EvalResult, METRICS
    monkeypatch.setattr(p, 'ROOT', tmp_path)
    monkeypatch.setattr(p, 'input_fingerprint', lambda: 'test-version')
    monkeypatch.setattr(p, 'OPENAI_API_KEY', 'test-key')
    monkeypatch.setattr(p, 'load_test_set', lambda: [{'question':'Rule?', 'ground_truth':'Expected'}])
    captured = {}
    def create(**kwargs):
        captured['generator'] = kwargs['messages'][-1]['content']
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='Answer'))])
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr(openai, 'OpenAI', lambda **kwargs: client)
    def judge(questions, answers, contexts, truths):
        captured['judge'] = contexts
        return {**{m:1.0 for m in METRICS}, 'eval_status':'complete',
                'metric_counts':{m:1 for m in METRICS}, 'errors':[],
                'per_question':[EvalResult(questions[0],answers[0],contexts[0],truths[0],1.,1.,1.,1.)]}
    monkeypatch.setattr(p, 'evaluate_ragas', judge)
    result = SimpleNamespace(text='generated retrieval aid',score=.8,
                            metadata={'parent_id':'p1','source':'rule.md','original_text':'raw child'})
    search = SimpleNamespace(search=lambda q:[result],
        parents={'p1':{'text':'Full raw policy','metadata':{'source':'rule.md'}}},
        build_metadata={'dense_backend':'test'})
    reranker = SimpleNamespace(rerank=lambda query,docs,top_k:
                              [SimpleNamespace(text=d['text'],metadata=d['metadata']) for d in docs])
    p.evaluate_pipeline(search,reranker)
    framed = '[Nguồn: rule.md]\nFull raw policy'
    assert captured['judge'] == [[framed]]
    assert framed in captured['generator']
    checkpoint=json.loads((tmp_path/'reports/production_answers.json').read_text(encoding='utf-8'))
    assert checkpoint['rows'][0]['contexts'] == ['Full raw policy']
