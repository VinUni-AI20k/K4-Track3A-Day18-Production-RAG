import json
from src.m4_eval import METRICS
import main


def test_reuse_rejects_incomplete_or_changed_reports(tmp_path, monkeypatch):
    from src import pipeline
    monkeypatch.setattr(main, 'ROOT', tmp_path)
    monkeypatch.setattr(pipeline, 'input_fingerprint', lambda: 'current')
    monkeypatch.setattr(main, 'load_test_set', lambda: [{'question': 'q', 'ground_truth': 'gt'}], raising=False)
    (tmp_path / 'reports').mkdir()
    path = tmp_path / 'reports' / 'ragas_report.json'
    report = {'eval_status': 'complete', 'num_questions': 1,
              'per_question': [{'question': 'q', 'ground_truth': 'gt', **{m: 0.8 for m in METRICS}}],
              'aggregate': {m: 0.8 for m in METRICS}, 'metric_counts': {m: 1 for m in METRICS},
              'run_metadata': {'input_fingerprint': 'current', 'generation_success_count': 1,
                               'generation_errors': []}, 'errors': []}
    path.write_text(json.dumps(report), encoding='utf-8')
    assert main.reusable_production_report() is not None
    report['run_metadata']['input_fingerprint'] = 'stale'
    path.write_text(json.dumps(report), encoding='utf-8')
    assert main.reusable_production_report() is None
    report['run_metadata']['input_fingerprint'] = 'current'
    report['eval_status'] = 'partial'
    path.write_text(json.dumps(report), encoding='utf-8')
    assert main.reusable_production_report() is None


def test_baseline_reuse_requires_implementation_and_config_fingerprint(monkeypatch):
    import config
    report = {'run_metadata': {'dataset_fingerprint': 'data', 'generation_model': config.GENERATION_MODEL},
              'evaluation_config': {'judge_model': config.EVALUATION_MODEL,
                                    'embedding_model': config.EVALUATION_EMBEDDING_MODEL}}
    monkeypatch.setattr(main, '_complete_report', lambda *a: report)
    monkeypatch.setattr(main, 'dataset_fingerprint', lambda: 'data')
    monkeypatch.setattr(main, 'baseline_fingerprint', lambda: 'current-code', raising=False)
    assert main.reusable_baseline_report() is None
    report['run_metadata']['baseline_fingerprint'] = 'current-code'
    assert main.reusable_baseline_report() is report
    report['run_metadata']['baseline_fingerprint'] = 'old-code'
    assert main.reusable_baseline_report() is None
