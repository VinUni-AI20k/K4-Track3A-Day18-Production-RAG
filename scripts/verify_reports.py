"""Verify saved evidence without judge/API/model calls. Run from any directory."""
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from main import reusable_baseline_report, reusable_production_report
from src.m1_chunking import load_documents, chunk_hierarchical
from src.pipeline import input_fingerprint, _evidence_contexts


def main():
    baseline, production = reusable_baseline_report(), reusable_production_report()
    assert baseline is not None, 'Baseline is incomplete or stale'
    assert production is not None, 'Production is incomplete or stale'
    assert production['num_questions'] == 20
    assert len(production['failures']) == 5
    assert production['run_metadata']['dense_backend'] == 'qdrant_server'
    answers = json.loads((ROOT / 'reports/production_answers.json').read_text(encoding='utf-8'))
    assert answers['input_fingerprint'] == input_fingerprint()
    assert len(answers['rows']) == 20
    parents = {}
    for doc in load_documents():
        ps, _ = chunk_hierarchical(doc['text'], metadata=doc['metadata'])
        for parent in ps:
            parents[parent.metadata['parent_id']] = parent
    evidence_count = 0
    for row, saved, sources in zip(production['per_question'], answers['rows'],
                                  production['run_metadata']['retrieval_sources']):
        assert all(row[k] == saved[k] for k in ('question', 'answer', 'ground_truth'))
        assert row['contexts'] == _evidence_contexts(saved['contexts'], sources)
        assert saved['metadata']['generation_mode'] == 'llm'
        assert 1 <= len(row['contexts']) <= 3
        assert len(sources) == len(row['contexts'])
        assert len({s['parent_id'] for s in sources}) == len(sources)
        for text, source in zip(saved['contexts'], sources):
            parent = parents[source['parent_id']]
            assert text == parent.text, 'Generated or modified text entered evidence'
            assert source['source'] == parent.metadata['source']
            evidence_count += 1
    check = subprocess.run(['git', '-c', 'safe.directory='+str(ROOT).replace('\\','/'),
                            'diff', '--exit-code', 'HEAD', '--', 'data', 'test_set.json'],
                           cwd=ROOT, capture_output=True)
    assert check.returncode == 0, 'Original corpus/test_set changed or cannot verify Git'
    report = {'baseline_status': baseline['eval_status'], 'production_status': production['eval_status'],
              'questions': 20, 'finite_scores_per_pipeline': 80, 'raw_parent_contexts_verified': evidence_count,
              'production_dense_backend': production['run_metadata']['dense_backend'],
              'original_corpus_test_unchanged': True, 'production_failures': 5}
    print(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    main()
