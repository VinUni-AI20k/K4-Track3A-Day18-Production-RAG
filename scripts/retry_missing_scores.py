"""Recover only missing live judge measurements, preserving successful scores.

Same model, evidence, metric and runtime configuration as src.m4_eval.
Archives the partial artifact and records recovery provenance explicitly.
"""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.m4_eval import METRICS, EvalResult, failure_analysis


def finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def merge_missing_scores(report, updates):
    result = deepcopy(report)
    for (index, metric), value in updates.items():
        if metric not in METRICS or not finite(value) or not 0 <= value <= 1:
            raise ValueError('Recovery requires a finite measured score')
        if finite(result['per_question'][index].get(metric)):
            raise ValueError('Cannot overwrite a successful measurement')
        result['per_question'][index][metric] = value
    rows = result['per_question']
    result['errors'] = [{'row_index':i, 'metric':m, 'type':'NonFiniteMetric'}
                        for i, row in enumerate(rows) for m in METRICS if not finite(row.get(m))]
    result['metric_counts'] = {m:sum(finite(row.get(m)) for row in rows) for m in METRICS}
    result['aggregate'] = {m:sum(row[m] for row in rows if finite(row.get(m))) / count if count else None
                           for m, count in result['metric_counts'].items()}
    result['eval_status'] = 'partial' if result['errors'] else 'complete'
    fields = ('question', 'answer', 'contexts', 'ground_truth') + METRICS
    result['failures'] = failure_analysis([EvalResult(**{k:row[k] for k in fields}) for row in rows], bottom_n=5)
    return result


def main():
    from src.pipeline import input_fingerprint
    path = ROOT / 'reports/ragas_report.json'
    original = path.read_bytes()
    report = json.loads(original)
    assert report['run_metadata']['input_fingerprint'] == input_fingerprint(), 'Stale source or inputs'
    assert report['run_metadata']['generation_success_count'] == len(report['per_question']) == 20
    missing = [(i, m) for i, row in enumerate(report['per_question']) for m in METRICS if not finite(row.get(m))]
    if not missing:
        print('No missing scores; no API calls.')
        return
    assert report['eval_status'] == 'partial', 'Need a partial report with live rows'
    from config import (OPENAI_API_KEY, OPENAI_BASE_URL, EVALUATION_MODEL,
                        EVALUATION_EMBEDDING_BACKEND, EVALUATION_EMBEDDING_MODEL)
    cfg = report['evaluation_config']
    import importlib.metadata
    assert cfg['ragas_version'] == importlib.metadata.version('ragas')
    assert cfg['judge_model'] == EVALUATION_MODEL and cfg['embedding_model'] == EVALUATION_EMBEDDING_MODEL
    assert cfg['embedding_backend'] == EVALUATION_EMBEDDING_BACKEND == 'local'
    assert cfg['answer_relevancy_strictness'] == 3
    assert all(e.get('type') == 'NonFiniteMetric' for e in report['errors']), 'Unsupported evaluation failure'
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    archive = ROOT / f'reports/history/v2_partial/{stamp}_ragas_report.json'
    archive.parent.mkdir(parents=True, exist_ok=True)
    archive.write_bytes(original)
    start = time.perf_counter()
    os.environ.setdefault('RAGAS_DO_NOT_TRACK', 'true')
    from datasets import Dataset
    from ragas import evaluate
    from ragas import metrics as ragas_metrics
    from ragas.run_config import RunConfig
    from langchain_openai import ChatOpenAI
    from langchain_community.embeddings import HuggingFaceEmbeddings
    llm = ChatOpenAI(model=EVALUATION_MODEL, api_key=OPENAI_API_KEY,
                    base_url=OPENAI_BASE_URL, temperature=0, timeout=60, max_retries=0)
    embeddings = HuggingFaceEmbeddings(model_name=EVALUATION_EMBEDDING_MODEL,
        model_kwargs={'local_files_only':True}, encode_kwargs={'normalize_embeddings':True})
    updates, attempts = {}, []
    for index, metric in missing:
        row = report['per_question'][index]
        dataset = Dataset.from_dict({k:[row[k]] for k in ('question','answer','contexts','ground_truth')})
        print(f'Retry question {index+1}: {metric}', flush=True)
        attempt = {'row_index':index, 'metric':metric}
        try:
            frame = evaluate(dataset, metrics=[getattr(ragas_metrics, metric)], llm=llm, embeddings=embeddings,
                raise_exceptions=False, run_config=RunConfig(timeout=120,max_retries=1,max_workers=2,max_wait=5)).to_pandas()
            value = float(frame.iloc[0][metric])
            if finite(value):
                updates[index, metric] = value
                attempt['score'] = value
            else:
                attempt['error_type'] = 'NonFiniteMetric'
        except Exception as exc:
            attempt['error_type'] = type(exc).__name__
        attempts.append(attempt)
    result = merge_missing_scores(report, updates)
    elapsed = (time.perf_counter()-start)*1000
    result['run_metadata'].setdefault('evaluation_recovery', []).append({
        'original_report':str(archive.relative_to(ROOT)).replace('\\','/'),
        'original_report_sha256':hashlib.sha256(original).hexdigest(), 'original_errors':report['errors'],
        'attempts':attempts, 'elapsed_ms':elapsed, 'updated_at_utc':datetime.now(timezone.utc).isoformat(),
        'policy':'Only missing cells retried; successful measurements and evidence preserved; same judge configuration'})
    result['run_metadata']['evaluation_ms'] += elapsed
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(f'Recovered {len(updates)}/{len(missing)} cells; status={result["eval_status"]}')
    if result['eval_status'] != 'complete':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
