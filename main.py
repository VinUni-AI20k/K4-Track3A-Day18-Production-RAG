"""Lab 18 runner; baseline reuse is explicit and checked against saved provenance."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
from src.m4_eval import METRICS, load_test_set


def dataset_fingerprint():
    digest = hashlib.sha256()
    for path in sorted((ROOT / 'data').rglob('*')) + [ROOT / 'test_set.json']:
        if path.is_file():
            digest.update(str(path.relative_to(ROOT)).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def baseline_fingerprint():
    import config
    import importlib.metadata
    digest = hashlib.sha256(dataset_fingerprint().encode())
    for name in ('naive_baseline.py', 'config.py', 'src/m1_chunking.py',
                 'src/m2_search.py', 'src/m4_eval.py'):
        digest.update((ROOT / name).read_bytes())
    identities = [config.GENERATION_MODEL, config.OPENAI_BASE_URL, config.EMBEDDING_MODEL,
                  config.EVALUATION_MODEL, config.EVALUATION_EMBEDDING_BACKEND,
                  config.EVALUATION_EMBEDDING_MODEL]
    identities += [importlib.metadata.version(package) for package in
                   ('ragas', 'sentence-transformers', 'langchain-openai', 'openai')]
    digest.update(json.dumps(identities).encode())
    return digest.hexdigest()


def _complete_report(path):
    try:
        report = json.loads(path.read_text(encoding='utf-8'))
        tests = load_test_set()
        rows = report['per_question']
        if report['eval_status'] != 'complete' or report['num_questions'] != len(tests) or len(rows) != len(tests):
            return None
        if report.get('errors') or report['run_metadata'].get('generation_errors'):
            return None
        if report['run_metadata'].get('generation_success_count') != len(tests):
            return None
        for row, item in zip(rows, tests):
            if row['question'] != item['question'] or row['ground_truth'] != item['ground_truth']:
                return None
            if any(not isinstance(row.get(m), (int, float)) or not math.isfinite(row[m]) for m in METRICS):
                return None
        for metric in METRICS:
            if report['metric_counts'].get(metric) != len(tests):
                return None
            average = sum(row[metric] for row in rows) / len(rows)
            if not math.isclose(average, report['aggregate'][metric], abs_tol=1e-10):
                return None
        return report
    except (OSError, ValueError, KeyError, TypeError, ZeroDivisionError):
        return None


def reusable_production_report():
    from src.pipeline import input_fingerprint
    report = _complete_report(ROOT / 'reports' / 'ragas_report.json')
    return report if report and report['run_metadata'].get('input_fingerprint') == input_fingerprint() else None


def reusable_baseline_report():
    from config import GENERATION_MODEL, EVALUATION_MODEL, EVALUATION_EMBEDDING_MODEL
    report = _complete_report(ROOT / 'reports' / 'naive_baseline_report.json')
    if not report:
        return None
    meta, config = report['run_metadata'], report['evaluation_config']
    if (meta.get('baseline_fingerprint') != baseline_fingerprint()
            or meta.get('dataset_fingerprint') != dataset_fingerprint() or meta.get('generation_model') != GENERATION_MODEL
            or config.get('judge_model') != EVALUATION_MODEL or config.get('embedding_model') != EVALUATION_EMBEDDING_MODEL):
        return None
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reuse', action='store_true', help='Reuse only matching complete live reports; otherwise fail')
    parser.add_argument('--refresh-baseline', action='store_true', help='Pay for a new baseline run')
    args = parser.parse_args()
    naive = reusable_baseline_report()
    if args.reuse:
        prod = reusable_production_report()
        if not naive or not prod:
            raise SystemExit('Saved reports are missing, incomplete or changed; run without --reuse')
        print('Verified saved live baseline and production reports; no new API calls.')
    else:
        if args.refresh_baseline or not naive:
            from naive_baseline import main as run_baseline
            run_baseline()
            naive = reusable_baseline_report()
        else:
            print('Baseline: reusing matching complete live report (20 questions).')
        from src.pipeline import build_pipeline, evaluate_pipeline
        search, reranker = build_pipeline()
        evaluate_pipeline(search, reranker)
        prod = reusable_production_report()
    if not naive or not prod:
        raise SystemExit('Incomplete evaluation; inspect reports before submission')
    print(f"{'Metric':<23} {'Baseline':>10} {'Production':>12} {'Delta':>10}")
    for metric in METRICS:
        n, p = naive['aggregate'][metric], prod['aggregate'][metric]
        print(f'{metric:<23} {n:>10.4f} {p:>12.4f} {p-n:>+10.4f}')


if __name__ == '__main__':
    main()
