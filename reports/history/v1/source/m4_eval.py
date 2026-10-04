from __future__ import annotations

"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import os, sys, json
import math
from datetime import datetime, timezone
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import asdict, dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH

METRICS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float | None
    answer_relevancy: float | None
    context_precision: float | None
    context_recall: float | None


def _run_ragas(dataset):
    """External judge boundary, using RAGAS 0.1.x with explicit project clients."""
    os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
    from ragas.run_config import RunConfig
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from config import (OPENAI_API_KEY, OPENAI_BASE_URL, EVALUATION_MODEL,
                        EVALUATION_EMBEDDING_BACKEND, EVALUATION_EMBEDDING_MODEL)

    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is missing")
    llm = ChatOpenAI(model=EVALUATION_MODEL, api_key=OPENAI_API_KEY,
                     base_url=OPENAI_BASE_URL, temperature=0, timeout=60, max_retries=0)
    if EVALUATION_EMBEDDING_BACKEND == "local":
        from langchain_community.embeddings import HuggingFaceEmbeddings
        embeddings = HuggingFaceEmbeddings(model_name=EVALUATION_EMBEDDING_MODEL,
            model_kwargs={"local_files_only": True}, encode_kwargs={"normalize_embeddings": True})
    elif EVALUATION_EMBEDDING_BACKEND == "openai":
        embeddings = OpenAIEmbeddings(model=EVALUATION_EMBEDDING_MODEL,
            api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL, max_retries=0)
    else:
        raise ValueError("Unsupported evaluation embedding backend")
    return evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
                    llm=llm, embeddings=embeddings, raise_exceptions=False,
                    run_config=RunConfig(timeout=120, max_retries=1, max_workers=2, max_wait=5))


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation."""
    if len({len(questions), len(answers), len(contexts), len(ground_truths)}) != 1:
        raise ValueError("Evaluation inputs must have the same length")
    if any(not isinstance(c, list) or any(not isinstance(t, str) for t in c) for c in contexts):
        raise ValueError("contexts must be list[list[str]]")
    if any(not isinstance(v, str) for values in (questions, answers, ground_truths) for v in values):
        raise ValueError("questions, answers and ground_truths must contain strings")
    from config import EVALUATION_MODEL, EVALUATION_EMBEDDING_BACKEND, EVALUATION_EMBEDDING_MODEL
    import importlib.metadata
    result = {**{metric: 0. for metric in METRICS}, "eval_status": "not_run", "errors": [],
              "metric_counts": {metric: 0 for metric in METRICS},
              "evaluation_config": {"judge_model": EVALUATION_MODEL,
                  "embedding_backend": EVALUATION_EMBEDDING_BACKEND,
                  "embedding_model": EVALUATION_EMBEDDING_MODEL,
                  "ragas_version": importlib.metadata.version("ragas"),
                  "answer_relevancy_strictness": 3},
              "per_question": [EvalResult(q, a, c, gt, None, None, None, None)
                  for q, a, c, gt in zip(questions, answers, contexts, ground_truths)]}
    if not questions:
        return result
    try:
        from datasets import Dataset
        dataset = Dataset.from_dict({"question": questions, "answer": answers,
                                     "contexts": contexts, "ground_truth": ground_truths})
        frame = _run_ragas(dataset).to_pandas()
        if len(frame) != len(questions):
            raise ValueError("RAGAS result row count does not match input")
        rows = []
        for index, (_, row) in enumerate(frame.iterrows()):
            scores = {}
            for metric in METRICS:
                value = float(row.get(metric, float("nan")))
                scores[metric] = value if math.isfinite(value) else None
                if scores[metric] is None:
                    result["errors"].append({"row_index": index, "metric": metric, "type": "NonFiniteMetric"})
            rows.append(EvalResult(questions[index], answers[index], contexts[index], ground_truths[index], **scores))
        result["per_question"] = rows
        for metric in METRICS:
            values = [getattr(row, metric) for row in rows if getattr(row, metric) is not None]
            result[metric] = sum(values) / len(values) if values else 0.
            result["metric_counts"][metric] = len(values)
        result["eval_status"] = "partial" if result["errors"] else "complete"
    except Exception as exc:
        # Exception messages can contain provider credentials/URLs; persist type/status only.
        result["eval_status"] = "failed"
        result["errors"].append({"type": type(exc).__name__, "http_status": getattr(exc, "status_code", None)})
        print(f"  RAGAS evaluation failed: {type(exc).__name__}", flush=True)
    return result


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    if bottom_n <= 0:
        return []
    diagnostic_tree = {
        "faithfulness": ("Có khẳng định chưa được context hỗ trợ; cần đọc lại chứng cứ.",
                         "Đối chiếu từng khẳng định, siết prompt và giữ nguyên văn nguồn.",
                         "Answer sai → Context đủ? → Nếu đủ, kiểm tra khẳng định LLM thêm vào."),
        "answer_relevancy": ("Answer có thể lệch yêu cầu hoặc thiếu phần hỏi.",
                             "Kiểm tra câu hỏi nhiều ý và hướng dẫn trả lời trực tiếp.",
                             "Answer sai → Context đúng? → Kiểm tra answer có trả lời đủ ý hỏi."),
        "context_precision": ("Có thể lấy hoặc xếp hạng nhiều context không liên quan.",
                              "Kiểm tra top-k, reranking và metadata/phiên bản.",
                              "Answer sai → Context lẫn nhiễu? → Kiểm tra ranking và phiên bản."),
        "context_recall": ("Context có thể thiếu thông tin cần thiết.",
                           "Kiểm tra corpus/chunk boundaries, hybrid retrieval và parent expansion.",
                           "Answer sai → Context thiếu? → Corpus có thông tin? → Kiểm tra chunk/index/search."),
    }
    ranked = []
    for row in eval_results:
        scores = {metric: getattr(row, metric) for metric in METRICS}
        if any(value is None or not math.isfinite(value) for value in scores.values()):
            continue
        ranked.append((sum(scores.values()) / len(METRICS), row, scores))
    ranked.sort(key=lambda item: item[0])
    failures = []
    for average, row, scores in ranked[:bottom_n]:
        worst = min(scores, key=scores.get)
        diagnosis, fix, tree = diagnostic_tree[worst]
        failures.append({"question": row.question, "answer": row.answer, "contexts": row.contexts,
            "ground_truth": row.ground_truth, "score": average, "metrics": scores, "worst_metric": worst,
            "diagnosis": diagnosis, "suggested_fix": fix, "error_tree": tree,
            "diagnosis_status": "metric_based_hypothesis_requires_context_review"})
    return failures


def save_report(results: dict, failures: list[dict], path: str = "reports/ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    report = {
        # Keep numeric fallback keys for scaffold callers, but never publish missing
        # measurements as zero scores in an evaluation artifact.
        "aggregate": {metric: (None if results.get("metric_counts", {}).get(metric) == 0
                               else results.get(metric, 0.)) for metric in METRICS},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
        "per_question": [asdict(row) for row in results.get("per_question", [])],
        "eval_status": results.get("eval_status", "unknown"),
        "metric_counts": results.get("metric_counts", {}),
        "evaluation_config": results.get("evaluation_config", {}),
        "errors": results.get("errors", []),
        "run_metadata": results.get("run_metadata", {}),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
