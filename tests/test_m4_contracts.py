"""Evaluation/report contracts; fake only the external judge boundary."""

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from src import m4_eval as m4


def test_evaluate_maps_inputs_and_aggregates_real_rows(monkeypatch):
    def judge(dataset):
        assert dataset.to_dict() == {"question": ["q1", "q2"], "answer": ["a1", "a2"],
            "contexts": [["c1"], ["c2"]], "ground_truth": ["gt1", "gt2"]}
        frame = pd.DataFrame({**dataset.to_dict(), "faithfulness": [.8, .6],
            "answer_relevancy": [.9, .7], "context_precision": [1., .5], "context_recall": [.5, 1.]})
        return SimpleNamespace(to_pandas=lambda: frame)

    monkeypatch.setattr(m4, "_run_ragas", judge, raising=False)
    result = m4.evaluate_ragas(["q1", "q2"], ["a1", "a2"], [["c1"], ["c2"]], ["gt1", "gt2"])
    assert result["faithfulness"] == pytest.approx(.7)
    assert result["context_precision"] == .75
    assert result["eval_status"] == "complete"
    assert len(result["per_question"]) == 2
    assert result["per_question"][1].answer == "a2"


def test_evaluation_failure_keeps_evidence_without_fabricated_scores(monkeypatch):
    def judge(dataset):
        raise RuntimeError("external service failed")

    monkeypatch.setattr(m4, "_run_ragas", judge, raising=False)
    result = m4.evaluate_ragas(["q"], ["a"], [["c"]], ["gt"])
    assert result["eval_status"] == "failed"
    assert result["per_question"][0].contexts == ["c"]
    assert result["per_question"][0].faithfulness is None
    assert result["errors"][0]["type"] == "RuntimeError"


@pytest.mark.parametrize("questions,answers,contexts,truths", [
    (["q"], [], [["c"]], ["gt"]), (["q"], ["a"], ["not a list"], ["gt"]),
])
def test_evaluation_rejects_invalid_inputs(questions, answers, contexts, truths):
    with pytest.raises(ValueError):
        m4.evaluate_ragas(questions, answers, contexts, truths)


def test_empty_evaluation_is_not_a_completed_run():
    result = m4.evaluate_ragas([], [], [], [])
    assert result["eval_status"] == "not_run"
    assert result["per_question"] == []


def test_partial_eval_preserves_missing_metric_as_null(monkeypatch, tmp_path):
    def judge(dataset):
        frame = pd.DataFrame({**dataset.to_dict(), "faithfulness": [float("nan")],
            "answer_relevancy": [.8], "context_precision": [.7], "context_recall": [.6]})
        return SimpleNamespace(to_pandas=lambda: frame)

    monkeypatch.setattr(m4, "_run_ragas", judge, raising=False)
    result = m4.evaluate_ragas(["q"], ["a"], [["c"]], ["gt"])
    assert result["eval_status"] == "partial"
    assert result["metric_counts"]["faithfulness"] == 0
    path = tmp_path / "report.json"
    m4.save_report(result, [], str(path))
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["per_question"][0]["faithfulness"] is None
    assert report["aggregate"]["faithfulness"] is None
    assert "NaN" not in path.read_text(encoding="utf-8")


def test_report_keeps_answer_context_and_run_metadata(tmp_path):
    row = m4.EvalResult("q", "a", ["c"], "gt", .8, .7, .6, .9)
    result = {"faithfulness": .8, "answer_relevancy": .7,
              "context_precision": .6, "context_recall": .9,
              "per_question": [row], "eval_status": "complete",
              "evaluation_config": {"judge_model": "example", "embedding_backend": "local"}}
    path = tmp_path / "report.json"
    m4.save_report(result, [], str(path))
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["num_questions"] == 1
    assert report["per_question"][0]["answer"] == "a"
    assert report["per_question"][0]["contexts"] == ["c"]
    assert report["eval_status"] == "complete"
    assert set(report["aggregate"]) == set(m4.METRICS)
    assert report["evaluation_config"]["judge_model"] == "example"


def test_bottom_failures_sort_by_mean_and_include_diagnostic_tree():
    rows = [m4.EvalResult("good", "a", ["c"], "gt", .9, .9, .9, .9),
            m4.EvalResult("missing", "a", ["c"], "gt", .8, .8, .7, .1),
            m4.EvalResult("unsupported", "a", ["c"], "gt", .1, .2, .3, .4)]
    failures = m4.failure_analysis(rows, bottom_n=2)
    assert [row["question"] for row in failures] == ["unsupported", "missing"]
    assert failures[0]["worst_metric"] == "faithfulness"
    assert failures[0]["score"] == pytest.approx(.25)
    assert failures[1]["worst_metric"] == "context_recall"
    assert all(row["error_tree"] and row["diagnosis"] and row["suggested_fix"] for row in failures)
    assert m4.failure_analysis(rows, bottom_n=0) == []


def test_failed_evaluation_rows_are_not_quality_failures():
    row = m4.EvalResult("q", "a", ["c"], "gt", None, None, None, None)
    assert m4.failure_analysis([row]) == []


def test_project_env_overrides_process_and_normalizes_proxy_root(tmp_path):
    source = Path(__file__).resolve().parents[1] / "config.py"
    copied = tmp_path / "config.py"
    copied.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    (tmp_path / ".env").write_text("OPENAI_API_KEY=project_key\nOPENAI_BASE_URL=https://proxy.invalid\n", encoding="utf-8")
    script = "import runpy,json; c=runpy.run_path(" + repr(str(copied)) + "); print(json.dumps({'project_key_used':c['OPENAI_API_KEY']=='project_key','endpoint':c.get('OPENAI_BASE_URL')}))"
    result = subprocess.run([sys.executable, "-c", script], cwd=tmp_path,
        env={**os.environ, "OPENAI_API_KEY": "other_key", "OPENAI_BASE_URL": "https://other.invalid/v1"},
        capture_output=True, text=True, check=True)
    parsed = json.loads(result.stdout)
    assert parsed == {"project_key_used": True, "endpoint": "https://proxy.invalid/v1"}
