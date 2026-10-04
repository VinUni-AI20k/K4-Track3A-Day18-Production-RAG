from types import SimpleNamespace
import check_lab


def test_pytest_collection_errors_count_as_failed_verification(monkeypatch):
    monkeypatch.setattr(check_lab.subprocess, 'run', lambda *a, **k: SimpleNamespace(
        stdout='3 passed, 1 error in 0.1s', returncode=1))
    assert check_lab.run_tests() == (3, 4)


def test_missing_analysis_and_reflection_block_submission(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(check_lab, 'check_file', lambda path, required=True: path != 'analysis/failure_analysis.md')
    monkeypatch.setattr(check_lab, 'check_json', lambda *args: True)
    monkeypatch.setattr(check_lab, 'check_production_report', lambda: True, raising=False)
    monkeypatch.setattr(check_lab, 'check_todos', lambda: 0)
    monkeypatch.setattr(check_lab, 'run_tests', lambda: (10, 10))
    assert check_lab.validate() >= 2
