from copy import deepcopy
import pytest
from scripts.retry_missing_scores import merge_missing_scores
from src.m4_eval import METRICS


def fixture_report():
    row = {'question':'Question', 'answer':'Live answer', 'contexts':['Original evidence'], 'ground_truth':'Golden'}
    row.update({m: 1.0 for m in METRICS})
    row['faithfulness'] = None
    return {'per_question':[row], 'run_metadata':{'input_fingerprint':'same'}, 'eval_status':'partial',
            'aggregate':{}, 'metric_counts':{}, 'errors':[{'row_index':0,'metric':'faithfulness','type':'NonFiniteMetric'}]}


def test_recovery_fills_only_missing_score_and_recomputes_report():
    report = fixture_report()
    before = deepcopy(report)
    recovered = merge_missing_scores(report, {(0,'faithfulness'):.5})
    assert report == before
    assert recovered['eval_status'] == 'complete'
    assert recovered['errors'] == []
    assert recovered['aggregate']['faithfulness'] == .5
    assert recovered['metric_counts'] == {m:1 for m in METRICS}
    assert recovered['per_question'][0] == {**before['per_question'][0], 'faithfulness':.5}
    assert recovered['failures'][0]['metrics']['faithfulness'] == .5


def test_recovery_refuses_overwriting_successful_measurements():
    with pytest.raises(ValueError):
        merge_missing_scores(fixture_report(), {(0,'context_precision'):.5})


@pytest.mark.parametrize('value', [float('nan'), None])
def test_failed_retry_cannot_be_published_as_complete(value):
    with pytest.raises(ValueError):
        merge_missing_scores(fixture_report(), {(0,'faithfulness'):value})
