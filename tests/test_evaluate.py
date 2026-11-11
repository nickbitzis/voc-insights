import numpy as np
import pytest

from vocinsights.evaluate import gate, multilabel_scores, prf, table


def test_prf_known_values():
    p, r, f = prf(tp=8, fp=2, fn=8)
    assert (p, r) == (0.8, 0.5) and f == pytest.approx(2 * 0.8 * 0.5 / 1.3)
    assert prf(0, 0, 0) == (0.0, 0.0, 0.0)


def test_micro_and_macro_f1_hand_computed():
    names = ("a", "b")
    y = np.array([[1, 0], [1, 0], [1, 0], [0, 1]])
    p = np.array([[1, 0], [1, 0], [0, 0], [1, 1]])
    m = multilabel_scores(y, p, names)
    # a: tp 2, fp 1, fn 1 -> f1 2/3. b: tp 1 -> f1 1. micro: tp 3, fp 1, fn 1 -> 0.75
    assert m["per_theme"]["a"]["f1"] == pytest.approx(0.6667, abs=1e-4)
    assert m["per_theme"]["b"]["f1"] == 1.0
    assert m["micro_f1"] == 0.75
    assert m["macro_f1"] == pytest.approx((2 / 3 + 1) / 2, abs=1e-4)
    assert m["exact_match"] == 0.5


def test_macro_skips_absent_themes():
    y = np.array([[1, 0, 0]])
    m = multilabel_scores(y, y.copy(), ("a", "b", "c"))
    assert m["macro_f1"] == 1.0


def test_report_rows_and_ordering(bench):
    rows = bench.evaluation["rows"]
    assert set(rows) == {"rules_only", "classifier_only", "llm_all", "cascade"}
    assert rows["rules_only"]["micro_f1"] < rows["classifier_only"]["micro_f1"] <= rows["cascade"]["micro_f1"] + 0.01
    assert rows["llm_all"]["llm_share"] == 1.0
    assert rows["cascade"]["cost_per_1k"] < rows["llm_all"]["cost_per_1k"] / 2


def test_table_marks_simulated_rows(bench):
    t = table(bench.evaluation)
    assert "(simulated)" in t and "Cascade" in t


def test_gate(bench):
    assert gate(bench.evaluation, floor=0.5) == []
    assert gate(bench.evaluation, floor=0.999)
