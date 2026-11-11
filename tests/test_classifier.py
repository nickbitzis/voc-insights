import numpy as np
import pytest

from vocinsights.classifier import ThemeClassifier, to_matrix
from vocinsights.taxonomy import THEME_NAMES


def test_to_matrix_ignores_unknown_labels():
    y = to_matrix([("slow_sync", "widget_crash_v6"), ()])
    assert y.shape == (2, len(THEME_NAMES)) and y.sum() == 1


def test_decide_threshold_and_confidence():
    p = np.array([[0.95, 0.02, 0.6], [0.4, 0.45, 0.1], [0.99, 0.01, 0.0]])
    y, conf = ThemeClassifier.decide(p)
    assert y.tolist() == [[1, 0, 1], [0, 0, 0], [1, 0, 0]]
    assert conf[0] == pytest.approx(0.2)  # 0.6 is the decision closest to the line
    assert conf[2] > 0.95 and conf[1] < 0.2


def test_multilabel_classifier_finds_both_themes(bench):
    clf = bench.cascade.clf
    p = clf.proba(["The app crashes all the time. I was charged twice this month."])
    y, _ = ThemeClassifier.decide(p)
    found = {THEME_NAMES[j] for j in np.flatnonzero(y[0])}
    assert {"app_stability", "billing_errors"} <= found


def test_classifier_handles_typos_and_paraphrase(bench):
    p = bench.cascade.clf.proba(["synk between my phone and laptop takes for ever"])
    assert THEME_NAMES[int(p[0].argmax())] == "slow_sync"


def test_probabilities_are_valid(bench):
    p = bench.cascade.clf.proba([f.text for f in bench.test[:50]])
    assert p.shape == (50, len(THEME_NAMES)) and (p >= 0).all() and (p <= 1).all()
