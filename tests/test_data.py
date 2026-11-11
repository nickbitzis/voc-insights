import datetime as dt
from collections import Counter

from vocinsights import data
from vocinsights.taxonomy import NEW_ISSUE, THEME_NAMES


def test_generator_is_deterministic():
    a = data.generate(n=1500, seed=11)
    b = data.generate(n=1500, seed=11)
    assert [(f.text, f.themes, f.score) for f in a] == [(f.text, f.themes, f.score) for f in b]


def test_different_seeds_differ():
    a = data.generate(n=500, seed=1)
    b = data.generate(n=500, seed=2)
    assert [f.text for f in a] != [f.text for f in b]


def test_shape_and_sources():
    items = data.generate(n=6000, seed=3)
    assert len(items) == 6000
    src = Counter(f.source for f in items)
    assert abs(src["nps"] / 6000 - 0.4) < 0.03 and abs(src["ticket"] / 6000 - 0.25) < 0.03
    assert set(f.quarter for f in items) == set(data.QUARTERS)


def test_scores_match_source():
    for f in data.generate(n=3000, seed=4):
        if f.source == "nps":
            assert 0 <= f.score <= 10
        elif f.source == "review":
            assert 1 <= f.score <= 5
        else:
            assert f.score is None


def test_multi_theme_sarcasm_and_general_shares():
    items = data.generate(n=8000, seed=5)
    multi = sum(len(f.themes) > 1 for f in items) / len(items)
    general = sum(len(f.themes) == 0 for f in items) / len(items)
    sarcastic = sum(f.sarcastic for f in items) / len(items)
    assert 0.3 < multi < 0.55
    assert 0.03 < general < 0.07
    assert 0.01 < sarcastic < 0.08


def test_planted_q4_issue_only_after_release():
    items = data.generate(n=25000, seed=7)
    bug = [f for f in items if NEW_ISSUE.name in f.themes]
    assert len(bug) > 150
    assert all(dt.date.fromisoformat(f.date) >= data.RELEASE for f in bug)
    assert all(f.quarter == "2025Q4" for f in bug)
    assert all(f.sentiments["app_stability"] == "negative" for f in bug)


def test_bug_rate_ramps_then_fades():
    assert data.bug_rate(data.RELEASE - dt.timedelta(days=1)) == 0
    assert data.bug_rate(data.RELEASE + dt.timedelta(days=14)) > data.bug_rate(data.RELEASE)
    assert data.bug_rate(data.HOTFIX + dt.timedelta(days=30)) < 0.01


def test_price_change_lifts_pricing_share():
    items = data.generate(n=20000, seed=7)

    def share(q):
        qs = [f for f in items if f.quarter == q]
        return sum("pricing_too_high" in f.themes for f in qs) / len(qs)

    assert share("2025Q3") > 1.4 * share("2025Q2")


def test_labeled_subset_and_label_noise():
    items = data.generate(n=10000, seed=9)
    lab = [f for f in items if f.labeled]
    assert len(lab) == 2000
    assert all(set(f.labels) <= set(THEME_NAMES) for f in lab)
    differ = sum(set(f.labels) != (set(f.themes) & set(THEME_NAMES)) for f in lab) / len(lab)
    assert 0.02 < differ < 0.12
    train, test = data.split_labeled(items, seed=9)
    assert len(train) + len(test) == 2000 and not {f.id for f in train} & {f.id for f in test}


def test_overall_of():
    assert data.overall_of({"a": "positive", "b": "negative"}) == "mixed"
    assert data.overall_of({"a": "neutral"}) == "neutral"
    assert data.overall_of({"a": "negative", "b": "neutral"}) == "negative"
