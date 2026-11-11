import datetime as dt
from types import SimpleNamespace

import numpy as np

from vocinsights import data, emerging
from vocinsights.emerging import DetectorConfig, _episodes, _windows


def test_windows_sum():
    a = np.arange(10, dtype=float)
    w = _windows(a, 3)
    assert w[2] == 0 + 1 + 2 and w[9] == 7 + 8 + 9 and w[0] == 0


def test_episodes_merge_small_gaps():
    flags = np.zeros(20, dtype=bool)
    flags[[2, 3, 5, 12]] = True
    assert _episodes(flags) == [(2, 5), (12, 12)]


def _synthetic(spike_day: int | None, per_day: int = 40, rate: float = 0.05, spike_rate: float = 0.2):
    """Items and decisions where theme 'slow_sync' runs at `rate` and jumps to `spike_rate`."""
    rng = np.random.default_rng(0)
    items, decisions = [], []
    for day in range(data.DAYS):
        d = (data.START + dt.timedelta(days=day)).isoformat()
        r = spike_rate if spike_day is not None and day >= spike_day else rate
        for _ in range(per_day):
            hit = rng.random() < r
            items.append(SimpleNamespace(date=d, week=d, id=f"{day}"))
            decisions.append(SimpleNamespace(themes=["slow_sync"] if hit else [], sentiments={"slow_sync": "negative"}))
    return items, decisions


def test_no_alert_on_flat_series():
    items, decisions = _synthetic(None)
    assert [a for a in emerging.detect(items, decisions) if a["kind"] == "volume"] == []


def test_spike_is_caught_within_a_few_days():
    items, decisions = _synthetic(200)
    vol = [a for a in emerging.detect(items, decisions) if a["kind"] == "volume"]
    assert vol and vol[0]["theme"] == "slow_sync"
    first = (dt.date.fromisoformat(vol[0]["first_day"]) - data.START).days
    assert 200 <= first <= 204


def test_stricter_config_is_slower():
    items, decisions = _synthetic(200, spike_rate=0.1)
    loose = emerging.detect(items, decisions, DetectorConfig(min_ratio=1.3))
    strict = emerging.detect(items, decisions, DetectorConfig(min_ratio=1.9))
    first = lambda al: min((a["first_day"] for a in al if a["kind"] == "volume"), default="9999")  # noqa: E731
    assert first(loose) <= first(strict)


def test_q4_bug_is_flagged(bench):
    det = bench.detection
    assert det["caught"]
    assert det["cluster_alert_lag_days"] is not None and 0 <= det["cluster_alert_lag_days"] <= 14
    assert det["theme_alert_lag_days"] is not None and det["theme_alert_lag_days"] <= 21


def test_new_cluster_is_the_widget_bug(bench):
    c = bench.detection["cluster"]
    assert "widget" in c["terms"] and c["nearest_theme"] == "app_stability"
    ids = set(c["doc_ids"])
    bug = {f.id for f in bench.items if "widget_crash_v6" in f.themes}
    precision = len(ids & bug) / len(ids)
    assert precision > 0.6


def test_no_new_clusters_before_release(bench):
    assert all(c["first_day"] >= data.RELEASE.isoformat() for c in bench.clusters)


def test_few_false_alarms(bench):
    assert len(bench.detection["other_alerts"]) <= 2


def test_weekly_series_totals(bench):
    s = emerging.weekly_series(bench.items, bench.decisions)
    assert sum(s["total"]) == len(bench.items)
    assert len(s["weeks"]) == len(s["count"]["slow_sync"])
