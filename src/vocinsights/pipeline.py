"""End to end build: generate, train, evaluate, classify everything, and compute the analytics."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from . import data, discovery, drivers, emerging
from .cascade import Cascade, Decision, Policy
from .classifier import ThemeClassifier
from .evaluate import compare
from .llm import SimulatedTagger, Tagger
from .sentiment import AspectSentiment
from .taxonomy import NEW_ISSUE


@dataclass
class Bench:
    items: list
    train: list
    test: list
    cascade: Cascade
    evaluation: dict
    decisions: list[Decision]
    alerts: list[dict]
    clusters: list[dict]
    drivers: dict
    nps_sentiment: dict
    discovery: dict
    detection: dict
    timings: dict = field(default_factory=dict)


def planted_detection(alerts: list[dict], clusters: list[dict]) -> dict:
    """When was the quarter 4 release bug caught, by the theme tests and by new-cluster detection."""
    release = data.RELEASE.isoformat()
    vol = [a for a in alerts if a["theme"] == "app_stability" and a["kind"] == "volume" and a["first_day"] >= release]
    new = [c for c in clusters if c["first_day"] >= release and any(t in c["terms"] for t in ("widget", "v6", "6"))]
    planted = {("app_stability", release), ("pricing_too_high", data.PRICE_CHANGE.isoformat())}

    def is_planted(a: dict) -> bool:
        for theme, start in planted:
            end = (data.dt.date.fromisoformat(start) + data.dt.timedelta(days=70)).isoformat()
            if a["theme"] == theme and start <= a["first_day"] <= end:
                return True
        return False

    def lag(day: str | None) -> int | None:
        return None if day is None else (data.dt.date.fromisoformat(day) - data.RELEASE).days

    first_theme = vol[0]["first_day"] if vol else None
    first_cluster = new[0]["first_day"] if new else None
    return {
        "release": release,
        "hotfix": data.HOTFIX.isoformat(),
        "price_change": data.PRICE_CHANGE.isoformat(),
        "new_issue": NEW_ISSUE.name,
        "theme_alert_day": first_theme,
        "theme_alert_lag_days": lag(first_theme),
        "cluster_alert_day": first_cluster,
        "cluster_alert_lag_days": lag(first_cluster),
        "cluster": new[0] if new else None,
        "caught": bool(vol or new),
        "pricing_caught": any(a["theme"] == "pricing_too_high" and is_planted(a) for a in alerts),
        "other_alerts": [a for a in alerts if not is_planted(a)],
    }


def build(
    n: int = 25_000,
    seed: int = 7,
    llm: Tagger | None = None,
    policy: Policy | None = None,
    discovery_sample: int = 8000,
) -> Bench:
    t: dict[str, float] = {}
    t0 = time.perf_counter()
    items = data.generate(n=n, seed=seed)
    train, test = data.split_labeled(items, seed=seed)
    t["generate"] = time.perf_counter() - t0
    clf = ThemeClassifier(seed=seed).fit([f.text for f in train], [f.labels for f in train])
    sent = AspectSentiment().fit(train)
    sim = SimulatedTagger()
    cascade = Cascade(clf, sent, sim, policy or Policy())
    t["train"] = time.perf_counter() - t0
    evaluation = compare(cascade, test)
    op = evaluation["operating_point"]["policy"]
    cascade.policy = policy or Policy(t_rules=op["t_rules"], t_clf=op["t_clf"])
    t["evaluate"] = time.perf_counter() - t0
    # The batch analysis always uses the simulated tagger so it is free and repeatable.
    decisions = cascade.run([f.text for f in items], golds=items)
    if llm is not None:
        cascade.llm = llm  # live requests through the API use the configured tagger
    t["classify_all"] = time.perf_counter() - t0
    alerts = emerging.detect(items, decisions)
    clusters = emerging.new_clusters(items, decisions)
    t["emerging"] = time.perf_counter() - t0
    drv = drivers.nps_drivers(items, decisions)
    ns = drivers.nps_vs_sentiment(items, decisions)
    disc = {
        f"{m}_k{k}": discovery.discover(items, k=k, method=m, sample=discovery_sample, seed=seed)
        for m, k in (("nmf", 20), ("kmeans", 20), ("nmf", 40))
    }
    t["discovery"] = time.perf_counter() - t0
    det = planted_detection(alerts, clusters)
    return Bench(
        items,
        train,
        test,
        cascade,
        evaluation,
        decisions,
        alerts,
        clusters,
        drv,
        ns,
        disc,
        det,
        {k: round(v, 2) for k, v in t.items()},
    )
