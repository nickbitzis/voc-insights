"""Emerging issue detection.

Every day, the last 7 days of predicted theme counts are tested against a trailing 8 week baseline
that ends where the window starts:
- Volume: a one-sided Poisson test of the window count against the baseline rate times the window
  volume. Flag when p is below `p_max`, the count is at least `min_count`, and the ratio is at least
  `min_ratio`.
- Negative share: a one-sided two-proportion z-test of the theme's negative share in the window
  against the baseline. Flag when p is below `p_max` and the share rose by at least `min_lift`.
Consecutive flagged days become one alert episode.

New-cluster detection looks for issues no theme covers: terms (unigrams and bigrams) whose document
count in the window bursts against their own baseline, grouped by co-occurrence into clusters and
named by their top terms.
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm, poisson
from sklearn.feature_extraction.text import CountVectorizer

from .data import DAYS, START
from .discovery import STOP
from .taxonomy import LABELS, THEME_NAMES


@dataclass(frozen=True)
class DetectorConfig:
    window: int = 7
    baseline_days: int = 56
    p_max: float = 1e-4
    min_count: int = 10
    min_ratio: float = 1.5
    min_lift: float = 0.12
    term_p_max: float = 1e-8
    term_min_count: int = 12
    term_min_ratio: float = 4.0


def day_index(date: str) -> int:
    return (dt.date.fromisoformat(date) - START).days


def day_date(i: int) -> str:
    return (START + dt.timedelta(days=int(i))).isoformat()


def daily_counts(items, decisions, names=THEME_NAMES) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Totals per day, theme mentions per day, and negative mentions per day."""
    idx = {n: j for j, n in enumerate(names)}
    total = np.zeros(DAYS)
    cnt = np.zeros((DAYS, len(names)))
    neg = np.zeros((DAYS, len(names)))
    for f, d in zip(items, decisions, strict=True):
        i = day_index(f.date)
        total[i] += 1
        for t in d.themes:
            if t in idx:
                cnt[i, idx[t]] += 1
                if d.sentiments.get(t) == "negative":
                    neg[i, idx[t]] += 1
    return total, cnt, neg


def _windows(a: np.ndarray, size: int) -> np.ndarray:
    """Sum of the `size` days ending on each day (rows before the first full window are zero)."""
    c = np.concatenate([np.zeros((1,) + a.shape[1:]), np.cumsum(a, axis=0)], axis=0)
    out = np.zeros_like(a, dtype=float)
    out[size - 1 :] = c[size:] - c[:-size]
    return out


def _episodes(flags: np.ndarray, gap: int = 2) -> list[tuple[int, int]]:
    days = np.flatnonzero(flags)
    eps: list[list[int]] = []
    for d in days:
        if eps and d - eps[-1][1] <= gap:
            eps[-1][1] = int(d)
        else:
            eps.append([int(d), int(d)])
    return [(a, b) for a, b in eps]


def detect(items, decisions, cfg: DetectorConfig | None = None, names=THEME_NAMES) -> list[dict]:
    cfg = cfg or DetectorConfig()
    total, cnt, neg = daily_counts(items, decisions, names)
    w_tot, w_cnt, w_neg = _windows(total, cfg.window), _windows(cnt, cfg.window), _windows(neg, cfg.window)
    b_tot, b_cnt, b_neg = (_windows(x, cfg.baseline_days) for x in (total, cnt, neg))
    start = cfg.window + cfg.baseline_days - 1
    alerts = []
    for j, name in enumerate(names):
        vol_p = np.ones(DAYS)
        vol_ratio = np.zeros(DAYS)
        neg_p = np.ones(DAYS)
        neg_lift = np.zeros(DAYS)
        for i in range(start, DAYS):
            b = i - cfg.window  # baseline ends the day before the window starts
            if b_tot[b] <= 0:
                continue
            lam = max(0.5, b_cnt[b, j] / b_tot[b] * w_tot[i])
            x = w_cnt[i, j]
            vol_ratio[i] = x / lam
            vol_p[i] = poisson.sf(x - 1, lam)
            if x >= 15 and b_cnt[b, j] >= 15:
                p1, p0 = w_neg[i, j] / x, b_neg[b, j] / b_cnt[b, j]
                pooled = (w_neg[i, j] + b_neg[b, j]) / (x + b_cnt[b, j])
                se = math.sqrt(max(pooled * (1 - pooled) * (1 / x + 1 / b_cnt[b, j]), 1e-12))
                neg_lift[i] = p1 - p0
                neg_p[i] = norm.sf((p1 - p0) / se)
        vol_flags = (vol_p < cfg.p_max) & (w_cnt[:, j] >= cfg.min_count) & (vol_ratio >= cfg.min_ratio)
        neg_flags = (neg_p < cfg.p_max) & (neg_lift >= cfg.min_lift)
        for kind, flags, stat in (("volume", vol_flags, vol_ratio), ("negative_share", neg_flags, neg_lift)):
            for a, b in _episodes(flags):
                peak = a + int(np.argmax(stat[a : b + 1]))
                alerts.append(
                    {
                        "theme": name,
                        "label": LABELS.get(name, name),
                        "kind": kind,
                        "first_day": day_date(a),
                        "last_day": day_date(b),
                        "days_flagged": int(flags[a : b + 1].sum()),
                        "peak_day": day_date(peak),
                        "peak_ratio" if kind == "volume" else "peak_lift": round(float(stat[peak]), 3),
                        "window_count": int(w_cnt[peak, j]),
                        "p_value": float(f"{(vol_p if kind == 'volume' else neg_p)[peak]:.3g}"),
                    }
                )
    alerts.sort(key=lambda a: a["first_day"])
    return alerts


def new_clusters(items, decisions=None, cfg: DetectorConfig | None = None, max_terms: int = 8) -> list[dict]:
    """Bursting terms grouped into clusters of co-occurring terms. Finds issues outside the taxonomy."""
    cfg = cfg or DetectorConfig()
    texts = [f.text for f in items]
    vec = CountVectorizer(binary=True, min_df=8, ngram_range=(1, 2), stop_words=STOP)
    x = vec.fit_transform(texts).tocsc()
    terms = vec.get_feature_names_out()
    days = np.array([day_index(f.date) for f in items])
    # Daily document frequency per term: (days x terms), via a sparse day indicator.
    from scipy.sparse import csr_matrix

    ind = csr_matrix((np.ones(len(items)), (days, np.arange(len(items)))), shape=(DAYS, len(items)))
    df = (ind @ x).toarray()
    total = np.bincount(days, minlength=DAYS).astype(float)
    w_df, b_df = _windows(df, cfg.window), _windows(df, cfg.baseline_days)
    w_tot, b_tot = _windows(total, cfg.window), _windows(total, cfg.baseline_days)
    start = cfg.window + cfg.baseline_days - 1
    burst_days: dict[int, list[int]] = {}
    for i in range(start, DAYS):
        b = i - cfg.window
        lam = np.maximum(0.5, (b_df[b] + 0.5) / b_tot[b] * w_tot[i])
        obs = w_df[i]
        cand = np.flatnonzero((obs >= cfg.term_min_count) & (obs / lam >= cfg.term_min_ratio))
        hits = [int(t) for t in cand if poisson.sf(obs[t] - 1, lam[t]) < cfg.term_p_max]
        if hits:
            burst_days[i] = hits
    if not burst_days:
        return []
    # Group days into episodes, then terms within an episode by co-occurrence in the same documents.
    clusters = []
    flags = np.zeros(DAYS, dtype=bool)
    flags[list(burst_days)] = True
    for a, b in _episodes(flags, gap=3):
        ep_terms = sorted({t for i in range(a, b + 1) for t in burst_days.get(i, [])})
        in_ep = np.flatnonzero((days >= a - cfg.window + 1) & (days <= b))
        sub = x[in_ep][:, ep_terms].toarray().astype(bool)
        groups = _merge_by_docs(sub, ep_terms, _cooccur_groups(sub, ep_terms))
        for g in groups:
            cols = [ep_terms.index(t) for t in g]
            matched = sub[:, cols].sum(axis=1)
            member = matched >= (2 if len(g) >= 3 else 1)  # one generic word is not enough
            docs = in_ep[member]
            if len(docs) < cfg.term_min_count:
                continue
            docs = docs[np.argsort(-matched[member], kind="stable")]
            order = sorted(g, key=lambda t: -int(sub[member, ep_terms.index(t)].sum()))[:max_terms]
            top_theme = None
            if decisions is not None:
                from collections import Counter

                c = Counter(t for k in docs for t in decisions[k].themes)
                top_theme = c.most_common(1)[0][0] if c else None
            clusters.append(
                {
                    "name": " / ".join(str(terms[t]) for t in order[:3]),
                    "terms": [str(terms[t]) for t in order],
                    "first_day": day_date(a),
                    "last_day": day_date(b),
                    "size": int(len(docs)),
                    "doc_ids": [items[k].id for k in docs[:2000]],
                    "examples": [items[k].text for k in docs[:3]],
                    "nearest_theme": top_theme,
                }
            )
    clusters.sort(key=lambda c: c["first_day"])
    return clusters


def _merge_by_docs(sub: np.ndarray, ep_terms: list[int], groups: list[list[int]], min_overlap: float = 0.2):
    """Merge term groups whose documents overlap: two phrasings of the same complaint are one issue."""
    docsets = [set(np.flatnonzero(sub[:, [ep_terms.index(t) for t in g]].any(axis=1))) for g in groups]
    merged = True
    while merged:
        merged = False
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                small = min(len(docsets[i]), len(docsets[j])) or 1
                if len(docsets[i] & docsets[j]) / small >= min_overlap:
                    groups[i] = groups[i] + groups[j]
                    docsets[i] |= docsets[j]
                    del groups[j], docsets[j]
                    merged = True
                    break
            if merged:
                break
    return groups


def _cooccur_groups(sub: np.ndarray, ep_terms: list[int], min_jaccard: float = 0.25) -> list[list[int]]:
    n = len(ep_terms)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            inter = np.logical_and(sub[:, i], sub[:, j]).sum()
            union = np.logical_or(sub[:, i], sub[:, j]).sum()
            if union and inter / union >= min_jaccard:
                parent[find(i)] = find(j)
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(ep_terms[i])
    return list(groups.values())


def weekly_series(items, decisions, names=THEME_NAMES) -> dict:
    """Per-week totals, theme counts, and negative counts, for charts."""
    weeks = sorted({f.week for f in items})
    wi = {w: k for k, w in enumerate(weeks)}
    idx = {n: j for j, n in enumerate(names)}
    total = [0] * len(weeks)
    cnt = {n: [0] * len(weeks) for n in names}
    neg = {n: [0] * len(weeks) for n in names}
    for f, d in zip(items, decisions, strict=True):
        k = wi[f.week]
        total[k] += 1
        for t in d.themes:
            if t in idx:
                cnt[t][k] += 1
                neg[t][k] += int(d.sentiments.get(t) == "negative")
    return {"weeks": weeks, "total": total, "count": cnt, "negative": neg}
