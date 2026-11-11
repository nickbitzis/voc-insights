"""NPS driver analysis and how NPS relates to aspect sentiment.

Drivers: ordinary least squares of the 0 to 10 score on predicted theme presence, with
heteroskedasticity-robust (HC1) standard errors. A coefficient of -2.1 reads as "comments that
mention this theme score 2.1 points lower, holding the other themes fixed".

Caveats that go in the report with the numbers: this is observational, not causal. Only people who
leave a comment are in it. Themes come from a classifier, so errors in tagging dilute effects.
Themes that travel together share credit. A large coefficient on a rare theme moves overall NPS less
than a medium one on a common theme, so `impact` (coefficient times prevalence) is shown as well.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.stats import t as student_t

from .taxonomy import BY_NAME, LABELS, THEME_NAMES

CAVEATS = (
    "Associations, not causes: the regression shows which themes travel with lower or higher scores.",
    "Only respondents who wrote a comment are included, and themes come from the classifier, so tagging "
    "errors shrink the effects toward zero.",
    "Themes that are often mentioned together share credit, so read close coefficients as a tie.",
)


def ols(x: np.ndarray, y: np.ndarray) -> dict:
    """OLS with an intercept and HC1 robust standard errors."""
    n, k = x.shape
    xi = np.hstack([np.ones((n, 1)), x])
    xtx_inv = np.linalg.pinv(xi.T @ xi)
    beta = xtx_inv @ xi.T @ y
    resid = y - xi @ beta
    meat = (xi * resid[:, None] ** 2).T @ xi
    cov = xtx_inv @ meat @ xtx_inv * n / max(1, n - k - 1)
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    tval = np.divide(beta, se, out=np.zeros_like(beta), where=se > 0)
    p = 2 * student_t.sf(np.abs(tval), df=max(1, n - k - 1))
    r2 = 1 - (resid**2).sum() / max(1e-12, ((y - y.mean()) ** 2).sum())
    return {"beta": beta, "se": se, "p": p, "r2": float(r2), "n": n}


def planted_effect(name: str) -> float:
    """Expected score change from a mention, given the generator's planted effects and sentiment mix."""
    t = BY_NAME[name]
    p_neu = (1 - t.p_pos) * t.p_neu
    p_neg = 1 - t.p_pos - p_neu
    return t.p_pos * t.effect_pos + p_neg * t.effect_neg


def nps_drivers(items, decisions, names=THEME_NAMES, min_mentions: int = 30) -> dict:
    rows = [(f, d) for f, d in zip(items, decisions, strict=True) if f.source == "nps"]
    y = np.array([f.score for f, _ in rows], dtype=float)
    x = np.array([[int(n in d.themes) for n in names] for _, d in rows], dtype=float)
    keep = [j for j in range(len(names)) if x[:, j].sum() >= min_mentions]
    fit = ols(x[:, keep], y)
    out = []
    for pos, j in enumerate(keep, start=1):
        name = names[j]
        coef, se = float(fit["beta"][pos]), float(fit["se"][pos])
        prev = float(x[:, j].mean())
        out.append(
            {
                "theme": name,
                "label": LABELS[name],
                "coef": round(coef, 3),
                "ci_low": round(coef - 1.96 * se, 3),
                "ci_high": round(coef + 1.96 * se, 3),
                "p_value": float(f"{fit['p'][pos]:.3g}"),
                "prevalence": round(prev, 4),
                "impact": round(coef * prev, 4),
                "planted": round(planted_effect(name), 3) if name in BY_NAME else None,
            }
        )
    out.sort(key=lambda r: r["coef"])
    return {
        "n": fit["n"],
        "r2": round(fit["r2"], 4),
        "intercept": round(float(fit["beta"][0]), 3),
        "drivers": out,
        "caveats": list(CAVEATS),
    }


def nps(scores) -> float:
    scores = list(scores)
    if not scores:
        return 0.0
    pro = sum(1 for s in scores if s >= 9)
    det = sum(1 for s in scores if s <= 6)
    return round(100 * (pro - det) / len(scores), 1)


def nps_vs_sentiment(items, decisions) -> dict:
    """Mean score by number of negative aspects, and mean score when each theme is positive vs negative."""
    rows = [(f, d) for f, d in zip(items, decisions, strict=True) if f.source == "nps"]
    by_neg: dict[int, list[int]] = defaultdict(list)
    by_theme: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for f, d in rows:
        k = sum(1 for s in d.sentiments.values() if s == "negative")
        by_neg[min(k, 3)].append(f.score)
        for t, s in d.sentiments.items():
            by_theme[t][s].append(f.score)
    neg_rows = [
        {
            "negative_aspects": f"{k}+" if k == 3 else str(k),
            "n": len(v),
            "mean_score": round(float(np.mean(v)), 2),
            "nps": nps(v),
        }
        for k, v in sorted(by_neg.items())
    ]
    theme_rows = []
    for t in THEME_NAMES:
        d = by_theme.get(t, {})
        row = {"theme": t, "label": LABELS[t]}
        for s in ("negative", "neutral", "positive"):
            v = d.get(s, [])
            row[s] = {"n": len(v), "mean_score": round(float(np.mean(v)), 2) if v else None}
        theme_rows.append(row)
    scores = np.array([f.score for f, _ in rows], dtype=float)
    net = np.array([sum({"positive": 1, "negative": -1}.get(s, 0) for s in d.sentiments.values()) for _, d in rows])
    corr = float(np.corrcoef(scores, net)[0, 1]) if len(rows) > 2 and net.std() > 0 else 0.0
    return {"by_negative_aspects": neg_rows, "by_theme": theme_rows, "corr_score_net_sentiment": round(corr, 3)}
