"""Multi-label metrics and the policy comparison behind the README evaluation table."""

from __future__ import annotations

import numpy as np

from .cascade import Cascade, Decision, Policy
from .classifier import to_matrix
from .sentiment import evaluate_aspects
from .taxonomy import THEME_NAMES

DEFAULT_F1_FLOOR = 0.93
F1_SLACK = 0.005  # take the cheapest point within half a point of the best micro F1


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def multilabel_scores(y_true: np.ndarray, y_pred: np.ndarray, names=THEME_NAMES) -> dict:
    """Micro F1 pools every (text, theme) decision; macro F1 averages per-theme F1 so rare themes count
    as much as common ones. Themes with no gold and no predicted examples are left out of the macro."""
    tp = ((y_true == 1) & (y_pred == 1)).sum(axis=0)
    fp = ((y_true == 0) & (y_pred == 1)).sum(axis=0)
    fn = ((y_true == 1) & (y_pred == 0)).sum(axis=0)
    per = {}
    for j, n in enumerate(names):
        p, r, f = prf(int(tp[j]), int(fp[j]), int(fn[j]))
        per[n] = {
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f, 4),
            "support": int(y_true[:, j].sum()),
        }
    mp, mr, mf = prf(int(tp.sum()), int(fp.sum()), int(fn.sum()))
    active = [per[n]["f1"] for j, n in enumerate(names) if tp[j] + fp[j] + fn[j] > 0]
    exact = float((y_true == y_pred).all(axis=1).mean()) if len(y_true) else 0.0
    return {
        "micro_precision": round(mp, 4),
        "micro_recall": round(mr, 4),
        "micro_f1": round(mf, 4),
        "macro_f1": round(float(np.mean(active)), 4) if active else 0.0,
        "exact_match": round(exact, 4),
        "per_theme": per,
    }


def score_decisions(items, decisions: list[Decision]) -> dict:
    y_true = to_matrix([f.themes for f in items])
    y_pred = to_matrix([d.themes for d in decisions])
    m = multilabel_scores(y_true, y_pred)
    tiers: dict[str, int] = {}
    for d in decisions:
        tiers[d.tier] = tiers.get(d.tier, 0) + 1
    n = len(decisions)
    m["cost_per_1k"] = round(sum(d.cost for d in decisions) / n * 1000, 4) if n else 0.0
    m["tier_share"] = {k: round(v / n, 4) for k, v in sorted(tiers.items())}
    m["llm_share"] = round(tiers.get("llm", 0) / n, 4) if n else 0.0
    m["sentiment"] = evaluate_aspects(items, [d.sentiments for d in decisions])
    return m


def compare(cascade: Cascade, items, thresholds=(0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)) -> dict:
    """Score single tiers and a threshold sweep on the labeled test set, then pick an operating point."""
    texts = [f.text for f in items]
    probs = cascade.clf.proba(texts)
    base = cascade.policy

    def run(policy: Policy) -> dict:
        cascade.policy = policy
        return score_decisions(items, cascade.run(texts, golds=items, probs=probs))

    rows = {
        "rules_only": _rules_only(cascade, items, texts),
        "classifier_only": run(Policy(t_rules=None, t_clf=0.0, use_llm=False)),
        "llm_all": run(Policy(t_rules=None, t_clf=2.0, use_llm=True)),
    }
    sweep = []
    for t_rules in (None, 0.9):
        for t in thresholds:
            pol = Policy(t_rules=t_rules, t_clf=t)
            m = run(pol)
            sweep.append({"policy": {"t_rules": t_rules, "t_clf": t}, "label": pol.label(), **_brief(m)})
    best = max(s["micro_f1"] for s in sweep)
    eligible = [s for s in sweep if s["micro_f1"] >= best - F1_SLACK]
    chosen = min(eligible, key=lambda s: (s["cost_per_1k"], -s["micro_f1"]))
    pol = Policy(t_rules=chosen["policy"]["t_rules"], t_clf=chosen["policy"]["t_clf"])
    rows["cascade"] = run(pol)
    cascade.policy = base
    return {
        "rows": rows,
        "sweep": sweep,
        "operating_point": {"policy": chosen["policy"], "label": pol.label()},
        "n_test": len(items),
    }


def _rules_only(cascade: Cascade, items, texts) -> dict:
    from . import rules

    decisions = []
    for text in texts:
        r = rules.match(text)
        decisions.append(cascade._local(text, r.themes, "rules", r.confidence, {}))
    return score_decisions(items, decisions)


def _brief(m: dict) -> dict:
    return {k: m[k] for k in ("micro_f1", "macro_f1", "cost_per_1k", "llm_share")}


ROW_NAMES = {
    "rules_only": "Keyword rules only",
    "classifier_only": "Classifier only (TF-IDF + logistic regression)",
    "llm_all": "Claude Haiku for everything (simulated)",
    "cascade": "Cascade, chosen point (LLM tier simulated)",
}


def table(result: dict) -> str:
    lines = [
        "| Policy | Micro F1 | Macro F1 | Exact match | Aspect sentiment acc | Cost per 1k texts | Sent to an LLM |",
        "|---|---|---|---|---|---|---|",
    ]
    for key, name in ROW_NAMES.items():
        m = result["rows"][key]
        lines.append(
            f"| {name} | {m['micro_f1']:.3f} | {m['macro_f1']:.3f} | {m['exact_match']:.1%} | "
            f"{m['sentiment']['aspect_accuracy']:.1%} | ${m['cost_per_1k']:.4f} | {m['llm_share']:.0%} |"
        )
    return "\n".join(lines)


def gate(result: dict, floor: float = DEFAULT_F1_FLOOR) -> list[str]:
    m = result["rows"]["cascade"]
    return [] if m["micro_f1"] >= floor else [f"cascade micro F1 {m['micro_f1']:.3f} is below the floor {floor:.3f}"]
