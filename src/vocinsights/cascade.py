"""The theme classification cascade: keyword rules, then the classifier, then the LLM tagger.

A text stops at the first tier that is confident enough. Rules and classifier decisions get their
per-theme sentiment from the aspect sentiment model; the LLM tier returns its own.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from . import rules
from .classifier import ThemeClassifier
from .data import overall_of
from .llm import OTHER, StructuredOutputError, Tagger
from .pricing import LOCAL_COST_PER_TEXT
from .sentiment import AspectSentiment, clause_score, label


@dataclass(frozen=True)
class Policy:
    t_rules: float | None = 0.9  # None turns the rules tier off
    t_clf: float = 0.6
    use_llm: bool = True

    def label(self) -> str:
        r = "off" if self.t_rules is None else f"{self.t_rules:.2f}"
        return f"rules {r}, classifier {self.t_clf:.2f}, llm {'on' if self.use_llm else 'off'}"


@dataclass
class Decision:
    themes: list[str]
    sentiments: dict[str, str]
    overall: str
    tier: str
    confidence: float
    evidence: dict[str, str]
    cost: float
    simulated: bool = False
    new_issue: str | None = None  # LLM said "other": evidence for a theme outside the taxonomy
    llm_error: str | None = None
    probs: dict[str, float] = field(default_factory=dict, repr=False)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["cost"] = round(self.cost, 8)
        d["confidence"] = round(self.confidence, 4)
        d["probs"] = {k: round(v, 3) for k, v in sorted(self.probs.items(), key=lambda kv: -kv[1])[:5]}
        return d


class Cascade:
    def __init__(self, clf: ThemeClassifier, sentiment: AspectSentiment, llm: Tagger | None, policy: Policy):
        self.clf = clf
        self.sentiment = sentiment
        self.llm = llm
        self.policy = policy

    def _local(self, text: str, themes: list[str], tier: str, conf: float, probs: dict) -> Decision:
        sents = self.sentiment.predict(text, themes)
        spans = self.sentiment.clauses_for(text, themes)
        return Decision(
            themes=sorted(themes),
            sentiments=sents,
            overall=overall_of(sents) if themes else label(clause_score(text)),
            tier=tier,
            confidence=float(conf),
            evidence={t: spans[t][0] for t in themes},
            cost=LOCAL_COST_PER_TEXT,
            probs=probs,
        )

    def run(self, texts: list[str], golds=None, probs: np.ndarray | None = None) -> list[Decision]:
        if probs is None:
            probs = self.clf.proba(texts)
        y, conf = ThemeClassifier.decide(probs)
        names = self.clf.names
        out = []
        for i, text in enumerate(texts):
            p = {names[j]: float(probs[i, j]) for j in range(len(names)) if probs[i, j] >= 0.05}
            clf_themes = [names[j] for j in np.flatnonzero(y[i])]
            r = rules.match(text)
            if self.policy.t_rules is not None and r.confidence >= self.policy.t_rules:
                out.append(self._local(text, r.themes, "rules", r.confidence, p))
                continue
            if conf[i] >= self.policy.t_clf or not self.policy.use_llm or self.llm is None:
                out.append(self._local(text, clf_themes, "classifier", conf[i], p))
                continue
            gold = golds[i] if golds is not None else None
            hint = self.sentiment.predict(text, clf_themes)
            try:
                res = self.llm.tag(text, gold=gold, hint=hint)
            except StructuredOutputError as exc:
                d = self._local(text, clf_themes, "classifier", conf[i], p)
                d.llm_error = str(exc)[:200]
                out.append(d)
                continue
            themes = res.labels
            sents = res.sentiments
            ev = {t["theme"]: t["evidence"] for t in res.themes if t["theme"] != OTHER}
            other = next((t["evidence"] for t in res.themes if t["theme"] == OTHER), None)
            out.append(
                Decision(
                    themes=sorted(themes),
                    sentiments=sents,
                    overall=res.overall,
                    tier="llm",
                    confidence=float(conf[i]),
                    evidence={t: ev.get(t, text[:80]) for t in themes},
                    cost=LOCAL_COST_PER_TEXT + res.cost,
                    simulated=res.simulated,
                    new_issue=other,
                    probs=p,
                )
            )
        return out
