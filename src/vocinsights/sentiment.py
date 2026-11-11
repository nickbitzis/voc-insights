"""Overall and per-theme (aspect based) sentiment.

Each theme is tied to the clauses that mention it (the clause a rule matched, or else the clauses no
rule explained). Those clauses are scored with a small lexicon that handles negation ("no crashes"
is positive) and common sarcasm frames ("great, another notification at 3am"). Requests such as
"please add a dark mode" are neutral. When a clause has no sentiment words, the theme's most common
sentiment in the labeled training set is used.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict

from .data import overall_of
from .rules import PATTERNS, split_clauses

POS = set(
    "love loved lovely great easy simple intuitive beautiful gorgeous clean pretty nice nicest helpful kind fast quick "
    "quickly instantly instant snappy responsive worth good finally organised organized calm best smooth "
    "straightforward exactly fixed shout happy awesome perfect excellent".split()
)
NEG = set(
    "cant cannot can't fail failing fails failed slow forever hours ages stuck expensive dealbreaker spam spammy nuts "
    "crash crashes crashing crashed freezes froze drains drain eats tanks useless broken broke nothing disappeared "
    "lost vanished gone wiped confusing skips sluggish laggy charged twice refund wrong waited loops hurts blank "
    "stopped quitting closes ugly cluttered harder spins wipes justify cancelling cancel lot much costs annoying "
    "bad terrible worst duplicate".split()
)
NEGATORS = {
    "no",
    "not",
    "never",
    "dont",
    "don't",
    "didnt",
    "didn't",
    "doesnt",
    "doesn't",
    "isnt",
    "isn't",
    "wont",
    "won't",
    "without",
}
REQUEST = re.compile(
    r"\b(please add|would be nice|would love|any plans|is there a way|does it have|wish)\b|\?\s*$", re.I
)
SARCASM = re.compile(
    r"\b(love|great|nice|thanks|favorite|very helpful)\b.*\b(another|again|every time|five times|half a day|"
    r"if you like|right when|deleting|about nothing|paying more|every year)\b",
    re.I,
)
TOKEN = re.compile(r"[a-z']+")


def clause_score(clause: str) -> float:
    """Positive above zero, negative below. Zero means no sentiment words."""
    if SARCASM.search(clause):
        return -2.0
    toks = TOKEN.findall(clause.lower())
    score = 0.0
    i = 0
    while i < len(toks):
        t = toks[i]
        if t in NEGATORS:
            window = toks[i + 1 : i + 3]
            if any(w in NEG for w in window):
                score += 1.0  # "no crashes"
                i += 1 + next(j for j, w in enumerate(window) if w in NEG) + 1
                continue
            if any(w in POS for w in window):
                score -= 1.0  # "not easy"
                i += 1 + next(j for j, w in enumerate(window) if w in POS) + 1
                continue
            score -= 0.75  # "never replied", "doesn't work"
        elif t in POS:
            score += 1.0
        elif t in NEG:
            score -= 1.0
        i += 1
    return score


def label(score: float) -> str:
    return "positive" if score > 0 else "negative" if score < 0 else "neutral"


class AspectSentiment:
    def __init__(self):
        self.prior: dict[str, str] = {}

    def fit(self, items) -> AspectSentiment:
        counts: dict[str, Counter] = defaultdict(Counter)
        for f in items:
            for theme, s in f.sentiments.items():
                counts[theme][s] += 1
        self.prior = {t: c.most_common(1)[0][0] for t, c in counts.items()}
        return self

    def clauses_for(self, text: str, themes) -> dict[str, list[str]]:
        clauses = split_clauses(text) or [text]
        hits = [[n for n in themes if n in PATTERNS and PATTERNS[n].search(c)] for c in clauses]
        unexplained = [c for c, h in zip(clauses, hits, strict=True) if not h]
        out = {}
        for theme in themes:
            mine = [c for c, h in zip(clauses, hits, strict=True) if theme in h]
            out[theme] = mine or unexplained or clauses
        return out

    def theme_sentiment(self, theme: str, clauses: list[str]) -> str:
        if all(REQUEST.search(c) for c in clauses) and not any(SARCASM.search(c) for c in clauses):
            score = sum(clause_score(c) for c in clauses)
            return "neutral" if score > -1.5 else "negative"
        score = sum(clause_score(c) for c in clauses)
        if score == 0:
            return self.prior.get(theme, "neutral")
        return label(score)

    def predict(self, text: str, themes) -> dict[str, str]:
        return {t: self.theme_sentiment(t, cl) for t, cl in self.clauses_for(text, themes).items()}

    def overall(self, text: str, themes) -> str:
        return overall_of(self.predict(text, themes))


def evaluate_aspects(items, predicted: list[dict[str, str]]) -> dict:
    """Aspect accuracy on themes found in both gold and prediction, and overall sentiment accuracy."""
    right = total = 0
    by_class: dict[str, list[int]] = {s: [0, 0, 0] for s in ("negative", "neutral", "positive")}  # tp, fp, fn
    overall_right = 0
    for f, pred in zip(items, predicted, strict=True):
        for theme, s in pred.items():
            g = f.sentiments.get(theme)
            if g is None:
                continue
            total += 1
            right += int(g == s)
            if g == s:
                by_class[s][0] += 1
            else:
                by_class[s][1] += 1
                by_class[g][2] += 1
        overall_right += int(overall_of(pred) == f.overall)
    f1s = {}
    for s, (tp, fp, fn) in by_class.items():
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1s[s] = round(2 * p * r / (p + r), 4) if p + r else 0.0
    return {
        "aspect_accuracy": round(right / total, 4) if total else 0.0,
        "aspect_pairs": total,
        "aspect_f1": f1s,
        "aspect_macro_f1": round(sum(f1s.values()) / len(f1s), 4),
        "overall_accuracy": round(overall_right / len(items), 4) if items else 0.0,
    }
