"""Seeded synthetic feedback: NPS comments, app store reviews, and support ticket summaries.

Each text is built from one to three theme clauses picked from the hidden taxonomy, with paraphrase
slots, sarcasm, typos, and source-specific framing. Ground truth themes and per-theme sentiment are
kept for every text. A random subset is marked `labeled`: that is what a team would have hand
labeled, and it is the only part the supervised classifier trains on.

Two events are planted:
- A price increase on 2025-07-01 that makes pricing complaints about 1.9 times as common.
- Release 6.0 on 2025-10-20 ships a home screen widget bug that crashes the app. Hotfix 6.0.2 lands
  on 2025-11-17 and the complaints fade.
"""

from __future__ import annotations

import datetime as dt
import math
import random
import re
from dataclasses import dataclass, field

from .taxonomy import GENERAL, NEW_ISSUE, SLOTS, THEME_NAMES, THEMES, VAGUE, Theme

YEAR = 2025
START = dt.date(YEAR, 1, 1)
DAYS = 365
PRICE_CHANGE = dt.date(YEAR, 7, 1)
RELEASE = dt.date(YEAR, 10, 20)
HOTFIX = dt.date(YEAR, 11, 17)
QUARTERS = ("2025Q1", "2025Q2", "2025Q3", "2025Q4")
SOURCE_MIX = {"nps": 0.4, "review": 0.35, "ticket": 0.25}
BASE_SCORE = 8.6
SARCASM_RATE = 0.15
VAGUE_RATE = 0.3  # share of theme mentions phrased without the obvious keywords
GENERAL_RATE = 0.05  # comments with no theme
MISS_RATE = 0.06  # annotators drop a theme from a multi-theme text
EXTRA_RATE = 0.03  # annotators add a wrong theme

TICKET_PREFIX = (
    "Customer reports:",
    "User says",
    "Chat summary:",
    "Email from user:",
    "Caller said",
    "Ticket: customer writes that",
)
REVIEW_OPENERS = ("", "", "", "Update:", "Honest review.", "Used it for a year.", "Ok so", "Edit:")
SYNONYMS = {
    "app": ("application", "thing", "tool", "planner"),
    "calendar": ("schedule", "planner", "cal", "agenda"),
    "notes": ("memos", "entries", "stuff"),
    "phone": ("iphone", "cell", "mobile"),
    "really": ("rly", "truly", "seriously"),
    "very": ("v", "really", "extremely"),
    "every": ("each",),
    "forever": ("ages", "an eternity"),
    "great": ("awesome", "amazing", "fantastic"),
    "love": ("adore", "really like", "luv"),
    "slow": ("sluggish", "painfully slow", "glacial"),
    "keeps": ("constantly", "always"),
    "events": ("appointments", "plans", "entries"),
    "too": ("way too", "far too"),
}
SYNONYM_RATE = 0.35
FILLERS = (
    "Otherwise fine.",
    "That's all.",
    "Thanks.",
    "Please look into it.",
    "Just my two cents.",
    "Been using it since the start.",
    "",
)


@dataclass
class Feedback:
    id: str
    source: str
    date: str
    quarter: str
    week: str  # Monday of the week, ISO date
    text: str
    themes: tuple[str, ...]
    sentiments: dict[str, str]
    overall: str
    score: int | None  # NPS 0 to 10, or stars 1 to 5, or None for tickets
    sarcastic: bool = False
    labeled: bool = False
    labels: tuple[str, ...] = ()  # what the annotator wrote down; differs from `themes` by label noise
    clauses: list[tuple[str, str, str]] = field(default_factory=list, repr=False)  # (theme, sentiment, text)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "source": self.source,
            "date": self.date,
            "quarter": self.quarter,
            "week": self.week,
            "text": self.text,
            "score": self.score,
        }


def quarter_of(d: dt.date) -> str:
    return f"{d.year}Q{(d.month - 1) // 3 + 1}"


def week_of(d: dt.date) -> str:
    return (d - dt.timedelta(days=d.weekday())).isoformat()


def bug_rate(d: dt.date) -> float:
    """Share of feedback on a given day that is about the release 6.0 widget crash."""
    if d < RELEASE:
        return 0.0
    days = (d - RELEASE).days
    if d < HOTFIX:
        return min(0.11, 0.035 + 0.006 * days)  # ramps as the release rolls out to all users
    return 0.11 * math.exp(-(d - HOTFIX).days / 9)


def _fill(rng: random.Random, phrase: str) -> str:
    return re.sub(r"\{(\w+)\}", lambda m: rng.choice(SLOTS[m.group(1)]), phrase)


def _typo(rng: random.Random, text: str) -> str:
    words = text.split(" ")
    for _ in range(rng.randint(1, 3)):
        i = rng.randrange(len(words))
        w = words[i]
        if len(w) < 4 or not w.isalpha():
            continue
        j = rng.randrange(1, len(w) - 1)
        kind = rng.random()
        if kind < 0.4:
            w = w[:j] + w[j + 1] + w[j] + w[j + 2 :]
        elif kind < 0.75:
            w = w[:j] + w[j + 1 :]
        else:
            w = w[:j] + w[j] + w[j:]
        words[i] = w
    return " ".join(words)


def _paraphrase(rng: random.Random, text: str) -> str:
    words = text.split(" ")
    for i, w in enumerate(words):
        alts = SYNONYMS.get(w.lower())
        if alts and rng.random() < SYNONYM_RATE:
            words[i] = rng.choice(alts)
    return " ".join(words)


def _sentence(s: str) -> str:
    s = s.strip()
    s = s[0].upper() + s[1:]
    return s if s[-1] in ".!?" else s + rng_end(s)


def rng_end(s: str) -> str:
    return "?" if s.lower().startswith(("is there", "does it", "any plans")) else "."


def _pick_sentiment(rng: random.Random, t: Theme) -> str:
    u = rng.random()
    if u < t.p_pos:
        return "positive"
    if u < t.p_pos + (1 - t.p_pos) * t.p_neu:
        return "neutral"
    return "negative"


def _theme_weights(source: str, d: dt.date) -> list[float]:
    out = []
    for t in THEMES:
        w = t.weight * t.affinity[source]
        if t.name == "pricing_too_high" and d >= PRICE_CHANGE:
            w *= 1.9
        out.append(w)
    return out


def overall_of(sentiments: dict[str, str]) -> str:
    vals = set(sentiments.values())
    if "positive" in vals and "negative" in vals:
        return "mixed"
    if "negative" in vals:
        return "negative"
    if "positive" in vals:
        return "positive"
    return "neutral"


def _score(rng: random.Random, source: str, sentiments: dict[str, str], effects: dict) -> int | None:
    latent = BASE_SCORE + rng.gauss(0, 1.2)
    for name, s in sentiments.items():
        if s == "negative":
            latent += effects[name][0]
        elif s == "positive":
            latent += effects[name][1]
    if source == "nps":
        return int(min(10, max(0, round(latent))))
    if source == "review":
        return int(min(5, max(1, round(1 + latent * 0.42))))
    return None


def generate(n: int = 25_000, seed: int = 7, labeled_share: float = 0.1) -> list[Feedback]:
    rng = random.Random(seed)
    effects = {t.name: (t.effect_neg, t.effect_pos) for t in (*THEMES, NEW_ISSUE)}
    # Volume grows about 20% over the year with a weekend dip.
    day_w = [
        (1 + 0.2 * i / DAYS) * (0.8 if (START + dt.timedelta(days=i)).weekday() >= 5 else 1.0) for i in range(DAYS)
    ]
    days = sorted(rng.choices(range(DAYS), weights=day_w, k=n))
    sources = list(SOURCE_MIX)
    out: list[Feedback] = []
    for i, day in enumerate(days):
        d = START + dt.timedelta(days=day)
        source = rng.choices(sources, weights=[SOURCE_MIX[s] for s in sources])[0]
        clauses: list[tuple[str, str, str]] = []
        sarcastic = False
        is_bug = rng.random() < bug_rate(d) * {"nps": 0.6, "review": 1.3, "ticket": 1.3}[source]
        if is_bug:
            clauses.append((NEW_ISSUE.name, "negative", _fill(rng, rng.choice(NEW_ISSUE.neg))))
        k = rng.choices((1, 2, 3), weights=(0.55, 0.33, 0.12))[0] - int(is_bug)
        names = [t.name for t in THEMES]
        weights = _theme_weights(source, d)
        chosen: list[Theme] = []
        while len(chosen) < k:
            t = THEMES[names.index(rng.choices(names, weights=weights)[0])]
            if t not in chosen:
                chosen.append(t)
        general = None
        if not is_bug and rng.random() < GENERAL_RATE:
            general, chosen = rng.choice(GENERAL), []
        for t in chosen:
            s = _pick_sentiment(rng, t)
            vague = [v for v in VAGUE.get(t.name, ()) if v[0] == {"positive": "+", "negative": "-", "neutral": "="}[s]]
            if vague and rng.random() < VAGUE_RATE:
                clauses.append((t.name, s, _fill(rng, rng.choice(vague)[1:])))
                continue
            if s == "negative" and t.sarcasm and rng.random() < SARCASM_RATE:
                phrase, sarcastic = rng.choice(t.sarcasm), True
            elif s == "positive" and t.pos:
                phrase = rng.choice(t.pos)
            elif s == "neutral" and t.neu:
                phrase = rng.choice(t.neu)
            else:
                s, phrase = "negative", rng.choice(t.neg)
            clauses.append((t.name, s, _fill(rng, phrase)))
        sentiments = {name: s for name, s, _ in clauses}
        if is_bug:
            sentiments["app_stability"] = "negative"  # a crash is also a stability complaint
        # Assemble: two clauses with opposite sentiment are often joined with "but".
        parts = [_sentence(c[2]) for c in clauses]
        if len(clauses) == 2 and {clauses[0][1], clauses[1][1]} == {"positive", "negative"} and rng.random() < 0.5:
            parts = [_sentence(clauses[0][2])[:-1] + ", but " + clauses[1][2] + "."]
        text = " ".join(parts) if parts else general[1]
        if source == "ticket":
            text = f"{rng.choice(TICKET_PREFIX)} {text[0].lower()}{text[1:]}"
        elif source == "review":
            opener = rng.choice(REVIEW_OPENERS)
            text = f"{opener} {text}".strip()
        if rng.random() < 0.3:
            filler = rng.choice(FILLERS)
            text = f"{text} {filler}".strip()
        text = _paraphrase(rng, text)
        if rng.random() < 0.3:
            text = _typo(rng, text)
        if rng.random() < 0.25:
            text = text.replace("'", "")
        if rng.random() < 0.15:
            text = text.lower()
        themes = tuple(sorted(sentiments))
        out.append(
            Feedback(
                id=f"F{i + 1:06d}",
                source=source,
                date=d.isoformat(),
                quarter=quarter_of(d),
                week=week_of(d),
                text=text,
                themes=themes,
                sentiments=sentiments,
                overall=general[0] if general else overall_of(sentiments),
                score=_score(rng, source, sentiments, effects),
                sarcastic=sarcastic,
                clauses=clauses,
            )
        )
    n_labeled = min(n, max(2000, round(labeled_share * n)))
    lrng = random.Random(seed + 1)
    for j in lrng.sample(range(n), n_labeled):
        f = out[j]
        f.labeled = True
        labels = [t for t in f.themes if t in THEME_NAMES]
        if len(labels) > 1 and lrng.random() < MISS_RATE * len(labels):
            labels.remove(lrng.choice(labels))
        if lrng.random() < EXTRA_RATE:
            labels.append(lrng.choice([t for t in THEME_NAMES if t not in labels]))
        f.labels = tuple(sorted(labels))
    return out


def split_labeled(items: list[Feedback], seed: int = 7, test_share: float = 0.3) -> tuple[list, list]:
    """Train and test split of the labeled subset."""
    lab = [f for f in items if f.labeled]
    rng = random.Random(seed + 2)
    rng.shuffle(lab)
    cut = int(len(lab) * (1 - test_share))
    return lab[:cut], lab[cut:]


def weeks() -> list[str]:
    return sorted({week_of(START + dt.timedelta(days=i)) for i in range(DAYS)})
