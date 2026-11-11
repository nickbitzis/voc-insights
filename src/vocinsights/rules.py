"""Tier 1: keyword rules. High precision patterns per theme, with a confidence from clause coverage.

The text is split into clauses. If every clause matches exactly one theme pattern, the rules have
explained the whole comment and the tier is confident. Any clause with no match, or with two
competing matches, lowers confidence and the text moves on to the classifier.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .taxonomy import THEMES

PATTERNS = {t.name: re.compile(t.rule, re.I) for t in THEMES}
CLAUSE_SPLIT = re.compile(r"(?<=[.!?])\s+|,\s*but\s+|\s+but\s+", re.I)


def split_clauses(text: str) -> list[str]:
    return [c.strip() for c in CLAUSE_SPLIT.split(text) if c and len(c.strip()) > 1]


@dataclass
class RuleResult:
    themes: list[str]
    clause_hits: list[list[str]]  # per clause, matched themes
    confidence: float


def match(text: str) -> RuleResult:
    clauses = split_clauses(text)
    hits = [[name for name, p in PATTERNS.items() if p.search(c)] for c in clauses]
    themes = sorted({h for hs in hits for h in hs})
    if not clauses or not themes:
        return RuleResult([], hits, 0.0)
    clean = sum(1 for hs in hits if len(hs) == 1)
    coverage = clean / len(clauses)
    conf = 0.92 if coverage == 1.0 else round(0.6 * coverage, 3)
    return RuleResult(themes, hits, conf)
