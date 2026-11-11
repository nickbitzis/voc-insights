"""Tier 3: an LLM tagger for the texts the cheaper tiers are unsure about.

`ClaudeTagger` calls Claude Haiku with a JSON schema in `output_config.format`, so the reply is an
object with a list of themes, a sentiment for each, and a quoted evidence span. The validator then
checks that every evidence span appears verbatim in the text, so a made-up quote is rejected. The
`anthropic` package is imported only when a ClaudeTagger is built without an injected client.

`SimulatedTagger` is the offline default. It does not call any model. Its recall, precision,
sentiment accuracy, and token counts come from `SimConfig`, so the cascade economics can be studied
for free. Every number it produces is simulated and is labeled that way in reports and the UI.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass
from typing import Protocol

from .pricing import call_cost
from .rules import split_clauses
from .taxonomy import BY_NAME, SENTIMENTS, THEME_NAMES

OTHER = "other"
OVERALL = ("negative", "neutral", "positive", "mixed")
MAX_THEMES = 5

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "themes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "theme": {"type": "string", "enum": [*THEME_NAMES, OTHER]},
                    "sentiment": {"type": "string", "enum": list(SENTIMENTS)},
                    "evidence": {"type": "string"},
                },
                "required": ["theme", "sentiment", "evidence"],
                "additionalProperties": False,
            },
        },
        "overall": {"type": "string", "enum": list(OVERALL)},
    },
    "required": ["themes", "overall"],
    "additionalProperties": False,
}


class StructuredOutputError(ValueError):
    """The model reply did not match the schema or quoted text that is not in the input."""


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def validate_output(obj: object, text: str) -> dict:
    """Check a parsed reply against OUTPUT_SCHEMA and the source text, without a schema library."""
    if not isinstance(obj, dict):
        raise StructuredOutputError("reply is not a JSON object")
    missing = [k for k in OUTPUT_SCHEMA["required"] if k not in obj]
    if missing:
        raise StructuredOutputError(f"missing fields: {missing}")
    extra = set(obj) - set(OUTPUT_SCHEMA["properties"])
    if extra:
        raise StructuredOutputError(f"unexpected fields: {sorted(extra)}")
    if obj["overall"] not in OVERALL:
        raise StructuredOutputError(f"overall={obj['overall']!r} is not allowed")
    themes = obj["themes"]
    if not isinstance(themes, list) or len(themes) > MAX_THEMES:
        raise StructuredOutputError(f"themes must be a list of at most {MAX_THEMES} items")
    source = _norm(text)
    seen = set()
    clean = []
    for t in themes:
        if not isinstance(t, dict) or set(t) != {"theme", "sentiment", "evidence"}:
            raise StructuredOutputError("each theme needs exactly theme, sentiment, evidence")
        if t["theme"] not in (*THEME_NAMES, OTHER):
            raise StructuredOutputError(f"theme={t['theme']!r} is not in the taxonomy")
        if t["sentiment"] not in SENTIMENTS:
            raise StructuredOutputError(f"sentiment={t['sentiment']!r} is not allowed")
        ev = t["evidence"]
        if not isinstance(ev, str) or len(_norm(ev)) < 3:
            raise StructuredOutputError("evidence must be a quoted span of at least 3 characters")
        if _norm(ev) not in source:
            raise StructuredOutputError(f"evidence {ev[:40]!r} is not a verbatim span of the text")
        if t["theme"] in seen and t["theme"] != OTHER:
            raise StructuredOutputError(f"duplicate theme {t['theme']!r}")
        seen.add(t["theme"])
        clean.append({"theme": t["theme"], "sentiment": t["sentiment"], "evidence": ev.strip()})
    return {"themes": clean, "overall": obj["overall"]}


@dataclass
class TagResult:
    themes: list[dict]  # [{"theme", "sentiment", "evidence"}]
    overall: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    simulated: bool

    @property
    def cost(self) -> float:
        return call_cost(self.model, self.input_tokens, self.output_tokens)

    @property
    def labels(self) -> list[str]:
        return [t["theme"] for t in self.themes if t["theme"] != OTHER]

    @property
    def sentiments(self) -> dict[str, str]:
        return {t["theme"]: t["sentiment"] for t in self.themes if t["theme"] != OTHER}

    def to_dict(self) -> dict:
        return asdict(self) | {"cost": round(self.cost, 8)}


def system_prompt() -> str:
    lines = [
        "You tag customer feedback for Fernhollow Planner, a family calendar and lists app.",
        "Return every theme the text mentions, from the list below. For each one give the customer's sentiment about "
        "that theme and copy a short evidence span word for word from the text. Read sarcasm for what it means: "
        "'great, another notification at 3am' is negative. Feature requests with no complaint are neutral.",
        "If the text describes a problem that fits no theme, add one item with theme 'other'. A comment with only a "
        "general feeling, such as 'Love it!', gets an empty theme list.",
        "",
        "Themes:",
    ]
    lines += [f"- {name}: {BY_NAME[name].label}" for name in THEME_NAMES]
    return "\n".join(lines)


SYSTEM_PROMPT = system_prompt()
SYSTEM_TOKENS = len(SYSTEM_PROMPT) // 4 + 80  # schema and framing overhead


class Tagger(Protocol):
    model: str

    def tag(self, text: str, gold=None, hint: dict | None = None) -> TagResult: ...


class ClaudeTagger:
    """Real Claude call. Inject `client` in tests; otherwise the SDK is imported here."""

    def __init__(self, model: str = "claude-haiku-5-5", client=None, max_tokens: int = 500):
        if client is None:
            try:
                import anthropic
            except ImportError as exc:  # pragma: no cover - depends on install extras
                raise RuntimeError("Install the Claude extra first: uv sync --extra claude") from exc
            client = anthropic.Anthropic()
        self.client = client
        self.model = model
        self.max_tokens = max_tokens

    def request(self, text: str) -> dict:
        return {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": f"<feedback>\n{text}\n</feedback>"}],
            "output_config": {"format": {"type": "json_schema", "schema": OUTPUT_SCHEMA}},
        }

    def tag(self, text: str, gold=None, hint: dict | None = None) -> TagResult:
        start = time.perf_counter()
        resp = self.client.messages.create(**self.request(text))
        latency = (time.perf_counter() - start) * 1000
        if getattr(resp, "stop_reason", None) == "refusal":
            raise StructuredOutputError(f"{self.model} declined to tag this text")
        raw = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise StructuredOutputError(f"reply is not valid JSON: {raw[:80]!r}") from exc
        out = validate_output(parsed, text)
        return TagResult(
            themes=out["themes"],
            overall=out["overall"],
            model=self.model,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            latency_ms=round(latency, 1),
            simulated=False,
        )


@dataclass(frozen=True)
class SimConfig:
    """Knobs for the simulated tagger. All values are assumptions, not measurements."""

    model: str = "claude-haiku-5-5"
    recall: float = 0.95  # chance each true theme is found
    false_theme: float = 0.04  # chance of one extra wrong theme per text
    sentiment_acc: float = 0.94
    sarcasm_acc: float = 0.88
    latency_ms: float = 450
    out_tokens_per_theme: int = 38


def _u(key: str, salt: str) -> float:
    h = hashlib.sha256(f"{key}|{salt}".encode()).digest()
    return int.from_bytes(h[:8], "big") / 2**64


def best_span(text: str, phrase: str) -> str:
    """The clause of `text` with the most word overlap with `phrase`. Always a verbatim span."""
    words = set(re.findall(r"[a-z]+", phrase.lower()))
    clauses = split_clauses(text) or [text]
    return max(clauses, key=lambda c: len(words & set(re.findall(r"[a-z]+", c.lower()))))


class SimulatedTagger:
    def __init__(self, config: SimConfig | None = None):
        self.config = config or SimConfig()
        self.model = self.config.model

    def tag(self, text: str, gold=None, hint: dict | None = None) -> TagResult:
        """With a labeled text (`gold`), each theme is found with the configured recall. With free text
        there is no ground truth, so it returns `hint` (the best local guess). That still shows routing,
        cost, and latency, not LLM quality."""
        c = self.config
        key = f"{c.model}|{getattr(gold, 'id', '')}|{text}"
        items: list[dict] = []
        if gold is None:
            for theme, s in (hint or {}).items():
                items.append({"theme": theme, "sentiment": s, "evidence": best_span(text, theme.replace("_", " "))})
        else:
            phrases = {name: phrase for name, _, phrase in gold.clauses}
            for theme, s in gold.sentiments.items():
                if theme not in THEME_NAMES:
                    items.append({"theme": OTHER, "sentiment": s, "evidence": best_span(text, phrases.get(theme, ""))})
                    continue
                if _u(key, "miss" + theme) > c.recall:
                    continue
                acc = c.sarcasm_acc if gold.sarcastic else c.sentiment_acc
                if _u(key, "sent" + theme) > acc:
                    s = SENTIMENTS[(SENTIMENTS.index(s) + 1 + int(_u(key, "flip" + theme) * 2)) % 3]
                items.append({"theme": theme, "sentiment": s, "evidence": best_span(text, phrases.get(theme, theme))})
            if _u(key, "extra") < c.false_theme:
                pool = [t for t in THEME_NAMES if t not in gold.sentiments]
                extra = pool[int(_u(key, "pick") * len(pool))]
                items.append({"theme": extra, "sentiment": "negative", "evidence": best_span(text, extra)})
        sents = {i["sentiment"] for i in items}
        if not items and gold is not None:
            sents = {gold.overall}
        overall = (
            "mixed"
            if {"positive", "negative"} <= sents
            else next((s for s in ("negative", "positive") if s in sents), "neutral")
        )
        out_tokens = 20 + c.out_tokens_per_theme * len(items)
        in_tokens = SYSTEM_TOKENS + max(1, len(text) // 4)
        return TagResult(
            themes=items,
            overall=overall,
            model=c.model,
            input_tokens=in_tokens,
            output_tokens=out_tokens,
            latency_ms=round(c.latency_ms * (0.8 + 0.4 * _u(key, "lat")) + 6 * out_tokens, 1),
            simulated=True,
        )
