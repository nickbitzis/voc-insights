"""Quarterly insights report in Markdown and HTML.

All numbers come from `facts()`. The summary at the top is written by a writer: the offline
`TemplateWriter` (default) or `ClaudeWriter`. Whatever the writer returns goes through `validate()`,
which rejects any number that is not in the facts and any quoted text that does not appear verbatim
in the feedback. A rejected Claude summary falls back to the template, and the report says so.
"""

from __future__ import annotations

import html
import json
import random
import re
from collections import Counter, defaultdict

from . import data
from .drivers import nps
from .taxonomy import LABELS, THEME_NAMES

NUM = re.compile(r"(?<![\w.])[-+]?\d[\d,]*(?:\.\d+)?%?")
QUOTE = re.compile(r"[\"“]([^\"”]{12,400})[\"”]")


class ReportValidationError(ValueError):
    pass


def _share(n: int, d: int) -> float:
    return round(n / d, 4) if d else 0.0


def _quotes(rng: random.Random, pool: list[str], k: int = 2) -> list[str]:
    good = [t for t in pool if 40 <= len(t) <= 170] or pool
    return rng.sample(good, min(k, len(good)))


def facts(bench, quarter: str | None = None, seed: int = 7) -> dict:
    items, decisions = bench.items, bench.decisions
    quarter = quarter or data.QUARTERS[-1]
    qi = data.QUARTERS.index(quarter)
    prev = data.QUARTERS[qi - 1] if qi > 0 else None
    per_q: dict[str, dict] = defaultdict(
        lambda: {"n": 0, "count": Counter(), "neg": Counter(), "scores": [], "stars": [], "sources": Counter()}
    )
    texts_by: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    for f, d in zip(items, decisions, strict=True):
        q = per_q[f.quarter]
        q["n"] += 1
        q["sources"][f.source] += 1
        if f.source == "nps":
            q["scores"].append(f.score)
        elif f.source == "review":
            q["stars"].append(f.score)
        for t in d.themes:
            q["count"][t] += 1
            s = d.sentiments.get(t, "neutral")
            q["neg"][t] += int(s == "negative")
            texts_by[(f.quarter, t, s)].append(f.text)
    cur, old = per_q[quarter], per_q[prev] if prev else None
    themes = []
    for t in THEME_NAMES:
        c = cur["count"][t]
        row = {
            "theme": t,
            "label": LABELS[t],
            "count": c,
            "share": _share(c, cur["n"]),
            "negative_share": _share(cur["neg"][t], c),
        }
        if old:
            row["prev_share"] = _share(old["count"][t], old["n"])
            row["change_pp"] = round(100 * (row["share"] - row["prev_share"]), 1)
        themes.append(row)
    themes.sort(key=lambda r: -r["count"])
    rng = random.Random(seed)
    movers = sorted((r for r in themes if "change_pp" in r), key=lambda r: -abs(r["change_pp"]))[:4]
    quote_themes = [r["theme"] for r in themes[:5]] + [
        m["theme"] for m in movers if m["theme"] not in [r["theme"] for r in themes[:5]]
    ]
    quotes = {}
    for t in quote_themes:
        row = next(r for r in themes if r["theme"] == t)
        main = "negative" if row["negative_share"] >= 0.5 else "positive"
        pool = texts_by[(quarter, t, main)] or texts_by[(quarter, t, "neutral")]
        quotes[t] = _quotes(rng, pool)
    q_start, q_end = _quarter_bounds(quarter)
    alerts = [a for a in bench.alerts if q_start <= a["first_day"] <= q_end]
    clusters = [
        {k: v for k, v in c.items() if k != "doc_ids"} for c in bench.clusters if q_start <= c["first_day"] <= q_end
    ]
    drv = bench.drivers["drivers"]
    return {
        "company": "Fernhollow",
        "product": "Fernhollow Planner",
        "quarter": quarter,
        "previous_quarter": prev,
        "volume": cur["n"],
        "previous_volume": old["n"] if old else None,
        "sources": dict(cur["sources"]),
        "nps": nps(cur["scores"]),
        "previous_nps": nps(old["scores"]) if old else None,
        "nps_responses": len(cur["scores"]),
        "avg_stars": round(sum(cur["stars"]) / len(cur["stars"]), 2) if cur["stars"] else None,
        "previous_avg_stars": round(sum(old["stars"]) / len(old["stars"]), 2) if old and old["stars"] else None,
        "themes": themes,
        "movers": movers,
        "quotes": quotes,
        "alerts": alerts,
        "new_clusters": clusters,
        "drivers_negative": drv[:3],
        "drivers_positive": drv[::-1][:2],
        "drivers_n": bench.drivers["n"],
        "caveats": bench.drivers["caveats"],
        "release": data.RELEASE.isoformat(),
        "hotfix": data.HOTFIX.isoformat(),
        "simulated_llm_share": bench.evaluation["rows"]["cascade"]["llm_share"],
    }


def _quarter_bounds(quarter: str) -> tuple[str, str]:
    y, q = int(quarter[:4]), int(quarter[-1])
    start = data.dt.date(y, 3 * q - 2, 1)
    end = data.dt.date(y + (q == 4), 1 if q == 4 else 3 * q + 1, 1) - data.dt.timedelta(days=1)
    return start.isoformat(), end.isoformat()


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def _signed(x: float, unit: str = "") -> str:
    return f"{x:+.1f}{unit}"


class TemplateWriter:
    name = "template"
    simulated = False

    def write(self, f: dict) -> str:
        lines = []
        top = f["themes"][0]
        if f["previous_nps"] is not None:
            lines.append(
                f"- NPS was {f['nps']:.1f} in {f['quarter']} against {f['previous_nps']:.1f} in "
                f"{f['previous_quarter']}, from {f['nps_responses']:,} survey responses."
            )
        lines.append(
            f"- {f['volume']:,} pieces of feedback came in. The most mentioned theme was {top['label']} "
            f"({_pct(top['share'])} of feedback, {_pct(top['negative_share'])} negative)."
        )
        if f["movers"]:
            m = f["movers"][0]
            lines.append(
                f"- The biggest mover was {m['label']}: {_signed(m['change_pp'], ' points')} of share "
                f"against {f['previous_quarter']}."
            )
        vol = [a for a in f["alerts"] if a["kind"] == "volume"]
        if f["new_clusters"]:
            c = f["new_clusters"][0]
            lines.append(
                f"- A new issue was flagged on {c['first_day']}: {c['size']:,} comments about "
                f'"{c["name"]}", closest to {LABELS.get(c["nearest_theme"], c["nearest_theme"])}.'
            )
        if vol:
            a = vol[0]
            lines.append(
                f"- {a['label']} volume alert from {a['first_day']} to {a['last_day']}, peaking at "
                f"{a['peak_ratio']:.2f} times the baseline."
            )
        d = f["drivers_negative"][0]
        lines.append(
            f"- The strongest negative NPS driver is {d['label']}: comments that mention it score "
            f"{abs(d['coef']):.2f} points lower on the 0 to 10 scale."
        )
        return "\n".join(lines)


class ClaudeWriter:
    """Asks Claude for the summary bullets from the facts. Inject `client` in tests."""

    name = "claude"
    simulated = False

    def __init__(self, model: str = "claude-opus-5-5", client=None, max_tokens: int = 700):
        if client is None:
            try:
                import anthropic
            except ImportError as exc:  # pragma: no cover - depends on install extras
                raise RuntimeError("Install the Claude extra first: uv sync --extra claude") from exc
            client = anthropic.Anthropic()
        self.client = client
        self.model = model
        self.max_tokens = max_tokens
        self.usage = (0, 0)

    SYSTEM = (
        "You write the summary for a quarterly voice of customer report read by a product leader. Write 4 to 6 "
        "Markdown bullet points in plain English. Use only numbers that appear in the facts JSON, written the same "
        "way or as a percentage of a share. Quote customers only with text copied exactly from the quotes in the "
        "facts. Do not speculate about causes the facts do not show."
    )

    def write(self, f: dict) -> str:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=self.SYSTEM,
            messages=[{"role": "user", "content": f"<facts>\n{json.dumps(f, default=str)}\n</facts>"}],
        )
        self.usage = (resp.usage.input_tokens, resp.usage.output_tokens)
        return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text").strip()


def _allowed_numbers(f: dict) -> set[float]:
    out: set[float] = set()

    def walk(v):
        if isinstance(v, bool):
            return
        if isinstance(v, int | float):
            for x in (v, 100 * v, abs(v), abs(100 * v)):
                for dgt in (0, 1, 2):
                    out.add(round(float(x), dgt))
        elif isinstance(v, str):
            for m in NUM.findall(v):
                out.add(round(float(m.rstrip("%").replace(",", "")), 2))
        elif isinstance(v, dict):
            for x in v.values():
                walk(x)
        elif isinstance(v, list | tuple):
            for x in v:
                walk(x)

    walk(f)
    out.update({0.0, 1.0, 10.0})  # "0 to 10 scale"
    return out


def validate(text: str, f: dict, corpus: list[str]) -> list[str]:
    """Problems with a written summary: numbers not in the facts, quotes not in the data."""
    problems = []
    allowed = _allowed_numbers(f)
    # Dates and quarter names are checked as whole tokens so their digits are not read as numbers.
    stripped = re.sub(r"\b\d{4}-\d{2}-\d{2}\b|\b\d{4}Q[1-4]\b", "", text)
    for m in NUM.findall(stripped):
        v = float(m.rstrip("%").replace(",", ""))
        if not any(abs(v - a) < 0.051 for a in allowed):
            problems.append(f"number {m} is not in the computed facts")
    for d in re.findall(r"\b\d{4}-\d{2}-\d{2}\b|\b\d{4}Q[1-4]\b", text):
        if d not in json.dumps(f, default=str):
            problems.append(f"date {d} is not in the computed facts")
    norm = [re.sub(r"\s+", " ", c).lower() for c in corpus]
    blob = "\n".join(norm)
    names = {c["name"].lower() for c in f.get("new_clusters", [])} | {r["label"].lower() for r in f.get("themes", [])}
    for q in QUOTE.findall(text):
        qn = re.sub(r"\s+", " ", q).strip().lower()
        if len(qn.split()) >= 4 and qn not in blob and qn not in names:
            problems.append(f'quote "{q[:50]}" does not appear in the feedback')
    return problems


def render_markdown(f: dict, summary: str, writer: str, note: str = "") -> str:
    lines = [f"# {f['product']} insights report, {f['quarter']}", ""]
    lines.append(
        f"Built from {f['volume']:,} survey comments, app reviews, and support ticket summaries. Summary "
        f"written by: {writer}.{(' ' + note) if note else ''}"
    )
    lines += ["", "## Summary", "", summary, "", "## Headline numbers", ""]
    lines += ["| Metric | " + f["quarter"] + " | " + (f["previous_quarter"] or "") + " |", "|---|---|---|"]
    lines.append(f"| Feedback volume | {f['volume']:,} | {f['previous_volume'] or 0:,} |")
    lines.append(f"| NPS | {f['nps']:.1f} | {f['previous_nps'] if f['previous_nps'] is not None else '-'} |")
    lines.append(f"| Average app rating | {f['avg_stars']} | {f['previous_avg_stars'] or '-'} |")
    lines += [
        "",
        "## Top themes",
        "",
        "| Theme | Mentions | Share | Negative | Change vs last quarter |",
        "|---|---|---|---|---|",
    ]
    for r in f["themes"][:10]:
        lines.append(
            f"| {r['label']} | {r['count']:,} | {_pct(r['share'])} | {_pct(r['negative_share'])} | "
            f"{_signed(r.get('change_pp', 0), ' pts')} |"
        )
    lines += ["", "## Movers", ""]
    for m in f["movers"]:
        lines.append(
            f"- **{m['label']}**: {_pct(m['prev_share'])} to {_pct(m['share'])} of feedback "
            f"({_signed(m['change_pp'], ' points')})."
        )
    lines += ["", "## Emerging issues", ""]
    if not f["alerts"] and not f["new_clusters"]:
        lines.append("- None flagged this quarter.")
    for c in f["new_clusters"]:
        lines.append(
            f'- **New cluster** "{c["name"]}": first flagged {c["first_day"]}, {c["size"]:,} comments, '
            f"closest known theme {LABELS.get(c['nearest_theme'], c['nearest_theme'])}. Terms: "
            f"{', '.join(c['terms'][:6])}."
        )
    for a in f["alerts"]:
        stat = (
            f"peak {a['peak_ratio']:.2f} times baseline"
            if a["kind"] == "volume"
            else f"negative share up {_pct(a['peak_lift'])}"
        )
        lines.append(
            f"- **{a['label']}** ({a['kind'].replace('_', ' ')}): {a['first_day']} to {a['last_day']}, "
            f"{stat}, p = {a['p_value']:.2g}."
        )
    lines += ["", "## What customers said", ""]
    for t, qs in f["quotes"].items():
        lines.append(f"**{LABELS[t]}**")
        lines.append("")
        for q in qs:
            lines.append(f"> {q}")
            lines.append("")
    lines += [
        "## NPS drivers",
        "",
        f"Regression of the 0 to 10 score on theme mentions, all four quarters, {f['drivers_n']:,} survey responses.",
        "",
        "| Theme | Score change when mentioned | 95% CI |",
        "|---|---|---|",
    ]
    for d in f["drivers_negative"] + f["drivers_positive"]:
        lines.append(f"| {d['label']} | {d['coef']:+.2f} | {d['ci_low']:+.2f} to {d['ci_high']:+.2f} |")
    lines += ["", "Caveats:", ""] + [f"- {c}" for c in f["caveats"]]
    lines += ["", f"Data is synthetic. {_pct(f['simulated_llm_share'])} of texts were tagged by a simulated LLM tier."]
    return "\n".join(lines) + "\n"


def _inline(s: str) -> str:
    s = html.escape(s, quote=False)
    return re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)


def markdown_to_html(md: str, title: str) -> str:
    """A small converter for the subset this report uses: headings, bullets, quotes, tables, paragraphs."""
    out, i = [], 0
    rows = md.split("\n")
    while i < len(rows):
        r = rows[i]
        if r.startswith("# "):
            out.append(f"<h1>{_inline(r[2:])}</h1>")
        elif r.startswith("## "):
            out.append(f"<h2>{_inline(r[3:])}</h2>")
        elif r.startswith("- "):
            out.append("<ul>")
            while i < len(rows) and rows[i].startswith("- "):
                out.append(f"<li>{_inline(rows[i][2:])}</li>")
                i += 1
            out.append("</ul>")
            continue
        elif r.startswith("> "):
            out.append(f"<blockquote>{_inline(r[2:])}</blockquote>")
        elif r.startswith("|"):
            cells = [c.strip() for c in r.strip("|").split("|")]
            out.append(
                '<div class="scroll"><table><thead><tr>'
                + "".join(f"<th>{_inline(c)}</th>" for c in cells)
                + "</tr></thead><tbody>"
            )
            i += 2
            while i < len(rows) and rows[i].startswith("|"):
                cells = [c.strip() for c in rows[i].strip("|").split("|")]
                out.append("<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in cells) + "</tr>")
                i += 1
            out.append("</tbody></table></div>")
            continue
        elif r.strip():
            out.append(f"<p>{_inline(r)}</p>")
        i += 1
    css = (
        ":root{--bg:#fff;--text:#1c2330;--muted:#5d6878;--border:#dde2e9;--accent:#2f6fdb;color-scheme:light}"
        "@media (prefers-color-scheme:dark){:root{--bg:#11151c;--text:#e6eaf0;--muted:#9aa5b5;--border:#2a3340;"
        "--accent:#6c9cf0;color-scheme:dark}}"
        "body{margin:0 auto;max-width:860px;padding:24px 16px;background:var(--bg);color:var(--text);"
        "font:15px/1.55 system-ui,-apple-system,'Segoe UI',Roboto,sans-serif;overflow-wrap:anywhere}"
        "h1{font-size:24px}h2{font-size:17px;margin-top:28px;border-bottom:1px solid var(--border);padding-bottom:4px}"
        "table{border-collapse:collapse;width:100%;font-size:14px}th,td{text-align:left;padding:6px 8px;"
        "border-top:1px solid var(--border)}th{color:var(--muted)}.scroll{overflow-x:auto}"
        "blockquote{margin:6px 0;padding:4px 12px;border-left:3px solid var(--accent);color:var(--muted)}"
    )
    return (
        f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" '
        f'content="width=device-width, initial-scale=1"><title>{html.escape(title)}</title><style>{css}</style>'
        f"</head><body>{''.join(out)}</body></html>"
    )


def build_report(bench, quarter: str | None = None, writer=None, fmt: str = "md") -> dict:
    f = facts(bench, quarter)
    writer = writer or TemplateWriter()
    corpus = [x.text for x in bench.items]
    note = ""
    name = writer.name
    try:
        summary = writer.write(f)
        problems = validate(summary, f, corpus)
    except Exception as exc:  # a failed API call falls back the same way as a rejected summary
        summary, problems = "", [f"writer failed: {exc}"[:200]]
    if problems:
        summary = TemplateWriter().write(f)
        note = f"The {name} summary was rejected by the validator ({problems[0]}), so the template was used."
        name = "template (fallback)"
    md = render_markdown(f, summary, name, note)
    out = {"quarter": f["quarter"], "writer": name, "problems": problems, "markdown": md, "facts": f}
    if fmt == "html":
        out["html"] = markdown_to_html(md, f"{f['product']} insights {f['quarter']}")
    return out
